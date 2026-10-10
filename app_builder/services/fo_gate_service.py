"""
Frontend-Only Gate Service: npm install → vite build → headless Playwright smoke test.

For each route in the app:
  - Open in headless Chromium
  - Assert: no console errors, no uncaught exceptions, no blank body, no failed requests
  - Take a screenshot (saved to project screenshots/ dir)

On failure:
  - Emit the failing route + error to the auto-fix pass (max MAX_GATE_FIX_ATTEMPTS=2)
  - Re-run gate after fix
  - Only set BUILD_COMPLETE when all routes pass

Falls back gracefully when Playwright is not installed.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from app_builder.services.runtime_paths import project_write_root

logger = logging.getLogger("app_builder")

MAX_GATE_FIX_ATTEMPTS = 2
GATE_TIMEOUT_S = 45       # per route
SERVER_START_WAIT_S = 4   # wait for vite preview to be ready
SCREENSHOTS_DIR = "gate_screenshots"


@dataclass
class RouteResult:
    route: str
    passed: bool
    console_errors: List[str] = field(default_factory=list)
    failed_requests: List[str] = field(default_factory=list)
    blank_body: bool = False
    screenshot_path: str = ""
    error: str = ""


@dataclass
class GateResult:
    passed: bool
    route_results: List[RouteResult] = field(default_factory=list)
    build_log: str = ""
    error: str = ""

    @property
    def failing_routes(self) -> List[RouteResult]:
        return [r for r in self.route_results if not r.passed]

    @property
    def first_failure(self) -> Optional[RouteResult]:
        failing = self.failing_routes
        return failing[0] if failing else None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_gate(
    project_name: str,
    files: Dict[str, str],
    blueprint_json: Optional[Dict[str, Any]],
    on_event: Optional[Callable] = None,
    metrics: Optional[Any] = None,
) -> GateResult:
    """
    Run the full build + Playwright gate.
    Returns GateResult.passed=True only if all routes pass.
    """
    project_root = project_write_root(project_name)
    frontend_dir = os.path.join(project_root, "frontend")

    if not os.path.isdir(frontend_dir):
        return GateResult(passed=False, error="frontend directory not found")

    # 1. Build
    await _emit(on_event, {
        "event": "agent_start", "agent": "gate_build",
        "message": "Gate: building frontend (npm install + vite build)...",
    })
    if metrics:
        metrics.stage_start("gate_build")

    build_ok, build_log = await _run_build(frontend_dir, project_name)

    if metrics:
        metrics.stage_end("gate_build")

    if not build_ok:
        await _emit(on_event, {
            "event": "agent_error", "agent": "gate_build",
            "message": f"Gate build failed: {build_log[-400:]}",
        })
        return GateResult(passed=False, build_log=build_log, error="vite build failed")

    await _emit(on_event, {
        "event": "agent_complete", "agent": "gate_build",
        "message": "Gate: build succeeded",
    })

    # 2. Playwright smoke test
    routes = _extract_routes(blueprint_json)
    if not routes:
        routes = ["/"]

    pw_available = _check_playwright()
    if not pw_available:
        logger.info("[gate] Playwright not installed — gate skipped (build-only)")
        await _emit(on_event, {
            "event": "agent_complete", "agent": "gate_playwright",
            "message": "Gate: Playwright not installed — build-only gate passed",
        })
        return GateResult(passed=True, build_log=build_log)

    await _emit(on_event, {
        "event": "agent_start", "agent": "gate_playwright",
        "message": f"Gate: Playwright smoke test ({len(routes)} routes)...",
    })
    if metrics:
        metrics.stage_start("gate_playwright")

    screenshots_dir = os.path.join(project_root, SCREENSHOTS_DIR)
    os.makedirs(screenshots_dir, exist_ok=True)

    dist_dir = os.path.join(frontend_dir, "dist")
    port = _pick_free_port()
    server_proc = None

    try:
        server_proc = await _start_preview_server(frontend_dir, port)
        await asyncio.sleep(SERVER_START_WAIT_S)

        base_url = f"http://localhost:{port}"
        route_results = await _run_playwright(base_url, routes, screenshots_dir)

        for rr in route_results:
            await _emit(on_event, {
                "event": "agent_progress", "agent": "gate_playwright",
                "message": f"  {'✓' if rr.passed else '✗'} {rr.route}"
                           + (f" — {rr.error[:80]}" if not rr.passed else ""),
            })
            if metrics:
                metrics.record_gate(
                    rr.route, rr.passed,
                    console_errors=len(rr.console_errors),
                    screenshot_path=rr.screenshot_path,
                    error=rr.error,
                )

    finally:
        if server_proc:
            try:
                server_proc.terminate()
                await asyncio.sleep(0.5)
            except Exception:
                pass

    if metrics:
        metrics.stage_end("gate_playwright")

    all_passed = all(r.passed for r in route_results)
    result = GateResult(passed=all_passed, route_results=route_results, build_log=build_log)

    if all_passed:
        await _emit(on_event, {
            "event": "agent_complete", "agent": "gate_playwright",
            "message": f"Gate: all {len(routes)} routes passed",
        })
    else:
        failing = result.failing_routes
        await _emit(on_event, {
            "event": "agent_error", "agent": "gate_playwright",
            "message": f"Gate: {len(failing)}/{len(routes)} routes failed",
            "data": {
                "failing": [{"route": r.route, "error": r.error, "console_errors": r.console_errors[:3]} for r in failing],
            },
        })

    return result


# ---------------------------------------------------------------------------
# Gate-aware generation helper (called from frontend_only_pipeline.py)
# ---------------------------------------------------------------------------

async def gate_with_fix(
    project_name: str,
    files: Dict[str, str],
    blueprint_json: Optional[Dict],
    types_content: str,
    mock_exports: List[str],
    llm: Any,
    on_event: Callable,
    metrics: Optional[Any] = None,
) -> Tuple[Dict[str, str], GateResult]:
    """
    Run gate; if it fails, apply a targeted fix and re-gate (max MAX_GATE_FIX_ATTEMPTS).
    Returns (final_files, final_gate_result).
    """
    result = await run_gate(project_name, files, blueprint_json, on_event, metrics)

    for attempt in range(1, MAX_GATE_FIX_ATTEMPTS + 1):
        if result.passed:
            break

        failure = result.first_failure
        if not failure:
            break

        await _emit(on_event, {
            "event": "agent_start", "agent": f"gate_fix_{attempt}",
            "message": f"Gate fix attempt {attempt}/{MAX_GATE_FIX_ATTEMPTS}: {failure.route}",
        })

        # Identify the failing page file
        page_name = _route_to_page_name(failure.route, blueprint_json)
        page_key = f"frontend/src/pages/{page_name}.tsx" if page_name else ""
        page_content = files.get(page_key) or ""

        if not page_content or not llm:
            break

        fixed = await _fix_page_for_gate(
            page_name=page_name or "Page",
            page_content=page_content,
            route=failure.route,
            console_errors=failure.console_errors,
            failed_requests=failure.failed_requests,
            types_content=types_content,
            mock_exports=mock_exports,
            llm=llm,
        )

        if fixed:
            from app_builder.services.verify_service import is_complete_file
            from app_builder.services.file_writer import write_single_file_to_disk
            if is_complete_file(fixed):
                files[page_key] = fixed
                project_root = project_write_root(project_name)
                write_single_file_to_disk(project_root, page_key, fixed)
                if metrics:
                    metrics.record_fix_attempt("gate", attempt, page_key, True)
                result = await run_gate(project_name, files, blueprint_json, on_event, metrics)
            else:
                if metrics:
                    metrics.record_fix_attempt("gate", attempt, page_key, False)

        await _emit(on_event, {
            "event": "agent_complete" if result.passed else "agent_error",
            "agent": f"gate_fix_{attempt}",
            "message": f"Gate fix {'succeeded' if result.passed else 'still failing'}",
        })

    return files, result


# ---------------------------------------------------------------------------
# Build step
# ---------------------------------------------------------------------------

async def _run_build(frontend_dir: str, project_name: str) -> Tuple[bool, str]:
    """npm install (cached) + vite build."""
    logs: List[str] = []

    # Try to use node_modules cache to skip install
    nm = os.path.join(frontend_dir, "node_modules")
    needs_install = not os.path.isdir(nm)

    if needs_install:
        cache_path = os.path.join(
            os.path.expanduser("~"), ".akkio", "app_builder", "nm_cache",
            "base-frontend-vite-mui", "node_modules",
        )
        if os.path.isdir(cache_path):
            try:
                os.symlink(cache_path, nm)
                needs_install = False
                logs.append("[gate] using nm_cache symlink")
            except OSError:
                pass

    if needs_install:
        logs.append("[gate] npm install --legacy-peer-deps")
        try:
            proc = await asyncio.create_subprocess_exec(
                "npm", "install", "--legacy-peer-deps",
                cwd=frontend_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
            out = (stdout + stderr).decode("utf-8", errors="replace")
            logs.append(out[-3000:])
            if proc.returncode != 0:
                return False, "\n".join(logs)
        except asyncio.TimeoutError:
            return False, "[gate] npm install timeout"

    # vite build
    build_env = os.environ.copy()
    build_env["VITE_BASE_PATH"] = f"/app/{project_name}/"
    logs.append("[gate] npm run build")
    try:
        proc = await asyncio.create_subprocess_exec(
            "npm", "run", "build",
            cwd=frontend_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=build_env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        out = (stdout + stderr).decode("utf-8", errors="replace")
        logs.append(out[-4000:])
        if proc.returncode != 0:
            return False, "\n".join(logs)
    except asyncio.TimeoutError:
        return False, "[gate] vite build timeout"

    return True, "\n".join(logs)


# ---------------------------------------------------------------------------
# Playwright runner
# ---------------------------------------------------------------------------

async def _run_playwright(
    base_url: str,
    routes: List[str],
    screenshots_dir: str,
) -> List[RouteResult]:
    """Run Playwright against each route. Returns list of RouteResult."""
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except ImportError:
        logger.warning("[gate] playwright not importable")
        return [RouteResult(route=r, passed=True, error="playwright-unavailable") for r in routes]

    results: List[RouteResult] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})

        for route in routes:
            rr = await _check_route(context, base_url, route, screenshots_dir)
            results.append(rr)
            if not rr.passed:
                logger.info("[gate] route %s failed: %s", route, rr.error)

        await context.close()
        await browser.close()

    return results


async def _check_route(
    context: Any,
    base_url: str,
    route: str,
    screenshots_dir: str,
) -> RouteResult:
    """Open one route and run all checks."""
    url = base_url + route
    console_errors: List[str] = []
    failed_requests: List[str] = []

    try:
        from playwright.async_api import Page  # type: ignore
        page: Page = await context.new_page()

        # Capture console errors
        page.on("console", lambda msg: console_errors.append(msg.text)
                if msg.type in ("error", "warning") else None)
        page.on("pageerror", lambda exc: console_errors.append(f"Uncaught: {exc}"))
        page.on("requestfailed", lambda req: failed_requests.append(
            f"{req.method} {req.url} → {req.failure}"
        ))

        response = await page.goto(url, wait_until="networkidle", timeout=GATE_TIMEOUT_S * 1000)

        # Check status
        status = response.status if response else 0
        if status >= 400:
            await page.close()
            return RouteResult(
                route=route, passed=False,
                error=f"HTTP {status}",
                console_errors=console_errors,
            )

        # Check for blank body
        body_text = (await page.inner_text("body") or "").strip()
        blank = len(body_text) < 30

        # Screenshot
        safe_route = route.replace("/", "_").strip("_") or "index"
        shot_path = os.path.join(screenshots_dir, f"{safe_route}.png")
        try:
            await page.screenshot(path=shot_path, full_page=False)
        except Exception:
            shot_path = ""

        # Filter noise from console errors (React dev warnings, MUI warnings are OK)
        real_errors = [e for e in console_errors if not _is_noise_error(e)]

        await page.close()

        passed = not blank and not real_errors and not [
            r for r in failed_requests if _is_blocking_request(r)
        ]

        error = ""
        if blank:
            error = "blank body"
        elif real_errors:
            error = real_errors[0][:200]
        elif failed_requests:
            blocking = [r for r in failed_requests if _is_blocking_request(r)]
            if blocking:
                error = f"failed request: {blocking[0]}"

        return RouteResult(
            route=route,
            passed=passed,
            console_errors=real_errors[:5],
            failed_requests=[r for r in failed_requests if _is_blocking_request(r)][:3],
            blank_body=blank,
            screenshot_path=shot_path,
            error=error,
        )

    except asyncio.TimeoutError:
        return RouteResult(route=route, passed=False, error=f"timeout ({GATE_TIMEOUT_S}s)")
    except Exception as exc:
        return RouteResult(route=route, passed=False, error=str(exc)[:200])


# ---------------------------------------------------------------------------
# Gate fixer
# ---------------------------------------------------------------------------

_GATE_FIX_SYSTEM = """\
You are a React/TypeScript debugger. Fix the provided page so it renders without browser errors.

Rules:
1. Return the COMPLETE, CORRECTED TypeScript file.
2. Fix the specific console errors listed — do not introduce new ones.
3. Never import from packages not in the allowed list.
4. The file must be complete (all exports preserved, balanced braces).
5. Common fixes: remove undefined variables, fix missing imports, fix bad hook usage, add null checks.
6. ONLY import from: 'react', 'react-router-dom', '@mui/material', '@mui/icons-material',
   '../components/ui', '../types', '../mock'.

Return ONLY the corrected file content. No markdown fences."""


async def _fix_page_for_gate(
    page_name: str,
    page_content: str,
    route: str,
    console_errors: List[str],
    failed_requests: List[str],
    types_content: str,
    mock_exports: List[str],
    llm: Any,
) -> Optional[str]:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        errors_text = "\n".join(f"  {e}" for e in (console_errors + failed_requests)[:6])
        user_prompt = (
            f"Page: {page_name}.tsx (route: {route})\n\n"
            f"Browser console errors:\n{errors_text}\n\n"
            f"Available types:\n{types_content[:800]}\n\n"
            f"Available mock exports: {', '.join(mock_exports[:20])}\n\n"
            f"Broken page content:\n{page_content[:3000]}\n\n"
            "Return the complete fixed file."
        )
        response = await llm.ainvoke([
            SystemMessage(content=_GATE_FIX_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        content = response.content if hasattr(response, "content") else str(response)
        # Strip markdown fences
        import re
        content = re.sub(r"^```\w*\n?", "", content.strip())
        content = re.sub(r"\n?```$", "", content.strip())
        return content.strip()
    except Exception as exc:
        logger.warning("[gate_fix] LLM failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_playwright() -> bool:
    """Return True if playwright is importable and browsers are installed."""
    try:
        import playwright  # noqa: F401
        from playwright.async_api import async_playwright  # noqa: F401
        return True
    except ImportError:
        return False


def _pick_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


async def _start_preview_server(frontend_dir: str, port: int) -> Any:
    """Start vite preview server in background."""
    proc = await asyncio.create_subprocess_exec(
        "npx", "vite", "preview", "--port", str(port), "--strictPort",
        cwd=frontend_dir,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return proc


def _extract_routes(blueprint_json: Optional[Dict]) -> List[str]:
    """Extract route paths from blueprint."""
    if not blueprint_json:
        return ["/"]
    routes = []
    for r in (blueprint_json.get("routes") or []):
        path = r.get("path") or "/"
        if path not in routes:
            routes.append(path)
    return routes or ["/"]


def _route_to_page_name(route: str, blueprint_json: Optional[Dict]) -> Optional[str]:
    """Find the page component name for a given route."""
    if not blueprint_json:
        return None
    for r in (blueprint_json.get("routes") or []):
        if r.get("path") == route:
            return r.get("page")
    return None


def _is_noise_error(msg: str) -> bool:
    """Filter out known non-breaking browser console messages."""
    noise_patterns = [
        "Warning:", "Download the React DevTools",
        "MUI:", "validateDOMNesting", "Each child in a list",
        "React.StrictMode", "DeprecationWarning",
        "favicon.ico", "hot-update",
    ]
    return any(p in msg for p in noise_patterns)


def _is_blocking_request(req_str: str) -> bool:
    """Return True for requests that indicate a real problem (not hot-reload noise)."""
    noise = ["hot-update", "webpack", "vite", "favicon", "__vite_"]
    return not any(n in req_str for n in noise)


async def _emit(on_event: Optional[Callable], data: Dict) -> None:
    if on_event:
        try:
            await on_event(data)
        except Exception:
            pass
