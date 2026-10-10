"""
Verify generated frontend builds during codegen and auto-fix with LLM (Cursor-style loop).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, Optional, Tuple

from app_builder.schemas.files import GeneratedFiles
from app_builder.services.file_writer import file_writer
from app_builder.services.runtime_paths import resolve_project_root
from llm_helper import get_llm_for_user

from api.app_creator.cra_npm_patch import (
    ensure_cra_build_env_file,
    patch_package_json_in_files,
    prepare_frontend_dir_on_disk,
)
from api.app_creator.pipeline_helpers import npm_build_timeout, npm_install_timeout

logger = logging.getLogger("app_builder")


def _resolve_frontend_dir(project_root: str) -> Optional[str]:
    frontend = os.path.join(project_root, "frontend")
    if os.path.isfile(os.path.join(frontend, "package.json")):
        return frontend
    client = os.path.join(frontend, "client")
    if os.path.isfile(os.path.join(client, "package.json")):
        return client
    if os.path.isfile(os.path.join(project_root, "package.json")):
        return project_root
    return None


def _capture_output(result: subprocess.CompletedProcess, cwd: str) -> str:
    parts = []
    if result.stdout:
        parts.append(result.stdout.decode("utf-8", errors="replace"))
    if result.stderr:
        parts.append(result.stderr.decode("utf-8", errors="replace"))
    text = "\n".join(parts).strip()
    if len(text) > 12000:
        text = text[-12000:]
    return text or f"npm failed in {cwd} (exit {result.returncode})"


def run_frontend_build(project_name: str, install: bool = True) -> Tuple[Optional[str], Optional[str], str]:
    """
    npm install + npm run build locally.
    Returns (static_dir, error_message, log_output).
    """
    project_root = resolve_project_root(project_name)
    if not os.path.isdir(project_root):
        return None, "Project not found on disk", ""

    frontend_dir = _resolve_frontend_dir(project_root)
    if not frontend_dir:
        return None, "No frontend/package.json found", ""

    from api.app_creator.vite_build import is_vite_frontend, prepare_vite_frontend_dir, run_vite_build

    if is_vite_frontend(frontend_dir):
        prepare_vite_frontend_dir(frontend_dir)
        return run_vite_build(frontend_dir, project_name, install=install)

    patched = prepare_frontend_dir_on_disk(frontend_dir)
    if patched:
        node_modules = os.path.join(frontend_dir, "node_modules")
        if os.path.isdir(node_modules):
            shutil.rmtree(node_modules, ignore_errors=True)
    logs: list[str] = []

    try:
        if install:
            logs.append("[build] npm install --legacy-peer-deps")
            r = subprocess.run(
                ["npm", "install", "--legacy-peer-deps"],
                cwd=frontend_dir,
                capture_output=True,
                timeout=npm_install_timeout(),
            )
            out = _capture_output(r, frontend_dir)
            logs.append(out)
            if r.returncode != 0:
                return None, f"npm install failed (exit {r.returncode}): {out[-800:]}", "\n".join(logs)

        build_env = os.environ.copy()
        build_env["PUBLIC_URL"] = f"/app/{project_name}"
        build_env["VITE_BASE_PATH"] = f"/app/{project_name}/"  # for Vite apps
        from api.app_creator.cra_npm_patch import CRA_BUILD_ENV
        for key, val in CRA_BUILD_ENV.items():
            build_env.setdefault(key, val)

        logs.append("[build] npm run build")
        r = subprocess.run(
            ["npm", "run", "build"],
            cwd=frontend_dir,
            capture_output=True,
            timeout=npm_build_timeout(),
            env=build_env,
        )
        out = _capture_output(r, frontend_dir)
        logs.append(out)
        if r.returncode != 0:
            return None, f"npm run build failed (exit {r.returncode}): {out[-1200:]}", "\n".join(logs)

        for sub in ("dist", "build"):
            path = os.path.join(frontend_dir, sub)
            if os.path.isdir(path):
                logs.append(f"[build] OK → {sub}/")
                return path, None, "\n".join(logs)
        return None, "Build completed but dist/build not found", "\n".join(logs)
    except subprocess.TimeoutExpired:
        return None, "Build timed out", "\n".join(logs)
    except FileNotFoundError:
        return None, "npm not found — install Node.js on the server", "\n".join(logs)
    except Exception as exc:
        return None, str(exc), "\n".join(logs)


def _pick_files_for_fix(files: Dict[str, str], build_log: str) -> Dict[str, str]:
    """Select files most likely relevant to the build error."""
    priority = []
    if "package.json" in build_log or "ajv" in build_log.lower():
        priority.append("frontend/package.json")
    for path in sorted(files.keys()):
        if path in priority:
            continue
        if not path.startswith("frontend/"):
            continue
        if "node_modules" in path:
            continue
        if path.endswith((".js", ".jsx", ".ts", ".tsx", ".css", ".json", ".env")):
            priority.append(path)
    out: Dict[str, str] = {}
    for path in priority[:20]:
        if path in files:
            out[path] = files[path]
    if "frontend/package.json" not in out and "frontend/package.json" in files:
        out["frontend/package.json"] = files["frontend/package.json"]
    return out


def extract_error_files(build_output: str, project_root: str) -> Dict[str, str]:
    """
    Parse build error output for file paths and read those files fully from disk.
    Handles TypeScript errors, ESLint errors, and Vite build errors.

    Patterns matched:
      src/pages/Foo.tsx(45,12)   — TypeScript
      src/pages/Foo.tsx:45:12    — ESLint / Vite
      src/pages/Foo.tsx           — plain reference

    Returns {relative_key: content} where relative_key matches the in-memory files
    dict convention, e.g. 'frontend/src/pages/Foo.tsx'.
    """
    result: Dict[str, str] = {}
    # Match src/… paths for .ts/.tsx/.js/.jsx files
    pattern = re.compile(r'\b(src/[^\s:()\'\"]+\.(?:tsx?|jsx?))')
    found_paths: set = set()
    for m in pattern.finditer(build_output):
        found_paths.add(m.group(1))

    for rel in found_paths:
        # Try project_root/frontend/src/… first (most common layout)
        disk_path = os.path.join(project_root, "frontend", rel)
        if not os.path.isfile(disk_path):
            # Fall back to project_root/src/…
            disk_path = os.path.join(project_root, rel)
        if os.path.isfile(disk_path):
            try:
                with open(disk_path, "r", encoding="utf-8") as fh:
                    content = fh.read()
                full_rel = f"frontend/{rel}"
                result[full_rel] = content
            except Exception:
                pass
    return result


_DEFAULT_EXPORT_LINE = "\n// Dual export — supports both named and default import styles\nexport default apiFetch;\n"

def _deterministic_build_fix(files: Dict[str, str], build_log: str, project_root: str) -> Dict[str, str]:
    """
    Apply deterministic fixes for known recurring build errors BEFORE calling the LLM.
    Returns updated files dict (also patches on disk).
    """
    updated = dict(files)
    log_lower = build_log.lower()

    # ── Fix: "default" is not exported by "src/api/client.ts" ────────────────
    # LLMs always generate: import apiFetch from '../api/client'  (default import)
    # But client.ts only has named export.  Solution: add default export to client.ts.
    if '"default" is not exported by' in build_log and "api/client" in build_log:
        client_key = "frontend/src/api/client.ts"
        client_path = os.path.join(project_root, "frontend", "src", "api", "client.ts")

        # Patch in-memory files
        if client_key in updated and "export default apiFetch" not in updated[client_key]:
            updated[client_key] = updated[client_key].rstrip() + _DEFAULT_EXPORT_LINE
            logger.info("[build-fix] Patched client.ts — added export default apiFetch")

        # Patch file on disk (build reads from disk)
        if os.path.isfile(client_path):
            content = open(client_path, "r", encoding="utf-8").read()
            if "export default apiFetch" not in content:
                with open(client_path, "a", encoding="utf-8") as f:
                    f.write(_DEFAULT_EXPORT_LINE)
                logger.info("[build-fix] Patched client.ts on disk")

    return updated


async def _llm_fix_build(
    files: Dict[str, str],
    build_log: str,
    user_email: Optional[str],
    error_files: Optional[Dict[str, str]] = None,
    failed_attempts: Optional[list] = None,
) -> Dict[str, str]:
    """Ask LLM to patch files to fix the build log.

    Args:
        files: full in-memory project file dict
        build_log: combined stdout/stderr from the failed build
        user_email: used to select the right LLM
        error_files: {path: content} of files specifically mentioned in the error output
        failed_attempts: list of (error_before_fix, error_after_fix) tuples from prior rounds
    """
    subset = _pick_files_for_fix(files, build_log)
    snippets = []
    for path, content in subset.items():
        # Skip files already included in error_files to avoid duplication
        if error_files and path in error_files:
            continue
        file_limit = 16000 if len(content) < 30000 else 16000
        snippets.append(f"--- {path} ---\n{content[:file_limit]}")

    # Build the error-specific files section (highest priority)
    error_files_section = ""
    if error_files:
        ef_snippets = []
        for path, content in error_files.items():
            ef_snippets.append(f"--- {path} (CONTAINS ERROR) ---\n{content}")
        error_files_section = "\n## FILES WITH ERRORS (read these carefully)\n" + "\n\n".join(ef_snippets)

    # Build the failed-attempts section so LLM doesn't repeat what didn't work
    failed_section = ""
    if failed_attempts:
        lines = []
        for i, (old_err, new_err) in enumerate(failed_attempts, 1):
            lines.append(
                f"### Attempt {i} — did NOT fix the build\n"
                f"Error before fix:\n```\n{old_err[-800:]}\n```\n"
                f"Error after fix (still broken):\n```\n{new_err[-800:]}\n```"
            )
        failed_section = "\n## PREVIOUS FIX ATTEMPTS THAT FAILED (do not repeat these)\n" + "\n\n".join(lines)

    prompt = f"""You are fixing a TypeScript/React build error. Be surgical — only change what is broken.
The bundler is Vite (NOT Create React App / react-scripts). Do NOT suggest craco, ajv-keywords, or NODE_OPTIONS hacks.

## BUILD ERROR
```
{build_log[-6500:]}
```
{error_files_section}

## ALL PROJECT FILES (for context)
{chr(10).join(snippets)}
{failed_section}

Rules:
1. Fix TypeScript type errors, missing imports, or wrong export styles shown in the log.
2. If the error mentions "default is not exported" for api/client: add `export default apiFetch;` at the end of client.ts.
3. If the error mentions a missing module: add the correct named import, do NOT add CRA/webpack dependencies.
4. If vite.config.ts is missing: create one with `defineConfig({{ plugins: [react()] }})`.
5. Fix syntax/import errors in source files shown in the log.
6. Do NOT rewrite files that don't have errors — only output files that must change.
7. Return ONLY changed files as valid JSON: {{"files": {{"path/to/file": "full file content"}}}}
8. Do not truncate files — return complete file contents for each changed path."""

    llm = get_llm_for_user(user_email, temperature=0.15)
    resp = await llm.ainvoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(text, list):
        text = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in text
        )

    match = re.search(r"\{[\s\S]*\"files\"[\s\S]*\}", text)
    if not match:
        return files
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
            data = json.loads(repair_json(match.group(0)))
        except Exception:
            return files

    updated = dict(files)
    for path, content in (data.get("files") or {}).items():
        if isinstance(content, str) and content.strip():
            updated[path] = content
    from app_builder.services.code_post_process import _is_vite_project
    if not _is_vite_project(updated):
        patch_package_json_in_files(updated)
        ensure_cra_build_env_file(updated)
    else:
        updated.pop("frontend/craco.config.js", None)
    return updated


async def verify_build_and_fix(
    project_name: str,
    files: Dict[str, str],
    user_email: Optional[str],
    on_event: Optional[Callable[[dict], Any]] = None,
    max_attempts: int = 3,
) -> Tuple[Dict[str, str], bool, str]:
    """
    Write files, run npm build, auto-fix with LLM on failure (up to max_attempts).
    Returns (updated_files, success, last_log).
    """
    async def emit(event: str, message: str, **extra):
        payload = {"event": event, "message": message, **extra}
        if on_event:
            await on_event(payload)

    # Log build mode so it's visible in server logs
    try:
        from api.app_creator.e2b_sandbox_build import is_e2b_enabled
        build_mode = "e2b-sandbox" if is_e2b_enabled() else "local"
    except Exception:
        build_mode = "local"
    logger.info("[build_verify] project=%s build_mode=%s", project_name, build_mode)
    await emit("agent_progress", f"Build mode: {build_mode}", agent="build_verify_agent")

    from app_builder.services.code_post_process import _is_vite_project
    if not _is_vite_project(files):
        patch_package_json_in_files(files)
        ensure_cra_build_env_file(files)
    else:
        files.pop("frontend/craco.config.js", None)

    file_writer(project_name, GeneratedFiles(files=files))

    last_log = ""
    failed_attempts: list = []          # list of (error_before_fix, error_after_fix)
    error_before_fix: str = ""          # captured just before each fix round

    for attempt in range(1, max_attempts + 1):
        await emit(
            "build_start",
            f"Running npm install + build (attempt {attempt}/{max_attempts})...",
            attempt=attempt,
        )
        static_dir, err, log = run_frontend_build(project_name, install=True)
        last_log = log or err or ""
        if log:
            for line in log.splitlines()[-30:]:
                if line.strip():
                    await emit("build_log", line.strip())

        if static_dir and not err:
            await emit("build_success", "Frontend build verified successfully.")
            return files, True, last_log

        current_error = last_log

        # Record the outcome of the previous fix attempt so the LLM can learn from it
        if attempt > 1 and error_before_fix:
            failed_attempts.append((error_before_fix, current_error))

        await emit(
            "build_failed",
            err or "Build failed",
            log=last_log[-2000:] if last_log else "",
            attempt=attempt,
        )

        if attempt >= max_attempts:
            break

        await emit(
            "build_fix_attempt",
            f"Build failed — AI is fixing errors (attempt {attempt}/{max_attempts})...",
            attempt=attempt,
        )
        try:
            project_root = resolve_project_root(project_name)
            # 1. Deterministic fixes first (fast, reliable for known patterns)
            files = _deterministic_build_fix(files, current_error, project_root)
            # 2. Extract the specific files the error references so the LLM can read them fully
            error_files = extract_error_files(current_error, project_root)
            if error_files:
                await emit("build_fix_attempt", f"Error localised to: {', '.join(error_files.keys())}")
            # 3. LLM fixes — pass error-specific files and history of failed attempts
            error_before_fix = current_error
            files = await _llm_fix_build(
                files,
                current_error,
                user_email,
                error_files=error_files or None,
                failed_attempts=failed_attempts or None,
            )
            file_writer(project_name, GeneratedFiles(files=files))
        except Exception as fix_err:
            logger.warning("[build_verify] LLM fix failed: %s", fix_err)
            await emit("build_fix_attempt", f"Auto-fix error: {fix_err}")

    return files, False, last_log
