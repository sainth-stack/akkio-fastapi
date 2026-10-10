"""
Frontend-only pipeline — Lovable-style, contract-first React/TypeScript/MUI SPA.

All domains NOT in STATIC_DOMAINS are routed here.  No backend, no SQLite.

Stage order (from blueprint.file_order):
  S1  types.ts + mock/index.ts          (LLM x2)
  S2  theme/tokens.ts + App.tsx          (deterministic)
  S3  shared components                  (LLM, may produce 0 files)
  S4  pages                              (LLM x N, parallel, max_concurrency=3)
  S5  App.tsx/router/nav (final)         (deterministic, overwrites S2 stub)
  S6  final verify + save                (no LLM)

Each stage:
  - Emits progress events via on_event()
  - Writes files to disk incrementally
  - Verifies with tsc + import check
  - Retries broken files up to MAX_FIX_ATTEMPTS=3 via fix_file()
  - Stores intermediate artifacts for resume support
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("app_builder")

_TEMPLATE_NAME = "base-frontend-vite-mui"
_TEMPLATES_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates"
)
_TEMPLATE_DIR = os.path.join(_TEMPLATES_BASE, _TEMPLATE_NAME)

MAX_FIX_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# Public async entry point
# ---------------------------------------------------------------------------

async def run_pipeline(
    *,
    requirement: str,
    plan_json: Optional[Dict[str, Any]],   # { prd, uiux, design_tokens, blueprint }
    project_name: str,
    llm: Any,
    on_event: Callable[[Dict], Coroutine],  # async fn → sends WS events
    resume_stage: Optional[str] = None,
    stored_files: Optional[Dict[str, str]] = None,
    app_id: Optional[str] = None,
    user_email: Optional[str] = None,
    uid: Optional[int] = None,
) -> Dict[str, str]:
    """
    Run the full frontend-only generation pipeline.

    Returns the complete file dict on success, raises on unrecoverable failure.
    Intermediate files are written to disk after every verified stage.
    """
    blueprint_json = (plan_json or {}).get("blueprint") or {}
    uiux_json = (plan_json or {}).get("uiux")
    design_tokens_json = (plan_json or {}).get("design_tokens")
    prd_json = (plan_json or {}).get("prd")

    # Derive legacy Blueprint dict for deterministic generator
    legacy_bp = _plan_json_to_legacy_blueprint(blueprint_json, design_tokens_json, requirement)

    # Accumulated files dict
    files: Dict[str, str] = stored_files.copy() if stored_files else {}

    # Metrics tracking
    try:
        from app_builder.services.metrics_service import GenerationMetrics
        metrics: Any = GenerationMetrics(app_id=app_id, project_name=project_name)
    except ImportError:
        metrics = None

    # Stage order
    stages = ["S1", "S2", "S3", "S4", "S5", "S6"]
    skip_before = stages.index(resume_stage) if resume_stage in stages else 0

    total = len(stages)

    # ── S1: Types + Mock ─────────────────────────────────────────────────────
    if skip_before <= 0:
        if metrics: metrics.stage_start("S1")
        await _emit_stage_start(on_event, "S1", "Types & Mock Data", 1, total)
        s1_files = await _run_s1(blueprint_json, prd_json, design_tokens_json, llm, on_event)
        files.update(_drop_frozen(s1_files))
        await _verify_and_fix_stage(
            "S1", project_name, files, s1_files, llm, on_event,
            types_content=s1_files.get("frontend/src/types/index.ts", ""),
            mock_exports=_extract_export_names(s1_files.get("frontend/src/mock/index.ts", "")),
            metrics=metrics,
        )
        await _emit_stage_done(on_event, "S1", "Types & Mock Data", s1_files)
        if metrics: metrics.stage_end("S1")

    # ── S2: Theme + scaffold layout ───────────────────────────────────────────
    if skip_before <= 1:
        if metrics: metrics.stage_start("S2")
        await _emit_stage_start(on_event, "S2", "Theme & Layout", 2, total)
        s2_files = _run_s2_deterministic(legacy_bp, design_tokens_json)
        # Load template scaffold (frozen infrastructure files)
        template_files = _load_template_files()
        merged = {**template_files, **files, **s2_files}
        # Always restore frozen template infrastructure paths (never allow LLM overwrite)
        for frozen_path in _FROZEN_FO_PATHS:
            if frozen_path in template_files:
                merged[frozen_path] = template_files[frozen_path]
        files = merged
        await _emit_stage_done(on_event, "S2", "Theme & Layout", s2_files)
        if metrics: metrics.stage_end("S2")

    # ── S3: Shared domain components ─────────────────────────────────────────
    if skip_before <= 2:
        if metrics: metrics.stage_start("S3")
        await _emit_stage_start(on_event, "S3", "Shared Components", 3, total)
        types_content = files.get("frontend/src/types/index.ts", "")
        s3_files = await _run_s3(blueprint_json, types_content, llm, on_event)
        if s3_files:
            files.update(_drop_frozen(s3_files))
            await _verify_and_fix_stage(
                "S3", project_name, files, s3_files, llm, on_event,
                types_content=types_content,
                mock_exports=_extract_export_names(files.get("frontend/src/mock/index.ts", "")),
                metrics=metrics,
            )
        await _emit_stage_done(on_event, "S3", "Shared Components", s3_files)
        if metrics: metrics.stage_end("S3")

    # ── S4: Pages (parallel) ─────────────────────────────────────────────────
    if skip_before <= 3:
        if metrics: metrics.stage_start("S4")
        await _emit_stage_start(on_event, "S4", "Pages", 4, total)
        types_content = files.get("frontend/src/types/index.ts", "")
        mock_exports = _extract_export_names(files.get("frontend/src/mock/index.ts", ""))
        s4_files = await _run_s4_pages(
            blueprint_json, uiux_json, design_tokens_json,
            types_content, mock_exports, llm, on_event,
        )
        # Deterministic post-processing: fix common chart prop mistakes before tsc verify
        try:
            from app_builder.services.code_post_process import fix_frontend_only_pages
            s4_files = fix_frontend_only_pages(s4_files)
        except Exception as _pp_err:
            logger.warning("[pipeline/S4] post-process skipped: %s", _pp_err)
        files.update(_drop_frozen(s4_files))
        await _verify_and_fix_stage(
            "S4", project_name, files, s4_files, llm, on_event,
            types_content=types_content,
            mock_exports=mock_exports,
            metrics=metrics,
        )
        await _emit_stage_done(on_event, "S4", "Pages", s4_files)
        if metrics: metrics.stage_end("S4")

    # ── S5: Deterministic App.tsx / router / nav ─────────────────────────────
    if skip_before <= 4:
        if metrics: metrics.stage_start("S5")
        await _emit_stage_start(on_event, "S5", "Router & Nav", 5, total)
        s5_files = _run_s5_deterministic(legacy_bp, blueprint_json, generated_files=files)
        files.update(_drop_frozen(s5_files))
        await _emit_stage_done(on_event, "S5", "Router & Nav", s5_files)
        if metrics: metrics.stage_end("S5")

    # ── S6: Final verify + gate ───────────────────────────────────────────────
    if metrics: metrics.stage_start("S6")
    await _emit_stage_start(on_event, "S6", "Final Verification", 6, total)
    await _run_s6_verify(project_name, files, on_event)

    # ── Optional Playwright gate (frontend_only) ──────────────────────────────
    try:
        from app_builder.services.fo_gate_service import gate_with_fix
        types_content = files.get("frontend/src/types/index.ts", "")
        mock_exports = _extract_export_names(files.get("frontend/src/mock/index.ts", ""))
        files, gate_result = await gate_with_fix(
            project_name=project_name,
            files=files,
            blueprint_json=blueprint_json,
            types_content=types_content,
            mock_exports=mock_exports,
            llm=llm,
            on_event=on_event,
            metrics=metrics,
        )
        if metrics and gate_result:
            try:
                for rr in (gate_result.route_results or []):
                    metrics.record_gate(
                        route=rr.route,
                        passed=rr.passed,
                        console_errors=len(rr.console_errors or []),
                        screenshot_path=rr.screenshot_path or "",
                        error=rr.error or "",
                    )
            except Exception:
                pass
    except ImportError:
        pass  # gate service not installed — skip silently
    except Exception as _gate_err:
        logger.warning("[frontend_only] gate error (non-fatal): %s", _gate_err)

    await _emit_stage_done(on_event, "S6", "Final Verification", {})
    if metrics: metrics.stage_end("S6")

    # ── Persist metrics (best-effort) ─────────────────────────────────────────
    if metrics:
        try:
            from db.app_builder import get_db
            with get_db() as db:
                metrics.save_to_db(db, user_email=user_email, uid=uid)
        except Exception:
            pass

    return files


# ---------------------------------------------------------------------------
# Sync entry point (called from old synchronous paths / codegen_api)
# ---------------------------------------------------------------------------

def generate_frontend_only_app(
    requirement: str,
    prd: str = "",
    uiux: str = "",
    architecture: Optional[Dict[str, Any]] = None,
    design_tokens: Optional[Dict[str, Any]] = None,
    domain: str = "generic",
    plan_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Sync wrapper: runs the pipeline in a new event loop.
    Used by the old fullstack_app_generator.py routing path.
    """
    import asyncio

    # Build minimal plan_json if not provided
    if not plan_json:
        # Try to derive from architecture/prd/uiux
        bp = _derive_blueprint_from_artifacts(requirement, prd, uiux, architecture, design_tokens, domain)
        plan_json = {
            "blueprint": bp,
            "design_tokens": design_tokens,
            "prd": None,
            "uiux": None,
        }

    # Use a dummy on_event for the sync path
    files_result: Dict[str, str] = {}

    async def _run() -> None:
        nonlocal files_result
        try:
            # Build a minimal non-streaming LLM for the sync path
            from llm_helper import get_llm_for_user
            llm = get_llm_for_user("system@akkio.com", temperature=0.4, streaming=False)
        except Exception:
            llm = None

        if llm is None:
            # Fall back to deterministic-only generation
            legacy_bp = _plan_json_to_legacy_blueprint(
                plan_json.get("blueprint") or {},
                plan_json.get("design_tokens"),
                requirement,
            )
            s2 = _run_s2_deterministic(legacy_bp, plan_json.get("design_tokens"))
            s5 = _run_s5_deterministic(legacy_bp, plan_json.get("blueprint") or {}, generated_files=files_result)
            template = _load_template_files()
            files_result = {**template, **s2, **s5}
            return

        async def noop_event(_data: Dict) -> None:
            pass

        project_name = (
            (plan_json.get("blueprint") or {}).get("app_name") or requirement
        )[:50].strip().lower().replace(" ", "_")

        try:
            files_result = await run_pipeline(
                requirement=requirement,
                plan_json=plan_json,
                project_name=project_name,
                llm=llm,
                on_event=noop_event,
            )
        except Exception as exc:
            logger.warning("[frontend_only] sync run_pipeline failed: %s", exc)
            # Fallback to deterministic
            legacy_bp = _plan_json_to_legacy_blueprint(
                (plan_json or {}).get("blueprint") or {},
                (plan_json or {}).get("design_tokens"),
                requirement,
            )
            s2 = _run_s2_deterministic(legacy_bp, (plan_json or {}).get("design_tokens"))
            s5 = _run_s5_deterministic(legacy_bp, (plan_json or {}).get("blueprint") or {}, generated_files=files_result)
            template = _load_template_files()
            files_result = {**template, **s2, **s5}

    try:
        asyncio.get_event_loop().run_until_complete(_run())
    except RuntimeError:
        # Already in an event loop (e.g. during testing)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _run())
            future.result(timeout=120)

    logger.info("[frontend_only_sync] done | files=%d", len(files_result))
    return files_result


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

async def _run_s1(
    blueprint_json: Dict,
    prd_json: Optional[Dict],
    design_tokens_json: Optional[Dict],
    llm: Any,
    on_event: Callable,
) -> Dict[str, str]:
    """S1: Generate types.ts and mock/index.ts."""
    from app_builder.services.context_builder import build_types_context, build_mock_context
    from app_builder.services.fo_stage_agents import generate_types, generate_mock

    types_ctx = build_types_context(blueprint_json, prd_json)
    types_content = await generate_types(types_ctx, llm, on_event)

    mock_ctx = build_mock_context(blueprint_json, types_content)
    mock_content = await generate_mock(mock_ctx, llm, on_event)

    return {
        "frontend/src/types/index.ts": types_content,
        "frontend/src/mock/index.ts": mock_content,
    }


def _run_s2_deterministic(
    legacy_bp: Dict,
    design_tokens_json: Optional[Dict],
) -> Dict[str, str]:
    """S2: Deterministic theme/tokens.ts (no LLM). App.tsx stub."""
    from app_builder.services.deterministic_generator import (
        _build_tokens_ts,
        _build_mock_index,
    )

    # Enrich legacy_bp with DesignTokenSchema data if available
    if design_tokens_json:
        light = design_tokens_json.get("light") or {}
        legacy_bp = dict(legacy_bp)
        legacy_bp["primary_color"] = light.get("primary") or legacy_bp.get("primary_color", "#1976d2")
        legacy_bp["background_color"] = light.get("background") or legacy_bp.get("background_color", "#f5f7fb")
        legacy_bp["font_family"] = design_tokens_json.get("font_family") or legacy_bp.get("font_family", "Inter")

    tokens_ts = _build_tokens_ts(legacy_bp)
    return {
        "frontend/src/theme/tokens.ts": tokens_ts,
    }


async def _run_s3(
    blueprint_json: Dict,
    types_content: str,
    llm: Any,
    on_event: Callable,
) -> Dict[str, str]:
    """S3: Optional shared domain sub-components."""
    from app_builder.services.fo_stage_agents import generate_shared
    try:
        return await generate_shared(blueprint_json, types_content, llm, on_event)
    except Exception as exc:
        logger.warning("[pipeline/S3] shared components failed: %s", exc)
        return {}


async def _run_s4_pages(
    blueprint_json: Dict,
    uiux_json: Optional[Dict],
    design_tokens_json: Optional[Dict],
    types_content: str,
    mock_exports: List[str],
    llm: Any,
    on_event: Callable,
) -> Dict[str, str]:
    """S4: Generate all pages in parallel."""
    from app_builder.services.context_builder import build_page_context
    from app_builder.services.fo_stage_agents import generate_pages_parallel

    pages = blueprint_json.get("pages") or []
    page_contexts: List[Dict] = []
    for page in pages:
        page_file = page.get("file") or page.get("name") or "DashboardPage"
        ctx = build_page_context(
            page_file=page_file,
            blueprint_json=blueprint_json,
            uiux_json=uiux_json,
            design_tokens_json=design_tokens_json,
            types_content=types_content,
            mock_exports=mock_exports,
        )
        page_contexts.append(ctx)

    if not page_contexts:
        logger.warning("[pipeline/S4] no pages in blueprint — using default stubs")
        return _generate_default_page_stubs(blueprint_json)

    return await generate_pages_parallel(page_contexts, llm, on_event, max_concurrency=3)


def _run_s5_deterministic(
    legacy_bp: Dict,
    blueprint_json: Dict,
    generated_files: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """S5: Deterministic App.tsx, router, package.json.

    Generates App.tsx from blueprint pages. Falls back to scanning `generated_files`
    for actual *.tsx page files so App.tsx is ALWAYS correct even when blueprint_json
    has incomplete routes/pages data.
    """
    from app_builder.services.deterministic_generator import generate_from_blueprint

    # Merge blueprint_json routes into legacy_bp pages list
    if blueprint_json.get("routes") and blueprint_json.get("pages"):
        merged_pages = _merge_blueprint_pages(blueprint_json)
        if merged_pages:
            legacy_bp = dict(legacy_bp)
            legacy_bp["pages"] = merged_pages

    # If pages still empty, rebuild from actual generated files on disk
    if not legacy_bp.get("pages") and generated_files:
        legacy_bp = dict(legacy_bp)
        legacy_bp["pages"] = _pages_from_files(generated_files)

    return generate_from_blueprint(legacy_bp)


async def _run_s6_verify(
    project_name: str,
    files: Dict[str, str],
    on_event: Callable,
) -> None:
    """S6: Final tsc verification."""
    from app_builder.services.verify_service import verify_stage as do_verify

    await on_event({"event": "stage_verify", "stage": "S6", "status": "running"})
    result = await do_verify(
        project_name=project_name,
        files=files,
        new_files=files,  # verify everything
        stage_name="S6",
    )
    if result.ok:
        await on_event({
            "event": "stage_verify", "stage": "S6",
            "status": "ok", "message": result.log,
        })
    else:
        # S6 failure is a warning, not a hard stop (we've already done per-stage fixes)
        await on_event({
            "event": "stage_verify", "stage": "S6",
            "status": "warning",
            "errors": result.errors[:5],
            "message": "Final tsc found issues; app may still run",
        })


# ---------------------------------------------------------------------------
# Verify + fix loop (runs after S1, S3, S4)
# ---------------------------------------------------------------------------

async def _verify_and_fix_stage(
    stage: str,
    project_name: str,
    files: Dict[str, str],
    new_files: Dict[str, str],
    llm: Any,
    on_event: Callable,
    types_content: str = "",
    mock_exports: Optional[List[str]] = None,
    metrics: Any = None,
) -> None:
    """
    Verify the stage files; if verification fails, invoke the fixer (up to MAX_FIX_ATTEMPTS).
    Modifies `files` and `new_files` in place on success.
    """
    from app_builder.services.verify_service import (
        verify_stage as do_verify,
        is_complete_file,
        extract_first_error,
    )
    from app_builder.services.fo_stage_agents import fix_file

    mock_exports_list: List[str] = mock_exports or []

    await on_event({"event": "stage_verify", "stage": stage, "status": "running"})

    result = await do_verify(
        project_name=project_name,
        files=files,
        new_files=new_files,
        stage_name=stage,
    )

    if result.ok:
        await on_event({
            "event": "stage_verify", "stage": stage,
            "status": "ok", "tsc_ran": result.tsc_ran,
        })
        return

    # Retry loop
    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        error_info = extract_first_error(result, files)
        offending = error_info.get("offending_file", "")

        if metrics:
            try:
                metrics.record_fix_attempt(stage=stage, attempt=attempt, offending_file=offending, succeeded=False)
            except Exception:
                pass

        await on_event({
            "event": "stage_fix", "stage": stage, "attempt": attempt,
            "file": offending, "error": error_info.get("error", "")[:200],
        })

        if not offending:
            break

        fixed_content = await fix_file(
            error_info=error_info,
            types_content=types_content,
            mock_exports=mock_exports_list,
            llm=llm,
        )

        if fixed_content and is_complete_file(fixed_content):
            files[offending] = fixed_content
            new_files[offending] = fixed_content
            # Re-verify
            result = await do_verify(
                project_name=project_name,
                files=files,
                new_files={offending: fixed_content},
                stage_name=stage,
            )
            if result.ok:
                if metrics:
                    try:
                        metrics.record_fix_attempt(
                            stage=stage, attempt=attempt, offending_file=offending, succeeded=True
                        )
                    except Exception:
                        pass
                await on_event({
                    "event": "stage_verify", "stage": stage,
                    "status": "ok", "fixed": True, "attempts": attempt,
                })
                return
        else:
            logger.warning("[pipeline/%s] fixer returned incomplete file (attempt %d)", stage, attempt)

    # Exhausted retries
    await on_event({
        "event": "stage_verify", "stage": stage,
        "status": "failed",
        "errors": result.errors[:5],
        "log": result.log[-500:],
        "message": f"Stage {stage} verification failed after {MAX_FIX_ATTEMPTS} fix attempts",
    })
    # We do NOT raise here — we continue with best-effort output
    logger.warning("[pipeline/%s] exhausted fix attempts; continuing anyway", stage)


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------

async def _emit_stage_start(on_event: Callable, stage: str, label: str, index: int, total: int) -> None:
    await on_event({
        "event": "agent_start",
        "agent": f"fo_stage_{stage.lower()}",
        "message": f"[{index}/{total}] {label}",
        "stage": stage,
        "label": label,
        "index": index,
        "total": total,
    })


async def _emit_stage_done(on_event: Callable, stage: str, label: str, new_files: Dict) -> None:
    await on_event({
        "event": "agent_complete",
        "agent": f"fo_stage_{stage.lower()}",
        "message": f"{label} complete ({len(new_files)} files)",
        "stage": stage,
        "files": list(new_files.keys()),
    })


# ---------------------------------------------------------------------------
# Template loader
# ---------------------------------------------------------------------------

def _load_template_files() -> Dict[str, str]:
    """Load all files from base-frontend-vite-mui template (excludes node_modules)."""
    if not os.path.isdir(_TEMPLATE_DIR):
        logger.error("[frontend_only] template missing at %s", _TEMPLATE_DIR)
        return {}
    files: Dict[str, str] = {}
    exclude_dirs = {"__pycache__", "node_modules", ".git", "dist", "build"}
    for root, dirs, filenames in os.walk(_TEMPLATE_DIR):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        rel_root = os.path.relpath(root, _TEMPLATE_DIR)
        for fn in filenames:
            if fn.endswith(".pyc") or fn == "base.json":
                continue
            rel_path = os.path.join(rel_root, fn) if rel_root != "." else fn
            rel_path = rel_path.replace("\\", "/")
            if not rel_path.startswith("frontend/"):
                continue
            full = os.path.join(root, fn)
            try:
                with open(full, "r", encoding="utf-8") as fh:
                    files[rel_path] = fh.read()
            except OSError:
                pass
    logger.info("[frontend_only] loaded template | files=%d", len(files))
    return files


# ---------------------------------------------------------------------------
# Blueprint conversion helpers
# ---------------------------------------------------------------------------

def _plan_json_to_legacy_blueprint(
    blueprint_json: Dict,
    design_tokens_json: Optional[Dict],
    requirement: str,
) -> Dict[str, Any]:
    """Convert a Blueprint schema dict to the legacy blueprint format for deterministic_generator."""
    app_name = blueprint_json.get("app_name") or requirement[:60]
    domain = blueprint_json.get("domain") or "generic"

    primary = "#1976d2"
    background = "#f5f7fb"
    font = "Inter"
    if design_tokens_json:
        light = design_tokens_json.get("light") or {}
        primary = light.get("primary") or primary
        background = light.get("background") or background
        font = design_tokens_json.get("font_family") or font

    pages: List[Dict] = _merge_blueprint_pages(blueprint_json)

    return {
        "app_name": app_name,
        "description": requirement[:300],
        "pages": pages,
        "primary_color": primary,
        "background_color": background,
        "style": "minimal",
        "domain": domain,
        "font_family": font.split(",")[0].strip(),
    }


def _merge_blueprint_pages(blueprint_json: Dict) -> List[Dict]:
    """Merge blueprint routes + pages into legacy PageSpec list."""
    routes = blueprint_json.get("routes") or []
    pages = blueprint_json.get("pages") or []
    nav_items = blueprint_json.get("nav_items") or []

    # Build nav item index by path
    nav_by_path = {n.get("path"): n for n in nav_items}

    result: List[Dict] = []
    seen_paths = set()

    for route in routes:
        path = route.get("path", "/")
        page_file = route.get("page") or ""
        if not page_file or path in seen_paths:
            continue
        seen_paths.add(path)

        # Find matching blueprint page for kit info
        bp_page = next((p for p in pages if p.get("file") == page_file), {})
        nav = nav_by_path.get(path, {})

        result.append({
            "name": page_file if page_file.endswith("Page") else page_file,
            "path": path,
            "nav_label": nav.get("label") or page_file.replace("Page", ""),
            "icon": nav.get("icon") or "Article",
            "description": bp_page.get("description") or "",
        })

    return result


def _derive_blueprint_from_artifacts(
    requirement: str,
    prd: str,
    uiux: str,
    architecture: Optional[Dict],
    design_tokens: Optional[Dict],
    domain: str,
) -> Dict:
    """Derive a Blueprint from legacy planning artifacts (no plan_json)."""
    # Reuse the old logic from the original stub
    try:
        from app_builder.services.fullstack_frontend_generator import theme_from_tokens
        from app_builder.services.fullstack_app_generator import extract_app_title
        app_name = extract_app_title(requirement, prd)
        colors = theme_from_tokens(design_tokens, uiux=uiux)
    except Exception:
        app_name = requirement[:60]
        colors = {}

    pages: List[Dict] = _default_pages_for(domain, app_name)
    return {
        "app_name": app_name,
        "description": requirement[:300],
        "pages": pages,
        "primary_color": colors.get("primary", "#1976d2"),
        "background_color": colors.get("background", "#f5f7fb"),
        "style": "minimal",
        "domain": domain,
        "font_family": "Inter",
    }


def _generate_default_page_stubs(blueprint_json: Dict) -> Dict[str, str]:
    """Generate placeholder page stubs when no pages found in blueprint."""
    from app_builder.services.fo_stage_agents import _fallback_page
    stubs = {}
    for page in _default_pages_for(blueprint_json.get("domain") or "generic", "App"):
        name = page["name"]
        stubs[f"frontend/src/pages/{name}.tsx"] = _fallback_page(name, page["nav_label"])
    return stubs


def _pages_from_files(files: Dict[str, str]) -> List[Dict]:
    """
    Derive a legacy pages list by scanning the files dict for *.tsx page files.
    Called as a fallback when blueprint_json has no usable routes/pages.
    """
    import re as _re

    PAGE_RE = _re.compile(r"frontend/src/pages/([A-Za-z0-9_]+Page)\.tsx$")
    ICON_MAP = {
        "Dashboard": "Dashboard", "Main": "Dashboard", "Home": "Home",
        "Contacts": "People", "Deals": "Handshake", "Leads": "PersonAdd",
        "Activities": "Event", "Calendar": "CalendarToday",
        "Reports": "BarChart", "Analytics": "Analytics",
        "Settings": "Settings", "Profile": "AccountCircle",
        "Users": "Group", "Teams": "Groups",
        "Products": "Inventory", "Inventory": "Warehouse",
        "Orders": "ShoppingCart", "Billing": "CreditCard",
        "Tickets": "ConfirmationNumber", "Support": "HeadsetMic",
        "Courses": "School", "Students": "School",
        "Events": "Event", "Venues": "LocationOn",
    }

    pages = []
    seen = set()
    for path in sorted(files.keys()):
        m = PAGE_RE.match(path)
        if not m:
            continue
        name = m.group(1)
        if name == "DemoPage" or name in seen:
            continue
        seen.add(name)
        base = name.replace("Page", "")
        slug = base.lower()
        route_path = "/" if base in ("Dashboard", "Main", "Home") else f"/{slug}"
        pages.append({
            "name": name,
            "path": route_path,
            "nav_label": base,
            "icon": ICON_MAP.get(base, "Article"),
            "description": f"{base} page",
        })
    return pages


# ---------------------------------------------------------------------------
# Frozen path guard — template infrastructure files must never be overwritten
# ---------------------------------------------------------------------------

_FROZEN_FO_PATHS: frozenset = frozenset({
    "frontend/src/components/ui/index.ts",
    "frontend/src/main.tsx",
    "frontend/vite.config.ts",
    "frontend/tsconfig.json",
    "frontend/index.html",
    "frontend/src/vite-env.d.ts",
    "frontend/src/theme/index.ts",
})


def _drop_frozen(files: Dict[str, str]) -> Dict[str, str]:
    """Remove any frozen template paths from a generated file dict."""
    out = {}
    for path, content in files.items():
        norm = path.replace("\\", "/").lstrip("/")
        if norm in _FROZEN_FO_PATHS:
            logger.warning(
                "[frontend_only] LLM-generated file '%s' is frozen — discarding", path
            )
        else:
            out[path] = content
    return out


def _extract_export_names(content: str) -> List[str]:
    """Extract exported names from a TypeScript file content."""
    if not content:
        return []
    exports: List[str] = []
    for m in re.finditer(r"export\s+(?:const|function|class|interface|type|enum)\s+(\w+)", content):
        exports.append(m.group(1))
    for m in re.finditer(r"export\s*\{([^}]+)\}", content):
        for name in m.group(1).split(","):
            clean = name.strip().split(" as ")[-1].strip()
            if clean:
                exports.append(clean)
    return list(dict.fromkeys(exports))


def _default_pages_for(domain: str, app_name: str) -> List[Dict[str, Any]]:
    defaults: Dict[str, List[Dict]] = {
        "crm": [
            {"name": "DashboardPage", "path": "/", "nav_label": "Dashboard", "icon": "Dashboard"},
            {"name": "ContactsPage", "path": "/contacts", "nav_label": "Contacts", "icon": "People"},
            {"name": "DealsPage", "path": "/deals", "nav_label": "Deals", "icon": "Handshake"},
            {"name": "ActivitiesPage", "path": "/activities", "nav_label": "Activities", "icon": "Event"},
            {"name": "ReportsPage", "path": "/reports", "nav_label": "Reports", "icon": "BarChart"},
        ],
        "inventory": [
            {"name": "DashboardPage", "path": "/", "nav_label": "Dashboard", "icon": "Dashboard"},
            {"name": "ProductsPage", "path": "/products", "nav_label": "Products", "icon": "Inventory"},
            {"name": "OrdersPage", "path": "/orders", "nav_label": "Orders", "icon": "ShoppingCart"},
            {"name": "SuppliersPage", "path": "/suppliers", "nav_label": "Suppliers", "icon": "LocalShipping"},
            {"name": "ReportsPage", "path": "/reports", "nav_label": "Reports", "icon": "BarChart"},
        ],
        "analytics_dashboard": [
            {"name": "DashboardPage", "path": "/", "nav_label": "Dashboard", "icon": "Dashboard"},
            {"name": "TrendsPage", "path": "/trends", "nav_label": "Trends", "icon": "TrendingUp"},
            {"name": "BreakdownPage", "path": "/breakdown", "nav_label": "Breakdown", "icon": "PieChart"},
            {"name": "RawDataPage", "path": "/data", "nav_label": "Raw Data", "icon": "TableChart"},
        ],
        "kanban": [
            {"name": "BoardsPage", "path": "/", "nav_label": "Boards", "icon": "Dashboard"},
            {"name": "BoardViewPage", "path": "/board", "nav_label": "Board", "icon": "ViewKanban"},
            {"name": "CardDetailPage", "path": "/card", "nav_label": "Card", "icon": "Article"},
        ],
        "booking": [
            {"name": "ServicesPage", "path": "/", "nav_label": "Services", "icon": "Build"},
            {"name": "AvailabilityPage", "path": "/availability", "nav_label": "Availability", "icon": "CalendarMonth"},
            {"name": "BookingFormPage", "path": "/book", "nav_label": "Book", "icon": "BookOnline"},
            {"name": "MyBookingsPage", "path": "/my-bookings", "nav_label": "My Bookings", "icon": "EventNote"},
        ],
        "hr_onboarding": [
            {"name": "DashboardPage", "path": "/", "nav_label": "Dashboard", "icon": "Dashboard"},
            {"name": "EmployeesPage", "path": "/employees", "nav_label": "Employees", "icon": "Badge"},
            {"name": "OnboardingPage", "path": "/onboarding", "nav_label": "Onboarding", "icon": "PersonAdd"},
            {"name": "LeavePage", "path": "/leave", "nav_label": "Leave", "icon": "EventBusy"},
            {"name": "ReportsPage", "path": "/reports", "nav_label": "Reports", "icon": "BarChart"},
        ],
        "finance": [
            {"name": "DashboardPage", "path": "/", "nav_label": "Dashboard", "icon": "Dashboard"},
            {"name": "InvoicesPage", "path": "/invoices", "nav_label": "Invoices", "icon": "Receipt"},
            {"name": "ExpensesPage", "path": "/expenses", "nav_label": "Expenses", "icon": "Money"},
            {"name": "ReportsPage", "path": "/reports", "nav_label": "Reports", "icon": "BarChart"},
        ],
    }
    return defaults.get(domain, [
        {"name": "DashboardPage", "path": "/", "nav_label": "Dashboard", "icon": "Dashboard"},
        {"name": "MainPage", "path": "/main", "nav_label": "Main", "icon": "Article"},
        {"name": "ReportsPage", "path": "/reports", "nav_label": "Reports", "icon": "BarChart"},
        {"name": "SettingsPage", "path": "/settings", "nav_label": "Settings", "icon": "Settings"},
    ])
