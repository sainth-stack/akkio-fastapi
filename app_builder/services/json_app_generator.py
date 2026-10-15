"""JSON-only app generation: template + mock/data.json (no LLM TSX)."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, Optional

from app_builder.schemas.app_definition import AppDefinition
from app_builder.services.app_definition_generator import (
    hydrate_plan_json,
    resolve_app_definition,
    resolve_app_definition_sync,
)
from app_builder.services.template_app_data import app_data_dict_from_definition
from app_builder.services.json_app_codegen_pipeline import (
    run_app_definition_phase,
    run_coding_phase,
    run_integration_phase,
    run_planner_phase,
    run_styling_phase,
    run_validator_handoff,
)

logger = logging.getLogger("app_builder")

TEMPLATE_NAME = "base-frontend-vite-mui"
_TEMPLATES_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "templates",
)
_TEMPLATE_DIR = os.path.join(_TEMPLATES_BASE, TEMPLATE_NAME)


def load_template_frontend_files() -> Dict[str, str]:
    if not os.path.isdir(_TEMPLATE_DIR):
        logger.error("[json_app] template missing: %s", _TEMPLATE_DIR)
        return {}
    files: Dict[str, str] = {}
    exclude = {"__pycache__", "node_modules", ".git", "dist", "build"}
    for root, dirs, filenames in os.walk(_TEMPLATE_DIR):
        dirs[:] = [d for d in dirs if d not in exclude]
        rel_root = os.path.relpath(root, _TEMPLATE_DIR)
        for fn in filenames:
            if fn.endswith(".pyc") or fn == "base.json":
                continue
            rel = os.path.join(rel_root, fn) if rel_root != "." else fn
            rel = rel.replace("\\", "/")
            if not rel.startswith("frontend/"):
                continue
            try:
                with open(os.path.join(root, fn), encoding="utf-8") as fh:
                    files[rel] = fh.read()
            except OSError:
                pass
    logger.info("[json_app] loaded template | files=%d", len(files))
    return files


def _tokens_ts_from_theme(theme: Any) -> str:
    t = theme.model_dump() if hasattr(theme, "model_dump") else theme
    palette = t.get("chartPalette") or []
    font = (t.get("fontFamily") or "Inter").replace("'", "\\'")
    return f"""/**
 * Design tokens — generated from AppDefinition.theme (edit mock/data.json to restyle).
 */
export const tokens = {{
  primary:        '{t.get("primary", "#1976d2")}',
  primaryDark:    '{t.get("primaryDark", "#1565c0")}',
  primaryLight:   '{t.get("primaryLight", "#42a5f5")}',
  secondary:      '{t.get("secondary", "#9c27b0")}',
  accent:         '#00bcd4',
  background:     '{t.get("background", "#f5f7fb")}',
  surface:        '{t.get("surface", "#ffffff")}',
  text:           '{t.get("text", "#0f172a")}',
  muted:          '{t.get("muted", "#64748b")}',
  border:         '#e2e8f0',
  danger:         '{t.get("danger", "#d32f2f")}',
  success:        '{t.get("success", "#2e7d32")}',
  warning:        '{t.get("warning", "#ed6c02")}',
  info:           '{t.get("info", "#0288d1")}',
  fontFamily:     "'{font}', system-ui, -apple-system, sans-serif",
  chartPalette:   {json.dumps(palette)},
  shellVariant:   '{t.get("shellVariant", "light")}',
  border:         '{t.get("border", "#e2e8f0")}',
}} as const;

export type Tokens = typeof tokens;
"""


def _coerce_plan_texts(
    plan_json: Optional[Dict[str, Any]],
    prd: str,
    uiux: str,
) -> tuple[str, str]:
    prd_text = prd or ""
    uiux_text = uiux or ""
    if plan_json:
        if isinstance(plan_json.get("prd"), dict):
            prd_text = prd_text or json.dumps(plan_json["prd"])
        elif isinstance(plan_json.get("prd"), str):
            prd_text = prd_text or plan_json["prd"]
        if isinstance(plan_json.get("uiux"), dict):
            uiux_text = uiux_text or json.dumps(plan_json["uiux"])
        elif isinstance(plan_json.get("uiux"), str):
            uiux_text = uiux_text or plan_json["uiux"]
    return prd_text, uiux_text


def generate_json_app_files(
    *,
    requirement: str = "",
    prd: str = "",
    uiux: str = "",
    plan_json: Optional[Dict[str, Any]] = None,
    app_name: Optional[str] = None,
    app_definition: Optional[AppDefinition] = None,
    architecture: Optional[Dict[str, Any]] = None,
    design_tokens: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Return full frontend file map: frozen template + mock/data.json + tokens."""
    merged_plan = hydrate_plan_json(
        plan_json,
        architecture=architecture,
        design_tokens=design_tokens,
    )
    prd_text, uiux_text = _coerce_plan_texts(merged_plan, prd, uiux)

    definition = app_definition or resolve_app_definition_sync(
        requirement=requirement,
        prd=prd_text,
        uiux=uiux_text,
        plan_json=merged_plan,
        app_name=app_name,
    )

    files = load_template_frontend_files()

    files["frontend/src/mock/data.json"] = json.dumps(
        app_data_dict_from_definition(definition), indent=2, ensure_ascii=False
    )
    files["frontend/src/theme/tokens.ts"] = _tokens_ts_from_theme(definition.theme)

    if "frontend/tsconfig.json" in files:
        try:
            tc = json.loads(files["frontend/tsconfig.json"])
            tc.setdefault("compilerOptions", {})["resolveJsonModule"] = True
            files["frontend/tsconfig.json"] = json.dumps(tc, indent=2) + "\n"
        except json.JSONDecodeError:
            pass

    files.pop("frontend/src/pages/DemoPage.tsx", None)

    logger.info(
        "[json_app] definition | pages=%d nav=%d mock_keys=%d",
        len(definition.pages),
        len(definition.navigation),
        len(definition.mock),
    )
    return files


async def run_json_app_codegen(
    *,
    requirement: str,
    plan_json: Optional[Dict[str, Any]],
    project_name: str,
    on_event: Optional[Callable] = None,
    prd: str = "",
    uiux: str = "",
    architecture: Optional[Dict[str, Any]] = None,
    design_tokens: Optional[Dict[str, Any]] = None,
    app_name: Optional[str] = None,
    use_llm: bool = True,
) -> Dict[str, str]:
    """Async entry for codegen WebSocket — LLM AppDefinition from PRD, then bind template."""
    async def emit(payload: Dict[str, Any]) -> None:
        if on_event:
            await on_event(payload)

    merged_plan = hydrate_plan_json(
        plan_json,
        architecture=architecture,
        design_tokens=design_tokens,
    )
    prd_text, uiux_text = _coerce_plan_texts(merged_plan, prd, uiux)

    from app_builder.services.app_definition_generator import resolve_app_display_name

    display_name = resolve_app_display_name(
        requirement=requirement,
        prd=prd_text,
        plan_json=merged_plan,
        app_name=app_name or project_name,
    )
    await emit({
        "event": "pipeline_log",
        "agent": "codegen",
        "message": (
            f"Production pipeline started — json_app track | target: {display_name} | "
            "PRD → theme → AppDefinition → template → build verify."
        ),
    })
    await emit({
        "event": "generation_start",
        "message": f"Generating production UI for {display_name}…",
    })

    screens = await run_planner_phase(
        emit,
        app_name=display_name,
        requirement=requirement,
        prd_text=prd_text,
        uiux_text=uiux_text,
        plan_json=merged_plan,
    )

    await run_styling_phase(emit, plan_json=merged_plan, app_name=display_name)

    async def _on_def_progress(msg: str) -> None:
        await emit({"event": "agent_progress", "agent": "app_definition_agent", "message": msg})

    async def _resolve() -> AppDefinition:
        return await resolve_app_definition(
            requirement=requirement,
            prd=prd_text,
            uiux=uiux_text,
            plan_json=merged_plan,
            app_name=app_name,
            use_llm=use_llm,
            on_progress=_on_def_progress,
        )

    definition = await run_app_definition_phase(
        emit,
        screens=screens,
        resolve_coro=_resolve,
    )

    await emit({
        "event": "app_definition_json",
        "data": definition.to_json_dict(),
    })

    files = generate_json_app_files(
        requirement=requirement,
        prd=prd_text,
        uiux=uiux_text,
        plan_json=merged_plan,
        app_definition=definition,
        architecture=architecture,
        design_tokens=design_tokens,
        app_name=app_name,
    )

    data_blob = files.get("frontend/src/mock/data.json") or ""
    data_kb = len(data_blob.encode("utf-8")) / 1024.0

    await run_coding_phase(
        emit,
        definition=definition,
        file_count=len(files),
        data_json_kb=data_kb,
    )

    await run_integration_phase(emit, definition=definition, template_name=TEMPLATE_NAME)
    await run_validator_handoff(emit)

    return files
