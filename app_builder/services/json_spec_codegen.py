"""JSON-spec codegen — no per-app React/TS generation."""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

from api.app_creator.pipeline_helpers import public_base_url
from app_builder.services.json_app_spec_builder import build_app_spec_v1
from app_builder.services.json_renderer_build import ensure_json_renderer_built

logger = logging.getLogger("app_builder")


async def run_json_spec_codegen(
    *,
    requirement: str,
    prd: str,
    uiux: str,
    architecture: Optional[Dict[str, Any]],
    design_tokens: Optional[Dict[str, Any]],
    plan_json: Optional[Dict[str, Any]],
    app_id: Optional[str],
    project_name: str,
    domain: str,
    app_name: str,
    on_event: Callable,
) -> tuple[Dict[str, str], Dict[str, Any], str]:
    await on_event({
        "event": "pipeline_log",
        "agent": "json_spec",
        "message": f"Track: json_spec | Building AppSpec JSON (domain={domain}) — no file codegen",
    })
    await on_event({
        "event": "agent_start",
        "agent": "spec_generator",
        "message": "Generating AppSpec JSON from PRD, UI/UX, and architecture...",
    })

    blueprint = (plan_json or {}).get("blueprint") or {}
    if blueprint.get("domain"):
        domain = str(blueprint["domain"])

    spec = build_app_spec_v1(
        requirement=requirement,
        prd=prd,
        uiux=uiux,
        architecture=architecture,
        design_tokens=design_tokens or (plan_json or {}).get("design_tokens"),
        app_name=app_name,
        domain=domain,
        app_id=str(app_id) if app_id else None,
    )

    await on_event({
        "event": "agent_complete",
        "agent": "spec_generator",
        "message": f"AppSpec ready — {len(spec.get('screens', []))} screens, {len(spec.get('navigation', []))} nav items",
    })

    await on_event({
        "event": "agent_start",
        "agent": "renderer_agent",
        "message": "Ensuring shared JSON renderer is built (one-time)...",
    })
    try:
        ensure_json_renderer_built()
    except Exception as exc:
        logger.warning("[json_spec] renderer build: %s", exc)
        await on_event({
            "event": "agent_error",
            "agent": "renderer_agent",
            "message": f"Renderer build failed: {exc}",
        })
        raise

    await on_event({
        "event": "agent_complete",
        "agent": "renderer_agent",
        "message": "Shared renderer ready — preview uses AppSpec JSON only",
    })

    base = public_base_url().rstrip("/")
    preview = f"{base}/spec-app/{app_id}" if app_id else f"{base}/spec-app/0"

    await on_event({
        "event": "pipeline_log",
        "agent": "json_spec",
        "message": f"Preview URL: {preview}",
    })

    # Minimal virtual files for Code tab / DB storage
    files = {
        "app_spec.json": json.dumps(spec, indent=2),
        "README.md": (
            "# JSON Spec Application\n\n"
            "This app is rendered from `app_spec.json` by the shared Akkio spec renderer.\n"
            "No per-project TypeScript was generated.\n"
        ),
    }
    files["_meta"] = json.dumps({"renderer": "json_spec_v1", "preview_url": preview})
    return files, spec, preview
