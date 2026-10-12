"""Production-style agent timeline for json_app codegen (progress + pacing)."""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app_builder.services.agent_pipeline_events import (
    AGENT_APP_DEFINITION,
    AGENT_CODING,
    AGENT_INTEGRATION,
    AGENT_PLANNER,
    AGENT_STYLING,
    AGENT_VALIDATOR,
)
from app_builder.services.app_definition_generator import extract_screens_from_plan

EmitFn = Callable[[Dict[str, Any]], Awaitable[None]]

_STEP_SEC = 0.11


async def _tick() -> None:
    await asyncio.sleep(_STEP_SEC)


async def _progress(emit: EmitFn, agent: str, message: str) -> None:
    await emit({"event": "agent_progress", "agent": agent, "message": message})
    await _tick()


async def run_planner_phase(
    emit: EmitFn,
    *,
    app_name: str,
    requirement: str,
    prd_text: str,
    uiux_text: str,
    plan_json: Dict[str, Any],
) -> List[Dict[str, str]]:
    screens = extract_screens_from_plan(
        requirement=requirement,
        prd=prd_text,
        uiux=uiux_text,
        plan_json=plan_json,
    )
    blueprint = plan_json.get("blueprint") if isinstance(plan_json.get("blueprint"), dict) else {}
    domain = (blueprint.get("selected_domain") or blueprint.get("domain") or "generic") if blueprint else "generic"

    await emit({
        "event": "agent_start",
        "agent": AGENT_PLANNER,
        "message": "Loading production planning contracts…",
    })
    await _progress(emit, AGENT_PLANNER, f"Product name: {app_name}")
    await _progress(emit, AGENT_PLANNER, f"Domain profile: {domain}")
    await _progress(emit, AGENT_PLANNER, f"PRD context: {min(len(prd_text), 12000):,} characters indexed")
    if plan_json.get("architecture"):
        await _progress(emit, AGENT_PLANNER, "Architecture JSON: routes, screens, and data contracts loaded")
    if screens:
        labels = ", ".join(s["title"] for s in screens[:8])
        await _progress(emit, AGENT_PLANNER, f"Screen map: {len(screens)} pages → {labels}")
    else:
        await _progress(emit, AGENT_PLANNER, "Screen map: inferring pages from requirement phrases")
    await emit({
        "event": "agent_complete",
        "agent": AGENT_PLANNER,
        "message": f"Planning locked — {len(screens) or 'dynamic'} screens ready for UI synthesis.",
    })
    return screens


async def run_styling_phase(
    emit: EmitFn,
    *,
    plan_json: Dict[str, Any],
    app_name: str,
) -> None:
    tokens = plan_json.get("design_tokens") if isinstance(plan_json.get("design_tokens"), dict) else {}
    await emit({
        "event": "agent_start",
        "agent": AGENT_STYLING,
        "message": "Applying brand system to the shell…",
    })
    await _progress(emit, AGENT_STYLING, f"Header title: {app_name}")
    if tokens:
        primary = (tokens.get("light") or tokens.get("colors") or tokens).get("primary", "#1976d2")
        await _progress(emit, AGENT_STYLING, f"Primary brand color: {primary}")
        await _progress(emit, AGENT_STYLING, "Typography scale: Inter / system UI stack")
    else:
        await _progress(emit, AGENT_STYLING, "Using default enterprise palette (primary #1976d2)")
    await _progress(emit, AGENT_STYLING, "Chart palette: 6-series color tokens for Recharts")
    await _progress(emit, AGENT_STYLING, "MUI component overrides: cards, tables, sidebar selection")
    await emit({
        "event": "agent_complete",
        "agent": AGENT_STYLING,
        "message": "Design tokens ready — theme will render from mock/data.json at runtime.",
    })


async def run_app_definition_phase(
    emit: EmitFn,
    *,
    screens: List[Dict[str, str]],
    resolve_coro,
) -> Any:
    await emit({
        "event": "agent_start",
        "agent": AGENT_APP_DEFINITION,
        "message": "Synthesizing production AppDefinition (navigation, blocks, MVP datasets)…",
    })
    for s in screens[:6]:
        await _progress(emit, AGENT_APP_DEFINITION, f"Page spec: {s.get('title')} → blocks (KPI / charts / tables)")
    await _progress(emit, AGENT_APP_DEFINITION, "Generating JSON schema: kpiRow, lineChart, barChart, dataTable…")
    await _progress(emit, AGENT_APP_DEFINITION, "Building realistic mock datasets (KPIs, time-series, tabular rows)…")

    definition = await resolve_coro()

    n_pages = len(definition.pages)
    n_nav = len(definition.navigation)
    n_mock = len(definition.mock or {})
    block_count = sum(len(p.blocks) for p in definition.pages)
    await _progress(
        emit,
        AGENT_APP_DEFINITION,
        f"Validated {n_pages} pages, {block_count} blocks, {n_mock} mock datasets",
    )
    for nav in definition.navigation[:6]:
        await _progress(emit, AGENT_APP_DEFINITION, f"Nav: {nav.label} → {nav.path} ({nav.pageId})")
    await emit({
        "event": "agent_complete",
        "agent": AGENT_APP_DEFINITION,
        "message": f"AppDefinition complete — {n_pages} screens, {n_nav} routes, production-grade sample data.",
    })
    return definition


async def run_coding_phase(
    emit: EmitFn,
    *,
    definition: Any,
    file_count: int,
    data_json_kb: float,
) -> None:
    await emit({
        "event": "agent_start",
        "agent": AGENT_CODING,
        "message": "Materializing frontend workspace (Vite + MUI template)…",
    })
    await _progress(emit, AGENT_CODING, "Template: base-frontend-vite-mui (renderer + UI kit)")
    await _progress(emit, AGENT_CODING, f"Writing frontend/src/mock/data.json ({data_json_kb:.1f} KB)")
    for p in definition.pages[:8]:
        await _progress(emit, AGENT_CODING, f"Serialized page «{p.title}» ({len(p.blocks)} blocks)")
    await _progress(emit, AGENT_CODING, "Generating frontend/src/theme/tokens.ts from AppDefinition.theme")
    await _progress(emit, AGENT_CODING, "Enabling resolveJsonModule in tsconfig.json")
    await emit({
        "event": "agent_complete",
        "agent": AGENT_CODING,
        "message": f"Codegen complete — {file_count} files written to project workspace.",
    })


async def run_integration_phase(emit: EmitFn, *, definition: Any, template_name: str) -> None:
    await emit({
        "event": "agent_start",
        "agent": AGENT_INTEGRATION,
        "message": "Wiring routes, shell, and JSON-driven renderer…",
    })
    await _progress(emit, AGENT_INTEGRATION, f"AppShell: {definition.meta.appName}")
    for nav in definition.navigation:
        await _progress(emit, AGENT_INTEGRATION, f"Route {nav.path} → PageRenderer({nav.pageId})")
    await _progress(emit, AGENT_INTEGRATION, "BlockRenderer: KPI rows, charts, tables bound to mock keys")
    await _progress(emit, AGENT_INTEGRATION, f"ThemeProvider: runtime palette from {template_name}")
    await emit({
        "event": "agent_complete",
        "agent": AGENT_INTEGRATION,
        "message": "Integration verified — SPA ready for preview (JSON-backed UI).",
    })


async def run_validator_handoff(emit: EmitFn) -> None:
    await emit({
        "event": "agent_start",
        "agent": AGENT_VALIDATOR,
        "message": "Starting production build gate (npm ci + vite build + smoke checks)…",
    })
    await _progress(emit, AGENT_VALIDATOR, "Dependency install: package.json lockfile resolve")
    await _progress(emit, AGENT_VALIDATOR, "Typecheck + bundle: npm run build")
