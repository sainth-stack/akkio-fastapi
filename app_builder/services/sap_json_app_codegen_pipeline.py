"""SAP domain: 4 agents — dashboard → data definition → RAG (step 2) → integration."""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app_builder.services.agent_pipeline_events import (
    AGENT_SAP_DASHBOARD,
    AGENT_SAP_DATA_DEFINITION,
    AGENT_SAP_INTEGRATION,
    AGENT_SAP_RAG,
)
from app_builder.services.app_definition_generator import extract_screens_from_plan

EmitFn = Callable[[Dict[str, Any]], Awaitable[None]]
_STEP_SEC = 0.12


async def _tick() -> None:
    await asyncio.sleep(_STEP_SEC)


async def _progress(emit: EmitFn, agent: str, message: str) -> None:
    await emit({"event": "agent_progress", "agent": agent, "message": message})
    await _tick()


async def run_sap_dashboard_phase(
    emit: EmitFn,
    *,
    app_name: str,
    requirement: str,
    plan_json: Dict[str, Any],
) -> List[Dict[str, str]]:
    screens = extract_screens_from_plan(requirement=requirement, plan_json=plan_json)
    await emit({
        "event": "agent_start",
        "agent": AGENT_SAP_DASHBOARD,
        "message": "Step 1 — designing your SAP tickets dashboard…",
    })
    await _progress(emit, AGENT_SAP_DASHBOARD, f"App name: {app_name}")
    await _progress(emit, AGENT_SAP_DASHBOARD, "Layout: KPI row + SAP Community ticket feed + assistant")
    if screens:
        await _progress(emit, AGENT_SAP_DASHBOARD, f"Pages: {', '.join(s['title'] for s in screens[:6])}")
    else:
        await _progress(emit, AGENT_SAP_DASHBOARD, "Default screen: Dashboard")
    await emit({
        "event": "agent_complete",
        "agent": AGENT_SAP_DASHBOARD,
        "message": "Dashboard shell planned — KPIs, search, and suggested-answer panel.",
    })
    return screens


async def run_sap_data_definition_phase(
    emit: EmitFn,
    *,
    screens: List[Dict[str, str]],
    resolve_coro,
) -> Any:
    await emit({
        "event": "agent_start",
        "agent": AGENT_SAP_DATA_DEFINITION,
        "message": "Defining SAP ticket data (schema, seed rows, ticket count)…",
    })
    await _progress(emit, AGENT_SAP_DATA_DEFINITION, "Dataset: source_sap_tickets + columns")
    await _progress(emit, AGENT_SAP_DATA_DEFINITION, "Blocks: sapCommunityFeed, kpiRow")
    definition = await resolve_coro()
    n_tickets = 0
    try:
        bag = (definition.mock or definition.model_dump().get("data")) if hasattr(definition, "mock") else {}
        if not bag and hasattr(definition, "model_dump"):
            bag = definition.model_dump().get("data") or {}
        rows = bag.get("source_sap_tickets") if isinstance(bag, dict) else []
        n_tickets = len(rows) if isinstance(rows, list) else 0
    except Exception:
        pass
    await _progress(
        emit,
        AGENT_SAP_DATA_DEFINITION,
        f"Seed tickets: {n_tickets or definition.meta.sapTicketSeedCount or 'default'}",
    )
    await emit({
        "event": "agent_complete",
        "agent": AGENT_SAP_DATA_DEFINITION,
        "message": "SAP data definition ready (RAG context comes in step 2).",
    })
    return definition


async def run_sap_rag_phase_pending(emit: EmitFn) -> None:
    await emit({
        "event": "agent_start",
        "agent": AGENT_SAP_RAG,
        "message": "RAG agent — waiting for your step 2 prompt…",
    })
    await _progress(
        emit,
        AGENT_SAP_RAG,
        "After codegen, send prompt 2: SAP data + Community search + per-ticket RAG.",
    )
    await emit({
        "event": "agent_complete",
        "agent": AGENT_SAP_RAG,
        "message": "Pending — run step 2 update to search SAP Community and build ticket context.",
    })


async def run_sap_integration_phase(
    emit: EmitFn,
    *,
    definition: Any,
    file_count: int,
    data_json_kb: float,
) -> None:
    await emit({
        "event": "agent_start",
        "agent": AGENT_SAP_INTEGRATION,
        "message": "Wiring template, chat assistant, and data.json…",
    })
    await _progress(emit, AGENT_SAP_INTEGRATION, f"Writing data.json ({data_json_kb:.1f} KB)")
    await _progress(emit, AGENT_SAP_INTEGRATION, "Similarity Tickets assistant + Community feed UI")
    for nav in definition.navigation[:6]:
        await _progress(emit, AGENT_SAP_INTEGRATION, f"Route {nav.path} → {nav.pageId}")
    await _progress(emit, AGENT_SAP_INTEGRATION, f"Project files: {file_count}")
    await emit({
        "event": "agent_complete",
        "agent": AGENT_SAP_INTEGRATION,
        "message": "Integration complete — open the app, then send step 2 for SAP RAG.",
    })