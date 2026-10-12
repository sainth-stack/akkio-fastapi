"""Deterministic static-domain codegen with WebSocket-friendly agent stages."""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Optional

from app_builder.services.fullstack_app_generator import generate_fullstack_application


async def _stage(
    emit: Callable,
    agent: str,
    start_msg: str,
    complete_msg: str,
    *,
    pause_ms: int = 120,
) -> None:
    await emit({
        "event": "agent_start",
        "agent": agent,
        "message": start_msg,
    })
    if pause_ms:
        await asyncio.sleep(pause_ms / 1000.0)
    await emit({
        "event": "agent_complete",
        "agent": agent,
        "message": complete_msg,
    })


async def run_legacy_static_codegen(
    *,
    domain: str,
    requirement: str,
    prd: str,
    uiux: str,
    architecture: Optional[Dict[str, Any]],
    design_tokens: Optional[Dict[str, Any]],
    on_event: Optional[Callable] = None,
) -> Dict[str, str]:
    """Build ecommerce / doc_chat / anomaly_detection without LLM codegen."""

    async def emit(payload: Dict[str, Any]) -> None:
        if on_event:
            await on_event(payload)

    domain_label = {
        "ecommerce": "E-Commerce",
        "doc_chat": "Document Chat",
        "anomaly_detection": "Anomaly Detection",
    }.get(domain, domain)

    await emit({
        "event": "pipeline_log",
        "agent": "codegen",
        "message": f"Track: legacy static | Domain: {domain_label} | Deterministic template (no LLM page codegen).",
    })

    catalog_note = ""
    if domain == "ecommerce":
        from app_builder.services.fullstack_ecommerce_generator import _detect_product_domain

        vertical = _detect_product_domain(requirement, prd, uiux)
        catalog_note = f" ({vertical} catalog)"

    await _stage(
        emit,
        "catalog_agent",
        f"Analyzing your prompt and selecting the {domain_label} template{catalog_note}…",
        f"Catalog and feature set locked for {domain_label}{catalog_note}.",
    )
    await _stage(
        emit,
        "frontend_agent",
        "Generating React storefront / UI pages, theme, and navigation…",
        "Frontend pages and cart shell are ready.",
    )
    await _stage(
        emit,
        "backend_agent",
        "Generating FastAPI routes, models, auth, and seed data…",
        "Backend API and database seed scripts are ready.",
    )
    await _stage(
        emit,
        "integration_agent",
        "Wiring frontend API client to backend with mock fallback…",
        "Integration layer connected.",
    )

    await emit({
        "event": "agent_start",
        "agent": "code_generator_agent",
        "message": f"Assembling {domain_label} application files…",
    })

    files = generate_fullstack_application(
        requirement,
        prd,
        uiux,
        architecture,
        design_tokens,
    )

    await emit({
        "event": "agent_complete",
        "agent": "code_generator_agent",
        "message": f"Generated {len(files)} files for {domain_label}.",
        "data": {"count": len(files), "files": list(files.keys())[:40]},
    })

    return files
