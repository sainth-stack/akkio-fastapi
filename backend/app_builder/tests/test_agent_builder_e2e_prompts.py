"""End-to-end agent builder flows for several real prompts.

Covers routing, planner screens, App Definition, feature pages
(RAG / Predict / Observability), and json_app file output.
No live LLM calls.
"""
from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import AsyncMock, patch

from app_builder.services.agent_pipeline_events import (
    AGENT_APP_DEFINITION,
    AGENT_CODING,
    AGENT_INTEGRATION,
    AGENT_PLANNER,
    AGENT_STYLING,
    AGENT_VALIDATOR,
)
from app_builder.services.app_definition_generator import (
    extract_screens_from_plan,
    resolve_app_definition_sync,
)
from app_builder.services.json_app_codegen_pipeline import (
    run_app_definition_phase,
    run_coding_phase,
    run_integration_phase,
    run_planner_phase,
    run_styling_phase,
    run_validator_handoff,
)
from app_builder.services.json_app_generator import generate_json_app_files
from app_builder.services.json_app_routing import should_force_json_app_track


@dataclass
class PromptCase:
    name: str
    requirement: str
    json_app: bool
    screen_ids: Sequence[str] = ()
    required_pages: Sequence[str] = ()
    required_caps: Sequence[str] = ()
    forbidden_pages: Sequence[str] = ()
    domain: Optional[str] = None
    prd: str = ""


def _mini_prd(pages: Sequence[str]) -> str:
    chunks = []
    for title in pages:
        chunks.append(
            f"### {title}\n"
            "**KPIs:**\n- Open items\n- On-time rate\n\n"
            "**Charts:**\n- Trend over time (line chart)\n"
        )
    return "\n".join(chunks)


CASES: List[PromptCase] = [
    PromptCase(
        name="manufacturing_inventory",
        requirement=(
            "Build a manufacturing vendor dashboard (Dashboard, Analytics, Reports) "
            "for warehouse stock, safety stock, reorder points, and stock-outs, "
            "with RAG chat, Predict, and LLM observability to track prompts, outputs, tokens, and model cost."
        ),
        json_app=True,
        screen_ids=("dashboard", "analytics", "reports"),
        required_pages=("dashboard", "analytics", "reports", "predict", "observability"),
        required_caps=("rag_chat", "predict", "llm_observability"),
    ),
    PromptCase(
        name="plant_maintenance",
        requirement=(
            "Create a plant maintenance dashboard (Dashboard, Analytics, Reports) "
            "tracking MTBF, MTTR, downtime, and work orders, with RAG chat and Predict."
        ),
        json_app=True,
        screen_ids=("dashboard", "analytics", "reports"),
        required_pages=("dashboard", "analytics", "reports", "predict"),
        required_caps=("rag_chat", "predict"),
        forbidden_pages=("observability",),
    ),
    PromptCase(
        name="vendor_quality",
        requirement=(
            "Build a vendor quality dashboard (Dashboard, Analytics, Reports) "
            "for inspection lots, defect rate, and yield percent."
        ),
        json_app=True,
        screen_ids=("dashboard", "analytics", "reports"),
        required_pages=("dashboard", "analytics", "reports"),
        required_caps=("rag_chat",),
    ),
    PromptCase(
        name="procurement",
        requirement=(
            "Build a manufacturing procurement dashboard (Dashboard, Analytics, Reports) "
            "for purchase orders, vendors, order value, and lead time."
        ),
        json_app=True,
        screen_ids=("dashboard", "analytics", "reports"),
        required_pages=("dashboard", "analytics", "reports"),
        required_caps=("rag_chat",),
    ),
    PromptCase(
        name="supply_chain",
        requirement=(
            "Build a supply chain dashboard (Dashboard, Analytics, Reports) "
            "for purchase orders, suppliers, delivery dates, and on-time delivery."
        ),
        json_app=True,
        screen_ids=("dashboard", "analytics", "reports"),
        required_pages=("dashboard", "analytics", "reports"),
        required_caps=("rag_chat",),
    ),
    PromptCase(
        name="quality_four_pages",
        requirement=(
            "Create a Manufacturing Quality Monitoring Dashboard with 4 pages:\n"
            "1. **Dashboard:** Quality KPIs\n"
            "2. **Sessions:** Inspection sessions\n"
            "3. **Reports:** Export reports\n"
            "4. **Analytics:** Trends\n"
        ),
        json_app=True,
        screen_ids=("dashboard", "sessions", "reports", "analytics"),
        required_pages=("dashboard", "sessions", "reports", "analytics"),
        required_caps=("rag_chat",),
    ),
    PromptCase(
        name="sap_similar_tickets",
        requirement=(
            "Build an SAP similar tickets app with a dashboard of SAP incidents, "
            "create ticket, and similar ticket search."
        ),
        json_app=True,
        required_pages=("create_ticket", "similar_tickets"),
        required_caps=("sap_similar_tickets", "rag_chat"),
        domain="sap",
    ),
    PromptCase(
        name="ecommerce_store",
        requirement="Build an ecommerce shopping cart and online store with a product catalog.",
        json_app=False,
        forbidden_pages=("predict", "observability"),
        required_caps=(),
    ),
]


class TestAgentBuilderEndToEndPrompts(unittest.TestCase):
    def test_every_prompt_routes_and_builds_files(self):
        for case in CASES:
            with self.subTest(prompt=case.name):
                self.assertEqual(
                    should_force_json_app_track(case.requirement, case.prd),
                    case.json_app,
                    case.name,
                )
                screens = extract_screens_from_plan(requirement=case.requirement, prd=case.prd)
                if case.screen_ids:
                    self.assertEqual([s["id"] for s in screens], list(case.screen_ids), case.name)

                prd_text = case.prd or (
                    _mini_prd([s["title"] for s in screens]) if screens else ""
                )
                app = resolve_app_definition_sync(
                    requirement=case.requirement,
                    prd=prd_text,
                    app_name=case.name,
                )
                page_ids = {p.id for p in app.pages}
                for pid in case.required_pages:
                    self.assertIn(pid, page_ids, f"{case.name} missing page {pid}")
                for pid in case.forbidden_pages:
                    self.assertNotIn(pid, page_ids, f"{case.name} unexpected page {pid}")
                caps = set(app.meta.capabilities or [])
                for cap in case.required_caps:
                    self.assertIn(cap, caps, f"{case.name} missing capability {cap}")
                if case.domain:
                    self.assertEqual(app.meta.domain, case.domain, case.name)

                files = generate_json_app_files(
                    requirement=case.requirement,
                    app_definition=app,
                    app_name=case.name,
                )
                self.assertIn("frontend/src/mock/data.json", files, case.name)
                self.assertTrue(
                    any("PageRenderer" in path for path in files),
                    case.name,
                )
                payload = json.loads(files["frontend/src/mock/data.json"])
                shipped = {p["id"] for p in payload["pages"]}
                for pid in case.required_pages:
                    self.assertIn(pid, shipped, f"{case.name} data.json missing {pid}")
                self.assertTrue(payload.get("navigation"), case.name)
                self.assertEqual(payload["meta"]["appName"], app.meta.appName)


class TestAgentPhasesAcrossPrompts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tick = patch(
            "app_builder.services.json_app_codegen_pipeline._tick",
            new=AsyncMock(),
        )
        self._tick.start()

    async def asyncTearDown(self):
        self._tick.stop()

    async def test_planner_through_validator_for_each_json_app_prompt(self):
        expected_agents = [
            AGENT_PLANNER,
            AGENT_STYLING,
            AGENT_APP_DEFINITION,
            AGENT_CODING,
            AGENT_INTEGRATION,
            AGENT_VALIDATOR,
        ]
        for case in CASES:
            if not case.json_app:
                continue
            with self.subTest(prompt=case.name):
                log: List[Dict[str, Any]] = []

                async def emit(payload: Dict[str, Any]) -> None:
                    log.append(payload)

                screens = await run_planner_phase(
                    emit,
                    app_name=case.name,
                    requirement=case.requirement,
                    prd_text=case.prd or _mini_prd(case.screen_ids or ("Dashboard",)),
                    uiux_text="Cards, charts, and a left navigation shell.",
                    plan_json={"blueprint": {"domain": case.domain or "manufacturing"}, "architecture": {"routes": []}},
                )
                if case.screen_ids:
                    self.assertEqual([s["id"] for s in screens], list(case.screen_ids))

                await run_styling_phase(
                    emit,
                    plan_json={"design_tokens": {"light": {"primary": "#0f766e"}}},
                    app_name=case.name,
                )
                prd_text = case.prd or (
                    _mini_prd([s["title"] for s in screens]) if screens else ""
                )
                app = resolve_app_definition_sync(
                    requirement=case.requirement,
                    prd=prd_text,
                    app_name=case.name,
                )

                async def _resolve():
                    return app

                await run_app_definition_phase(emit, screens=screens, resolve_coro=_resolve)
                await run_coding_phase(emit, definition=app, file_count=8, data_json_kb=2.0)
                await run_integration_phase(emit, definition=app, template_name="base-frontend-vite-mui")
                await run_validator_handoff(emit)

                seen = [e["agent"] for e in log if e.get("event") == "agent_start"]
                for agent in expected_agents:
                    self.assertIn(agent, seen, f"{case.name} did not start {agent}")
                completes = [e["agent"] for e in log if e.get("event") == "agent_complete"]
                for agent in (AGENT_PLANNER, AGENT_STYLING, AGENT_APP_DEFINITION, AGENT_CODING, AGENT_INTEGRATION):
                    self.assertIn(agent, completes, f"{case.name} did not complete {agent}")


if __name__ == "__main__":
    unittest.main()
