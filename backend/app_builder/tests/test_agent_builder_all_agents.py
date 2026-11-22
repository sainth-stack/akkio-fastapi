"""Agent builder coverage: routing, planning agents, and json_app codegen agents.

No live LLM. App Definition uses the PRD path; pipeline agents are exercised
through their public phase functions and the events they emit.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
import unittest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

from app_builder.schemas.app_definition_llm import AppShellPlan
from app_builder.services.agent_pipeline_events import (
    AGENT_APP_DEFINITION,
    AGENT_CODING,
    AGENT_INTEGRATION,
    AGENT_PLANNER,
    AGENT_STYLING,
    AGENT_VALIDATOR,
    PLANNING_STEP_AGENT,
    PLANNING_STEP_COMPLETE,
)
from app_builder.services.app_definition_generator import (
    build_shell_plan_from_screens,
    extract_screens_from_plan,
    resolve_app_definition_sync,
    resolve_app_display_name,
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

SAMPLE = (
    "Build a manufacturing vendor dashboard (Dashboard, Analytics, Reports) "
    "for warehouse stock, safety stock, reorder points, and stock-outs, "
    "with RAG chat, Predict, and LLM observability to track prompts, outputs, tokens, and model cost."
)

PRD = """
### Dashboard
**Purpose:** Warehouse stock health

**KPIs:**
- Stock-outs
- SKUs below safety stock

**Charts:**
- On-hand trend (line chart)
- Stock by warehouse (bar chart)

### Analytics
**Charts:**
- Reorder pressure (bar chart)

### Reports
**Charts:**
- Stock-out trend (line chart)
"""


def _events() -> tuple[List[Dict[str, Any]], Any]:
    log: List[Dict[str, Any]] = []

    async def emit(payload: Dict[str, Any]) -> None:
        log.append(payload)

    return log, emit


class TestPlanningAgents(unittest.TestCase):
    def test_planning_steps_map_to_known_agents(self):
        self.assertEqual(set(PLANNING_STEP_AGENT), {"prd", "uiux", "style", "architecture", "plan"})
        for step, (agent, message) in PLANNING_STEP_AGENT.items():
            self.assertIn(agent, {AGENT_PLANNER, AGENT_STYLING})
            self.assertTrue(message)
            self.assertIn(step, PLANNING_STEP_COMPLETE)


class TestJsonAppAgentPhases(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tick = patch(
            "app_builder.services.json_app_codegen_pipeline._tick",
            new=AsyncMock(),
        )
        self._tick.start()

    async def asyncTearDown(self):
        self._tick.stop()

    async def test_planner_styling_definition_coding_integration_validator(self):
        log, emit = _events()
        screens = await run_planner_phase(
            emit,
            app_name="Manufacturing Vendor",
            requirement=SAMPLE,
            prd_text=PRD,
            uiux_text="Sidebar + KPI cards",
            plan_json={"blueprint": {"domain": "manufacturing"}, "architecture": {"routes": []}},
        )
        self.assertEqual([s["title"] for s in screens], ["Dashboard", "Analytics", "Reports"])
        self.assertEqual(log[0]["agent"], AGENT_PLANNER)
        self.assertEqual(log[0]["event"], "agent_start")
        self.assertEqual(log[-1]["event"], "agent_complete")

        log.clear()
        await run_styling_phase(
            emit,
            plan_json={"design_tokens": {"light": {"primary": "#1976d2"}}},
            app_name="Manufacturing Vendor",
        )
        agents = {e["agent"] for e in log}
        self.assertEqual(agents, {AGENT_STYLING})
        self.assertTrue(any("1976d2" in e.get("message", "") for e in log))

        definition = resolve_app_definition_sync(
            requirement=SAMPLE,
            prd=PRD,
            app_name="Manufacturing Vendor",
        )

        async def _resolve():
            return definition

        log.clear()
        out = await run_app_definition_phase(emit, screens=screens, resolve_coro=_resolve)
        self.assertIs(out, definition)
        self.assertEqual(log[0]["agent"], AGENT_APP_DEFINITION)
        self.assertEqual(log[-1]["event"], "agent_complete")

        log.clear()
        await run_coding_phase(emit, definition=definition, file_count=12, data_json_kb=4.2)
        self.assertEqual({e["agent"] for e in log}, {AGENT_CODING})

        log.clear()
        await run_integration_phase(emit, definition=definition, template_name="base-frontend-vite-mui")
        self.assertEqual({e["agent"] for e in log}, {AGENT_INTEGRATION})
        self.assertTrue(any("/" in e.get("message", "") for e in log))

        log.clear()
        await run_validator_handoff(emit)
        self.assertEqual(log[0]["agent"], AGENT_VALIDATOR)
        self.assertEqual(log[0]["event"], "agent_start")


class TestManufacturingVendorFlow(unittest.TestCase):
    def test_routes_to_json_app_not_anomaly(self):
        from app_builder.services.fullstack_anomaly_generator import is_anomaly_detection_domain

        self.assertTrue(should_force_json_app_track(SAMPLE, PRD))
        self.assertFalse(is_anomaly_detection_domain(SAMPLE, prd=PRD))

    def test_shell_fallback_matches_parsed_screens(self):
        screens = extract_screens_from_plan(requirement=SAMPLE, prd=PRD)
        name = resolve_app_display_name(requirement=SAMPLE, app_name="Manufacturing Vendor")
        shell = build_shell_plan_from_screens(screens, app_name=name, requirement=SAMPLE)
        self.assertIsInstance(shell, AppShellPlan)
        self.assertEqual([p.id for p in shell.pages], ["dashboard", "analytics", "reports"])
        self.assertEqual(shell.navigation[0].path, "/")
        self.assertEqual(shell.domain, "inventory")

    def test_definition_includes_rag_predict_and_observability(self):
        app = resolve_app_definition_sync(requirement=SAMPLE, prd=PRD, app_name="Manufacturing Vendor")
        page_ids = {p.id for p in app.pages}
        nav_ids = {n.pageId for n in app.navigation}
        self.assertTrue({"dashboard", "analytics", "reports"} <= page_ids)
        self.assertIn("predict", nav_ids)
        self.assertIn("observability", nav_ids)
        caps = set(app.meta.capabilities or [])
        self.assertIn("rag_chat", caps)
        self.assertIn("llm_observability", caps)
        dash = next(p for p in app.pages if p.id == "dashboard")
        types = [b.get("type") for b in dash.blocks if isinstance(b, dict)]
        self.assertIn("kpiRow", types)

    def test_coding_agent_writes_data_json_renderer(self):
        app = resolve_app_definition_sync(requirement=SAMPLE, prd=PRD, app_name="Manufacturing Vendor")
        files = generate_json_app_files(requirement=SAMPLE, app_definition=app)
        self.assertIn("frontend/src/mock/data.json", files)
        self.assertTrue(any("PageRenderer" in path for path in files))
        payload = json.loads(files["frontend/src/mock/data.json"])
        ids = [p["id"] for p in payload["pages"]]
        self.assertIn("dashboard", ids)
        self.assertIn("predict", ids)
        self.assertIn("observability", ids)
        self.assertTrue(payload.get("data"))


class TestAppDefinitionAgentWithoutLiveLlm(unittest.IsolatedAsyncioTestCase):
    async def test_shell_llm_failure_still_builds_definition(self):
        from app_builder.services.app_definition_llm_pipeline import generate_app_definition_llm

        sys.modules["llm_helper"] = types.SimpleNamespace(
            get_llm_for_user=lambda *a, **k: object(),
        )

        async def no_llm(*_a, **_k):
            return None

        progress: List[str] = []

        async def on_progress(msg: str) -> None:
            progress.append(msg)

        with patch(
            "app_builder.services.app_definition_llm_pipeline._llm_validated",
            new=no_llm,
        ):
            defn = await generate_app_definition_llm(
                requirement=SAMPLE,
                prd=PRD,
                app_name="Manufacturing Vendor",
                on_progress=on_progress,
            )

        self.assertIsNotNone(defn)
        self.assertEqual(
            [p.id for p in defn.pages if p.id in ("dashboard", "analytics", "reports")],
            ["dashboard", "analytics", "reports"],
        )
        self.assertTrue(any(m.startswith("LLM phase 1/3") for m in progress))
        self.assertTrue(any("blocks for" in m for m in progress))

    async def test_pipeline_exception_falls_back_to_prd_definition(self):
        from app_builder.services.app_definition_generator import resolve_app_definition

        async def boom(**_kwargs):
            return None

        with patch(
            "app_builder.services.app_definition_llm_pipeline.generate_app_definition_llm",
            new=boom,
        ), patch(
            "app_builder.services.app_definition_generator.generate_app_definition_with_llm",
            new=AsyncMock(return_value=None),
        ):
            app = await resolve_app_definition(
                requirement=SAMPLE,
                prd=PRD,
                app_name="Manufacturing Vendor",
                use_llm=True,
            )

        self.assertEqual(app.meta.generatedBy, "prd_fallback")
        self.assertGreaterEqual(len(app.pages), 3)


if __name__ == "__main__":
    unittest.main()
