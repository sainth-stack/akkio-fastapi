import json
import unittest


class TestAppDefinitionGenerator(unittest.TestCase):
    def test_screens_from_requirement_phrase(self):
        from app_builder.services.app_definition_generator import extract_screens_from_plan

        screens = extract_screens_from_plan(
            requirement="generate manufacturing monitoring app with dashboard, analytics, reports",
        )
        titles = [s["title"].lower() for s in screens]
        self.assertIn("dashboard", titles)
        self.assertIn("analytics", titles)
        self.assertIn("reports", titles)

    def test_parenthetical_pages_ignore_capability_with_clause(self):
        from app_builder.services.app_definition_generator import extract_screens_from_plan

        screens = extract_screens_from_plan(
            requirement=(
                "Build a manufacturing operations dashboard (Dashboard, Analytics, Reports) "
                "with RAG chat, Predict, and LLM observability to track prompts, outputs, tokens, and model cost."
            ),
        )
        titles = [s["title"].lower() for s in screens]
        ids = [s["id"] for s in screens]
        self.assertEqual(len(screens), 3)
        self.assertIn("dashboard", ids)
        self.assertIn("analytics", ids)
        self.assertIn("reports", ids)
        self.assertFalse(any("token" in t for t in titles))
        self.assertFalse(any("observability" in t and "track" in t for t in titles))

    def test_resolve_sync_builds_three_pages(self):
        from app_builder.services.app_definition_generator import resolve_app_definition_sync

        app = resolve_app_definition_sync(
            requirement="manufacturing monitoring with dashboard, analytics, reports",
            plan_json={"blueprint": {"selected_domain": "manufacturing"}},
        )
        page_ids = [p.id for p in app.pages]
        self.assertEqual(len(app.pages), 3)
        self.assertIn("dashboard", page_ids)
        self.assertIn("analytics", page_ids)
        self.assertIn("reports", page_ids)
        self.assertNotIn("materials", page_ids)
        self.assertTrue(app.mock)

    def test_generate_json_app_files_from_requirement(self):
        from app_builder.services.json_app_generator import generate_json_app_files

        files = generate_json_app_files(
            requirement="CRM with dashboard, contacts, and deals",
            plan_json={"blueprint": {"selected_domain": "retail"}},
        )
        self.assertIn("frontend/src/mock/data.json", files)
        data = json.loads(files["frontend/src/mock/data.json"])
        nav_labels = [n["label"].lower() for n in data["navigation"]]
        self.assertIn("dashboard", nav_labels)
        self.assertIn("data", data)
        self.assertNotIn("mock", data)
        self.assertNotIn('"oee": "1,240"', json.dumps(data["data"]))

    def test_procurement_prompt_not_quality_dataset(self):
        from app_builder.services.app_definition_generator import (
            extract_screens_from_plan,
            resolve_app_definition_sync,
        )
        from app_builder.services.source_dataset_planner import infer_source_datasets

        req = (
            "Build a manufacturing procurement dashboard (Dashboard, Analytics, Reports) "
            "with RAG chat, Predict, and LLM observability to track prompts, outputs, tokens, and model cost."
        )
        screens = extract_screens_from_plan(requirement=req)
        ids = [s["id"] for s in screens]
        self.assertEqual(ids, ["dashboard", "analytics", "reports"])

        app = resolve_app_definition_sync(
            requirement=req,
            plan_json={"blueprint": {"selected_domain": "manufacturing"}},
        )
        self.assertIn("predict", [c for c in (app.meta.capabilities or [])])
        self.assertIn("llm_observability", [c for c in (app.meta.capabilities or [])])
        defn_dict = app.to_json_dict()
        datasets = infer_source_datasets(defn_dict, requirement=req, prd="")
        self.assertEqual(datasets[0]["id"], "purchase_orders")
        self.assertNotEqual(datasets[0]["id"], "quality_inspections")

    def test_supply_chain_prompt_clears_prior_sap_tickets(self):
        from app_builder.services.operational_metrics_compute import apply_source_data_pipeline
        from app_builder.services.sap_similar_tickets import is_sap_similarity_app

        req = (
            "Build a manufacturing supply chain dashboard (Dashboard, Analytics, Reports) "
            "with RAG chat, Predict, and LLM observability to track prompts, outputs, tokens, and model cost."
        )
        stale = {
            "meta": {
                "appName": "Sap Similarity",
                "domain": "sap",
                "capabilities": ["sap_similar_tickets", "rag_chat", "predict"],
            },
            "navigation": [
                {"label": "Dashboard", "path": "/", "icon": "Dashboard", "pageId": "dashboard"},
            ],
            "pages": [
                {
                    "id": "dashboard",
                    "title": "Dashboard",
                    "blocks": [
                        {
                            "type": "kpiRow",
                            "dataKey": "kpis_dashboard",
                            "items": [{"label": "Open tickets", "valueKey": "open_tickets"}],
                        },
                        {
                            "type": "dataTable",
                            "title": "SAP tickets",
                            "dataKey": "table_sap_tickets",
                            "columns": [],
                        },
                    ],
                }
            ],
            "data": {
                "source_sap_tickets": [{"ticket_id": "SAP-0001", "status": "Open"}],
                "kpis_dashboard": {"open_tickets": "9"},
            },
        }
        out = apply_source_data_pipeline(stale, requirement=req, prd="")
        data = out.get("data") or {}
        self.assertNotIn("source_sap_tickets", data)
        self.assertFalse(is_sap_similarity_app(out))
        self.assertNotEqual(str((out.get("meta") or {}).get("domain")), "sap")
        pc = (out.get("meta") or {}).get("predictConfig") or {}
        self.assertTrue(pc.get("fields"), "Predict should have numeric fields after supply-chain recompute")

    def test_procurement_pipeline_not_sap_ticket_dashboard(self):
        from app_builder.services.app_definition_generator import resolve_app_definition_sync
        from app_builder.services.operational_metrics_compute import apply_source_data_pipeline
        from app_builder.services.sap_similar_tickets import is_sap_similarity_app

        req = (
            "Build a manufacturing procurement dashboard (Dashboard, Analytics, Reports) "
            "with RAG chat, Predict, and LLM observability to track prompts, outputs, tokens, and model cost."
        )
        app = resolve_app_definition_sync(
            requirement=req,
            plan_json={"blueprint": {"selected_domain": "manufacturing"}},
        )
        out = apply_source_data_pipeline(app.to_json_dict(), requirement=req, prd="")
        self.assertFalse(is_sap_similarity_app(out))
        data = out.get("data") or {}
        self.assertNotIn("source_sap_tickets", data)
        self.assertNotIn("table_sap_tickets", data)
        dash = next(p for p in (out.get("pages") or []) if p.get("id") == "dashboard")
        for block in dash.get("blocks") or []:
            if isinstance(block, dict):
                self.assertNotEqual(block.get("dataKey"), "table_sap_tickets")
                self.assertNotIn("ticket", (block.get("title") or "").lower())

    def test_production_kpi_bundles_per_page(self):
        from app_builder.services.json_app_generator import generate_json_app_files

        files = generate_json_app_files(
            requirement="manufacturing monitoring with dashboard, analytics, reports",
            plan_json={"blueprint": {"selected_domain": "manufacturing"}},
            app_name="PulsePlant",
        )
        payload = json.loads(files["frontend/src/mock/data.json"])
        bag = payload["data"]
        self.assertIn("kpis_dashboard", bag)
        self.assertTrue(len(bag.get("kpis_dashboard", {})) >= 1)
        self.assertGreaterEqual(len(bag), 3)


if __name__ == "__main__":
    unittest.main()
