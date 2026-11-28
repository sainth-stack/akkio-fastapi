import asyncio
import unittest


class TestAppDataRag(unittest.TestCase):
    def test_build_rag_context_includes_kpi_and_chart(self):
        from app_builder.services.app_data_rag import build_rag_context

        defn = {
            "meta": {"appName": "Plant Ops", "domain": "manufacturing"},
            "pages": [
                {
                    "id": "dashboard",
                    "title": "Dashboard",
                    "blocks": [
                        {
                            "type": "kpiRow",
                            "dataKey": "kpis_dashboard",
                            "items": [{"label": "MTBF", "valueKey": "mtbf_hours"}],
                        },
                        {
                            "type": "lineChart",
                            "title": "Downtime trend",
                            "dataKey": "series_dash",
                            "xAxisKey": "period",
                            "series": [{"dataKey": "downtime", "name": "Downtime"}],
                        },
                    ],
                }
            ],
            "data": {
                "kpis_dashboard": {"mtbf_hours": "240 h"},
                "series_dash": [
                    {"period": "2024-06-01", "downtime": 2.1},
                    {"period": "2024-06-02", "downtime": 1.8},
                ],
            },
        }
        ctx = build_rag_context(defn, "What is MTBF and downtime trend?")
        self.assertIn("MTBF", ctx)
        self.assertIn("240", ctx)
        self.assertIn("Downtime trend", ctx)
        self.assertIn("downtime", ctx.lower())

    def test_manufacturing_export_has_knowledge_sales_and_prompt_policies(self):
        from app_builder.services.app_data_rag import build_policy_rag_export

        prompt = "Manufacturing vendor dashboard for inspection lots, safety stock, and purchase orders in 2025 and 2026"
        defn = {
            "meta": {
                "appName": "Plant Ops",
                "domain": "manufacturing",
                "requirement": prompt,
            },
            "pages": [
                {
                    "id": "dashboard",
                    "title": "Dashboard",
                    "blocks": [
                        {
                            "type": "lineChart",
                            "title": "Total Purchase Orders over time",
                            "dataKey": "chart_po",
                            "xAxisKey": "period",
                            "series": [{"dataKey": "order_value", "name": "Spend"}],
                        }
                    ],
                }
            ],
            "data": {
                "chart_po": [
                    {"period": "Jan 2025", "order_value": 1200},
                    {"period": "Feb 2025", "order_value": 800},
                ],
                "table_orders": [
                    {"po_number": "PO-8801", "vendor": "Apex Metals", "order_value": 4200, "ordered_at": "2025-03-02"},
                    {"po_number": "PO-8802", "vendor": "Nova Components", "order_value": 1800, "ordered_at": "2026-01-14"},
                ],
            },
        }
        model_policies = (
            "- Inspection policy: Hold a lot when it fails the inspection rule named in the prompt.\n"
            "- Stock policy: Replenish a SKU that falls to its safety stock."
        )
        seen = {}

        class _Reply:
            content = model_policies

        class _Llm:
            async def ainvoke(self, messages):
                seen["prompt"] = messages[-1].content
                return _Reply()

        text = asyncio.run(build_policy_rag_export(defn, llm=_Llm()))
        self.assertIn(prompt, seen["prompt"])
        self.assertIn("Knowledge", text)
        self.assertIn("Manufacturing sales", text)
        self.assertIn("PO-8801", text)
        self.assertIn("Policies", text)
        self.assertIn("Inspection policy", text)
        self.assertIn("Stock policy", text)
        self.assertNotIn("kpis_", text)
        self.assertNotIn("series_", text)


    def test_customer_service_export_is_policies_not_kpis(self):
        from app_builder.services.app_data_rag import build_policy_rag_export

        prompt = (
            "Build a customer service operation app where the app create tickets, "
            "analyse and monitor the tickets and also create in app assistant with rag functionality"
        )
        defn = {
            "meta": {"appName": "Workspace", "domain": "generic", "description": prompt},
            "pages": [{"id": "predict", "title": "Predict", "subtitle": "Run what-if forecasts"}],
            "data": {
                "kpis_dashboard": {
                    "total_tickets_created": "17,054",
                    "average_resolution_time": "27h 1m",
                    "customer_satisfaction_score": "16,399",
                    "tickets_resolved_today": "2,299",
                },
                "series_dashboard_1_line": [{"period": "Jan 2025", "value": 12}],
                "table_tickets": [{"ticket_id": "TKT-2025-0001", "status": "Open"}],
            },
        }
        model_policies = (
            "- Ticket creation policy: Create a ticket when a customer contacts the desk.\n"
            "- Assistant policy: The in-app assistant answers from these policies."
        )

        class _Reply:
            content = model_policies

        class _Llm:
            async def ainvoke(self, messages):
                assert prompt in messages[-1].content
                return _Reply()

        text = asyncio.run(build_policy_rag_export(defn, llm=_Llm()))
        self.assertIn("Policies", text)
        self.assertIn("Ticket creation policy", text)
        self.assertIn("Assistant policy", text)
        self.assertIn(prompt, text)
        self.assertNotIn("17,054", text)
        self.assertNotIn("kpis_dashboard", text)
        self.assertNotIn("series_dashboard", text)
        self.assertNotIn("customer_satisfaction_score", text)

    def test_sap_rag_export_ticket_and_solution_only(self):
        from app_builder.services.app_data_rag import build_sap_rag_export
        from app_builder.services.sap_similar_tickets import seed_sap_tickets_if_empty

        defn = seed_sap_tickets_if_empty(
            {"meta": {"appName": "Sap Similarity", "domain": "sap", "capabilities": ["sap_similar_tickets"]}}
        )
        text = build_sap_rag_export(defn)
        self.assertIn("Ticket SCN-58841", text)
        self.assertNotIn("Open tickets:", text)
        self.assertNotIn("kpis_dashboard", text)
        self.assertNotIn("Units on hand", text)
        self.assertNotIn("SAP ticket store (10 records)", text)
        ids = text.count("Ticket SCN-") + text.count("Ticket TKT-") + text.count("Ticket RETMFG")
        self.assertGreaterEqual(ids, 8)

    def test_ecommerce_context_includes_products(self):
        from app_builder.services.app_data_rag import build_rag_context

        defn = {
            "appKind": "ecommerce",
            "meta": {"appName": "Tech Shop", "domain": "electronics"},
            "products": [
                {"id": 1, "name": "Laptop", "price": 50000, "stock": 20, "description": "15 inch"},
            ],
            "dashboard": {"total_orders": 2, "total_spent": 50000},
        }
        ctx = build_rag_context(defn, "Which laptop do you sell?")
        self.assertIn("Laptop", ctx)
        self.assertIn("50000", ctx)


if __name__ == "__main__":
    unittest.main()
