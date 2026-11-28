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
        from app_builder.services.app_data_rag import build_rag_context

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
        text = build_rag_context(defn, "export rag data")
        self.assertIn("Knowledge", text)
        self.assertIn("Manufacturing sales", text)
        self.assertIn("PO-8801", text)
        self.assertIn("Policies", text)
        self.assertIn("inspection lots", text)
        self.assertIn("Inventory policy", text)
        self.assertIn("Quality policy", text)
        self.assertIn("Procurement policy", text)
        self.assertNotIn("Maintenance policy", text)
        self.assertIn("Total Purchase Orders over time", text)


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
