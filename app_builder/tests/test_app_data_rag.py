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
