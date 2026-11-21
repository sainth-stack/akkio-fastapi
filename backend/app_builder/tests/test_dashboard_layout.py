import unittest


class TestDashboardLayout(unittest.TestCase):
    def test_analytics_table_moves_to_dashboard(self):
        from app_builder.schemas.app_definition import AppDefinition, AppMeta, NavItemDef, PageDef
        from app_builder.services.dashboard_layout import consolidate_events_table_on_dashboard

        defn = AppDefinition(
            meta=AppMeta(appName="Maintenance", description="", domain="generic"),
            navigation=[
                NavItemDef(label="Dashboard", path="/", icon="Dashboard", pageId="dashboard"),
                NavItemDef(label="Analytics", path="/analytics", icon="Assessment", pageId="analytics"),
            ],
            pages=[
                PageDef(
                    id="dashboard",
                    title="Dashboard",
                    blocks=[
                        {
                            "type": "kpiRow",
                            "dataKey": "kpis_dashboard",
                            "items": [
                                {"label": "MTBF", "valueKey": "mtbf_hours"},
                                {"label": "MTTR", "valueKey": "mttr_hours"},
                            ],
                        },
                        {
                            "type": "lineChart",
                            "title": "Operational trends over time",
                            "dataKey": "chart_dashboard",
                            "xAxisKey": "label",
                            "series": [],
                        },
                    ],
                ),
                PageDef(
                    id="analytics",
                    title="Analytics",
                    blocks=[
                        {
                            "type": "dataTable",
                            "title": "Events",
                            "dataKey": "table_analytics",
                            "columns": [{"field": "machine", "header": "Machine"}],
                        },
                        {"type": "barChart", "title": "Performance by category", "dataKey": "chart_analytics"},
                    ],
                ),
            ],
            mock={},
        )
        out = consolidate_events_table_on_dashboard(defn, prd="manufacturing maintenance MTBF MTTR")
        dash = next(p for p in out.pages if p.id == "dashboard")
        anal = next(p for p in out.pages if p.id == "analytics")
        dash_types = [b.get("type") for b in dash.blocks if isinstance(b, dict)]
        anal_types = [b.get("type") for b in anal.blocks if isinstance(b, dict)]
        self.assertIn("dataTable", dash_types)
        self.assertNotIn("dataTable", anal_types)
        table = next(b for b in dash.blocks if b.get("type") == "dataTable")
        self.assertEqual(table.get("dataKey"), "table_dashboard")
        line = next(b for b in dash.blocks if b.get("type") == "lineChart")
        keys = {s.get("dataKey") for s in line.get("series", [])}
        self.assertTrue(keys & {"mtbf_hours", "mttr_hours", "downtime_hours"})


if __name__ == "__main__":
    unittest.main()
