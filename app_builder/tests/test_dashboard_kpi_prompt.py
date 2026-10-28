import unittest

from app_builder.schemas.app_definition import AppDefinition, AppMeta, AppTheme, NavItemDef, PageDef
from app_builder.services.app_definition_generator import finalize_app_definition
from app_builder.services.page_layout_intent import enforce_kpis_on_dashboard_only
from app_builder.services.prd_driven_layout import build_kpi_items_from_context, kpi_items_overlap


class TestDashboardKpiPrompt(unittest.TestCase):
    def test_build_kpi_items_from_user_requirement_only(self):
        requirement = """
Build a maintenance dashboard.

Dashboard KPIs:
- Total Maintenance Work Orders
- Mean Time Between Failures (MTBF)
- Mean Time to Repair (MTTR)
- Equipment Downtime (hours)
"""
        items = build_kpi_items_from_context("", "dashboard", requirement=requirement, prd="")
        labels = [i["label"].lower() for i in items]
        self.assertGreaterEqual(len(items), 3)
        self.assertTrue(any("work order" in lb for lb in labels))
        self.assertTrue(any("mtbf" in lb for lb in labels))

    def test_enforce_replaces_generic_llm_kpi_row(self):
        requirement = """
Maintenance operations dashboard with KPIs:
- Total Maintenance Work Orders
- MTBF
- MTTR
"""
        defn = AppDefinition(
            meta=AppMeta(appName="Maint", description="Maintenance", domain="operations"),
            theme=AppTheme(),
            navigation=[NavItemDef(label="Dashboard", path="/", icon="Dashboard", pageId="dashboard")],
            pages=[
                PageDef(
                    id="dashboard",
                    title="Dashboard",
                    subtitle="Overview",
                    blocks=[
                        {
                            "type": "kpiRow",
                            "dataKey": "kpis_dashboard",
                            "items": [
                                {"label": "OEE", "valueKey": "oee", "color": "primary"},
                                {"label": "Throughput", "valueKey": "throughput", "color": "success"},
                                {"label": "Output", "valueKey": "output", "color": "warning"},
                            ],
                        }
                    ],
                )
            ],
            mock={"kpis_dashboard": {"oee": "92%", "throughput": "1,200", "output": "880"}},
        )
        out = enforce_kpis_on_dashboard_only(defn, prd="", requirement=requirement)
        dash = out.pages[0]
        kpi = next(b for b in dash.blocks if b.get("type") == "kpiRow")
        labels = " ".join(i["label"].lower() for i in kpi["items"])
        self.assertIn("mtbf", labels)
        self.assertNotIn("throughput", " ".join(i.get("valueKey", "") for i in kpi["items"]))

    def test_finalize_fills_mock_for_prompt_kpis(self):
        requirement = """
Dashboard KPIs:
- Total Maintenance Work Orders
- MTBF
- MTTR
"""
        defn = AppDefinition(
            meta=AppMeta(appName="Maint", description="Maintenance", domain="operations"),
            theme=AppTheme(),
            navigation=[NavItemDef(label="Dashboard", path="/", icon="Dashboard", pageId="dashboard")],
            pages=[
                PageDef(
                    id="dashboard",
                    title="Dashboard",
                    subtitle="Overview",
                    blocks=[
                        {
                            "type": "kpiRow",
                            "dataKey": "kpis_dashboard",
                            "items": [
                                {"label": "OEE", "valueKey": "oee", "color": "primary"},
                                {"label": "Efficiency", "valueKey": "efficiency", "color": "success"},
                            ],
                        }
                    ],
                )
            ],
            mock={},
        )
        out = finalize_app_definition(defn, prd="", requirement=requirement)
        kpi = next(b for b in out.pages[0].blocks if b.get("type") == "kpiRow")
        bundle = out.mock.get("kpis_dashboard") or {}
        for item in kpi["items"]:
            self.assertIn(item["valueKey"], bundle)
            self.assertTrue(str(bundle[item["valueKey"]]).strip())

    def test_overlap_helper(self):
        existing = [{"label": "OEE", "valueKey": "oee"}]
        expected = [{"label": "MTBF", "valueKey": "mtbf"}, {"label": "MTTR", "valueKey": "mttr"}]
        self.assertLess(kpi_items_overlap(existing, expected), 0.5)


if __name__ == "__main__":
    unittest.main()
