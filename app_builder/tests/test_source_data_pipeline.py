import json
import unittest


class TestSourceDataPipeline(unittest.TestCase):
    def _manufacturing_defn(self):
        return {
            "meta": {"appName": "Ops", "domain": "manufacturing", "description": "MTBF maintenance"},
            "pages": [
                {
                    "id": "dashboard",
                    "title": "Dashboard",
                    "blocks": [
                        {
                            "type": "kpiRow",
                            "dataKey": "kpis_dashboard",
                            "items": [
                                {"label": "Total Maintenance Work Orders", "valueKey": "total_work_orders"},
                                {"label": "MTBF", "valueKey": "mtbf"},
                                {"label": "MTTR", "valueKey": "mttr"},
                            ],
                        }
                    ],
                },
                {
                    "id": "analytics",
                    "title": "Analytics",
                    "blocks": [
                        {
                            "type": "dataTable",
                            "dataKey": "table_analytics",
                            "columns": [
                                {"field": "work_order_id", "header": "Work order"},
                                {"field": "downtime_hours", "header": "Downtime (h)"},
                                {"field": "mtbf_hours", "header": "MTBF (h)"},
                                {"field": "mttr_hours", "header": "MTTR (h)"},
                                {"field": "status", "header": "Status"},
                            ],
                        }
                    ],
                },
            ],
            "navigation": [{"pageId": "dashboard", "path": "/", "label": "Dashboard"}],
            "data": {},
        }

    def test_template_has_operational_columns_not_label_value(self):
        from app_builder.services.source_dataset_planner import build_template_sheets
        from app_builder.services.source_data_constants import TEMPLATE_SAMPLE_ROWS
        from app_builder.services.operational_metrics_compute import apply_source_data_pipeline

        defn = apply_source_data_pipeline(self._manufacturing_defn())
        sheets = build_template_sheets(defn)
        self.assertTrue(sheets)
        first = next(iter(sheets.values()))
        self.assertIn("Work order", first[0].keys())
        self.assertNotIn("label", first[0])
        self.assertLessEqual(len(first), TEMPLATE_SAMPLE_ROWS)

    def test_template_ignores_huge_live_bag(self):
        from app_builder.services.source_dataset_planner import build_template_sheets
        from app_builder.services.source_data_constants import TEMPLATE_SAMPLE_ROWS
        from app_builder.services.operational_metrics_compute import apply_source_data_pipeline

        defn = apply_source_data_pipeline(self._manufacturing_defn())
        bag = defn.get("data") or {}
        for key in list(bag.keys()):
            if str(key).startswith("source_") and isinstance(bag[key], list):
                bag[key] = bag[key] * 20
                break
        defn["data"] = bag
        sheets = build_template_sheets(defn)
        rows = next(iter(sheets.values()))
        self.assertLessEqual(len(rows), TEMPLATE_SAMPLE_ROWS)

    def test_quality_charts_map_source_columns(self):
        from app_builder.services.operational_metrics_compute import apply_source_data_pipeline

        defn = {
            "meta": {"domain": "manufacturing", "description": "Quality FPY defect"},
            "pages": [
                {
                    "id": "dashboard",
                    "blocks": [
                        {
                            "type": "lineChart",
                            "title": "First Pass Yield (FPY) trend over time",
                            "dataKey": "chart_fpy_trend",
                            "xAxisKey": "period",
                            "series": [{"dataKey": "fpy", "name": "First-pass yield"}],
                        },
                        {
                            "type": "barChart",
                            "title": "Quality defects by production line",
                            "dataKey": "chart_defects_line",
                            "xAxisKey": "name",
                            "series": [{"dataKey": "defects", "name": "Defects"}],
                        },
                    ],
                }
            ],
            "data": {},
        }
        out = apply_source_data_pipeline(defn)
        bag = out.get("data") or {}
        trend = bag.get("chart_fpy_trend") or []
        self.assertTrue(trend)
        self.assertTrue(any((row.get("fpy") or 0) > 10 for row in trend))
        by_line = bag.get("chart_defects_line") or []
        self.assertTrue(by_line)
        self.assertIn("production_line", by_line[0])

    def test_upload_recomputes_kpis(self):
        from app_builder.services.data_import_service import merge_upload_bytes

        defn = self._manufacturing_defn()
        csv_body = (
            "Work order,Downtime (h),MTBF (h),MTTR (h),Status\n"
            "WO-1,2.0,400,1.5,Closed\n"
            "WO-2,3.0,380,2.0,Open\n"
        ).encode()
        updated, summary = merge_upload_bytes(
            defn,
            filename="maintenance_events.csv",
            raw=csv_body,
        )
        bag = updated.get("data") or {}
        self.assertIn("source_", summary["updatedKeys"][0])
        kpis = bag.get("kpis_dashboard") or {}
        self.assertIn("mtbf", kpis)
        self.assertIn("h", str(kpis.get("mtbf", "")))

    def test_inventory_all_nav_pages_get_widgets(self):
        from app_builder.services.operational_metrics_compute import apply_source_data_pipeline

        defn = {
            "version": 1,
            "meta": {"appName": "Inventory Visibility", "domain": "inventory"},
            "navigation": [
                {"label": "Dashboard", "path": "/", "icon": "Dashboard", "pageId": "dashboard"},
                {"label": "Inventory", "path": "/inventory", "icon": "Inventory", "pageId": "inventory"},
                {"label": "Reports", "path": "/reports", "icon": "Assessment", "pageId": "reports"},
            ],
            "pages": [
                {
                    "id": "dashboard",
                    "title": "Dashboard",
                    "blocks": [{"type": "kpiRow", "dataKey": "kpis_dashboard", "items": [{"label": "SKUs", "valueKey": "skus"}]}],
                },
                {"id": "inventory", "title": "Inventory", "subtitle": "Browse stock", "blocks": []},
                {"id": "reports", "title": "Reports", "subtitle": "Leadership summary", "blocks": []},
            ],
            "data": {},
        }
        out = apply_source_data_pipeline(defn, requirement="inventory visibility sku warehouse", prd="inventory app")
        for pid in ("dashboard", "inventory", "reports"):
            page = next(p for p in out["pages"] if p["id"] == pid)
            types = [
                b["type"]
                for b in page.get("blocks") or []
                if b.get("type") in ("kpiRow", "lineChart", "barChart", "pieChart", "dataTable")
            ]
            self.assertGreaterEqual(len(types), 2, msg=f"{pid} blocks={types}")


    def test_generic_dashboard_blocks_map_from_source(self):
        from app_builder.services.operational_metrics_compute import apply_source_data_pipeline

        defn = {
            "meta": {"domain": "manufacturing_maintenance", "description": "MTBF MTTR maintenance"},
            "pages": [
                {
                    "id": "dashboard",
                    "title": "Dashboard",
                    "blocks": [
                        {
                            "type": "lineChart",
                            "title": "Dashboard trend",
                            "dataKey": "chart_dashboard_trend",
                            "xAxisKey": "period",
                            "series": [{"dataKey": "value", "name": "Value"}],
                        },
                        {
                            "type": "barChart",
                            "title": "Dashboard breakdown",
                            "dataKey": "chart_dashboard_breakdown",
                            "xAxisKey": "category",
                            "series": [{"dataKey": "value", "name": "Value"}],
                        },
                        {
                            "type": "dataTable",
                            "title": "Dashboard records",
                            "dataKey": "table_dashboard",
                            "columns": [
                                {"field": "id", "header": "ID"},
                                {"field": "name", "header": "Name"},
                                {"field": "status", "header": "Status"},
                            ],
                        },
                    ],
                },
                {
                    "id": "analytics",
                    "title": "Analytics",
                    "blocks": [
                        {
                            "type": "barChart",
                            "title": "Maintenance events by machine",
                            "dataKey": "series_analytics_1_bar",
                            "xAxisKey": "category",
                            "series": [
                                {"dataKey": "actual", "name": "Actual"},
                                {"dataKey": "benchmark", "name": "Benchmark"},
                            ],
                        },
                    ],
                },
            ],
            "navigation": [{"pageId": "dashboard", "path": "/", "label": "Dashboard"}],
            "data": {},
        }
        out = apply_source_data_pipeline(defn)
        bag = out.get("data") or {}
        trend = bag.get("chart_dashboard_trend") or []
        self.assertTrue(trend)
        self.assertTrue(any((row.get("value") or 0) > 0 for row in trend))
        breakdown = bag.get("chart_dashboard_breakdown") or []
        self.assertTrue(breakdown)
        self.assertIn("machine", breakdown[0])
        table = bag.get("table_dashboard") or []
        self.assertTrue(table[0].get("id"))
        self.assertTrue(table[0].get("name"))
        bar = bag.get("series_analytics_1_bar") or []
        self.assertTrue(any((row.get("actual") or 0) > 0 for row in bar))


if __name__ == "__main__":
    unittest.main()
