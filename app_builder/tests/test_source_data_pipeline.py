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
        from app_builder.services.operational_metrics_compute import apply_source_data_pipeline

        defn = apply_source_data_pipeline(self._manufacturing_defn())
        sheets = build_template_sheets(defn)
        self.assertTrue(sheets)
        first = next(iter(sheets.values()))
        self.assertIn("Work order", first[0].keys())
        self.assertNotIn("label", first[0])

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


if __name__ == "__main__":
    unittest.main()
