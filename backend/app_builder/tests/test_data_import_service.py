import json
import unittest


class TestDataImportService(unittest.TestCase):
    def _mini_defn(self):
        return {
            "pages": [
                {
                    "id": "dashboard",
                    "title": "Dashboard",
                    "blocks": [
                        {
                            "type": "kpiRow",
                            "dataKey": "kpis_dashboard",
                            "items": [
                                {"label": "MTBF", "valueKey": "mtbf"},
                                {"label": "MTTR", "valueKey": "mttr"},
                            ],
                        },
                        {
                            "type": "dataTable",
                            "dataKey": "table_dashboard",
                            "columns": [
                                {"field": "work_order_id", "header": "Work order"},
                                {"field": "status", "header": "Status"},
                            ],
                        },
                    ],
                }
            ],
            "navigation": [{"pageId": "dashboard", "path": "/", "label": "Dashboard"}],
            "data": {
                "kpis_dashboard": {"mtbf": "100 h", "mttr": "2 h"},
                "table_dashboard": [{"work_order_id": "WO-1", "status": "Open"}],
            },
        }

    def test_csv_kpi_upload(self):
        from app_builder.services.data_import_service import merge_upload_bytes

        csv_body = b"label,value\nMTBF,320 h\nMTTR,1.5 h\n"
        updated, summary = merge_upload_bytes(
            self._mini_defn(),
            filename="kpis_dashboard.csv",
            raw=csv_body,
        )
        bag = updated.get("data") or {}
        self.assertIn("kpis_dashboard", summary["updatedKeys"])
        self.assertEqual(bag["kpis_dashboard"]["mtbf"], "320 h")

    def test_json_partial_merge(self):
        from app_builder.services.data_import_service import merge_upload_bytes

        payload = json.dumps({"table_dashboard": [{"work_order_id": "WO-9", "status": "Closed"}]}).encode()
        updated, summary = merge_upload_bytes(
            self._mini_defn(),
            filename="patch.json",
            raw=payload,
        )
        rows = (updated.get("data") or {})["table_dashboard"]
        self.assertEqual(rows[0]["work_order_id"], "WO-9")
        self.assertIn("table_dashboard", summary["updatedKeys"])


if __name__ == "__main__":
    unittest.main()
