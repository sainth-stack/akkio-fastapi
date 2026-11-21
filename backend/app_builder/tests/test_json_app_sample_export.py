import json
import os
import tempfile
import unittest


class TestJsonAppSampleExport(unittest.TestCase):
    def test_build_sheets_from_mock_file(self):
        from app_builder.services.json_app_sample_export import build_sheets_from_json_app

        from app_builder.services.runtime_paths import project_write_root

        pname = "test_json_export_proj"
        root = project_write_root(pname)
        os.makedirs(os.path.join(root, "frontend", "src", "mock"), exist_ok=True)
        mock = {
            "kpis_dashboard": {"oee": "87%", "throughput": "120 u/h"},
            "analytics_data": [{"id": f"R{i}", "name": f"M{i}", "status": "OK"} for i in range(20)],
        }
        with open(os.path.join(root, "frontend", "src", "mock", "data.json"), "w") as f:
            json.dump({"meta": {"appName": "Test"}, "mock": mock, "data": mock}, f)

        try:
            sheets = build_sheets_from_json_app(pname)
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)

            self.assertIsNotNone(sheets)
            self.assertIn("KPIs", sheets)
            self.assertGreaterEqual(len(sheets["Detail"]), 18)


if __name__ == "__main__":
    unittest.main()
