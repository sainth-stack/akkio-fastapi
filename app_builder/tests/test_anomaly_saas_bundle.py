"""Anomaly SaaS bundle: PRD/UIUX/architecture → appConfig + mockData."""
import unittest

from app_builder.services.anomaly_bundle_from_plan import screen_name_to_page_key
from app_builder.services.anomaly_saas_frontend import build_data_bundle


class TestAnomalySaasBundle(unittest.TestCase):
    def test_manufacturing_profile_metrics(self):
        bundle = build_data_bundle(
            "CNC Monitor",
            {"primary": "#0B3D6F"},
            requirement="CNC lathe spindle anomaly detection for manufacturing",
            prd="",
            uiux="",
        )
        self.assertEqual(bundle["meta"]["metricProfile"], "manufacturing")
        names = [m["metric_name"] for m in bundle["metricsConfig"]]
        self.assertIn("Spindle Vibration", names)

    def test_uiux_screen_labels_navigation(self):
        uiux = """
        ## Screens
        - **Operations Dashboard** — live metrics
        - **Anomaly Feed** — triage queue
        """
        bundle = build_data_bundle(
            "PlantWatch",
            {},
            uiux=uiux,
            architecture={"screens": [{"name": "Alert Config", "description": "Thresholds"}]},
        )
        labels = [n["label"] for n in bundle["navigation"]]
        self.assertIn("Operations Dashboard", labels)
        self.assertIn("Anomaly Feed", labels)
        self.assertIn("Alert Config", labels)

    def test_design_tokens_merge_theme(self):
        bundle = build_data_bundle(
            "Test",
            {},
            design_tokens={"colors": {"primary": "#112233", "danger": "#FF0000"}},
        )
        self.assertEqual(bundle["theme"]["primary"], "#112233")
        self.assertEqual(bundle["theme"]["anomalyMarker"], "#FF0000")

    def test_screen_name_mapping(self):
        self.assertEqual(screen_name_to_page_key("Anomaly Feed"), "anomalies")
        self.assertEqual(screen_name_to_page_key("Dashboard"), "dashboard")


if __name__ == "__main__":
    unittest.main()
