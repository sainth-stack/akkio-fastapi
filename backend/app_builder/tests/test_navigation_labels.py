import unittest

from app_builder.services.navigation_labels import normalize_definition_navigation


class TestNavigationLabels(unittest.TestCase):
    def test_no_duplicate_dashboard_from_kpi_subtitle(self):
        defn = {
            "pages": [
                {"id": "dashboard", "title": "Kpi cards"},
                {"id": "analytics", "title": "Charts only (no full data table on dashboard)"},
                {"id": "reports", "title": "Reports"},
            ],
            "navigation": [
                {"label": "Kpi cards", "path": "/", "icon": "Dashboard", "pageId": "dashboard"},
                {"label": "Charts only", "path": "/analytics", "icon": "Article", "pageId": "analytics"},
                {"label": "Reports", "path": "/reports", "icon": "Assessment", "pageId": "reports"},
            ],
        }
        out = normalize_definition_navigation(defn)
        labels = [n["label"] for n in out["navigation"]]
        self.assertEqual(labels.count("Dashboard"), 1)
        self.assertIn("Analytics", labels)
        self.assertNotIn(labels[1], ("Dashboard",))

    def test_dedupes_same_page_id(self):
        defn = {
            "pages": [{"id": "dashboard", "title": "Dashboard"}],
            "navigation": [
                {"label": "A", "path": "/", "pageId": "dashboard"},
                {"label": "B", "path": "/", "pageId": "dashboard"},
            ],
        }
        out = normalize_definition_navigation(defn)
        self.assertEqual(len(out["navigation"]), 1)


if __name__ == "__main__":
    unittest.main()
