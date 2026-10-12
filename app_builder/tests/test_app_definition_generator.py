import json
import unittest


class TestAppDefinitionGenerator(unittest.TestCase):
    def test_screens_from_requirement_phrase(self):
        from app_builder.services.app_definition_generator import extract_screens_from_plan

        screens = extract_screens_from_plan(
            requirement="generate manufacturing monitoring app with dashboard, analytics, reports",
        )
        titles = [s["title"].lower() for s in screens]
        self.assertIn("dashboard", titles)
        self.assertIn("analytics", titles)
        self.assertIn("reports", titles)

    def test_resolve_sync_builds_three_pages(self):
        from app_builder.services.app_definition_generator import resolve_app_definition_sync

        app = resolve_app_definition_sync(
            requirement="manufacturing monitoring with dashboard, analytics, reports",
            plan_json={"blueprint": {"selected_domain": "manufacturing"}},
        )
        page_ids = [p.id for p in app.pages]
        self.assertEqual(len(app.pages), 3)
        self.assertIn("dashboard", page_ids)
        self.assertIn("analytics", page_ids)
        self.assertIn("reports", page_ids)
        self.assertNotIn("materials", page_ids)
        self.assertTrue(app.mock)

    def test_generate_json_app_files_from_requirement(self):
        from app_builder.services.json_app_generator import generate_json_app_files

        files = generate_json_app_files(
            requirement="CRM with dashboard, contacts, and deals",
            plan_json={"blueprint": {"selected_domain": "retail"}},
        )
        self.assertIn("frontend/src/mock/data.json", files)
        data = json.loads(files["frontend/src/mock/data.json"])
        nav_labels = [n["label"].lower() for n in data["navigation"]]
        self.assertIn("dashboard", nav_labels)


if __name__ == "__main__":
    unittest.main()
