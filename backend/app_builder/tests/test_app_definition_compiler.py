import unittest


class TestAppDefinitionCompiler(unittest.TestCase):
    def test_manufacturing_domain_three_screens_compiler_preset(self):
        from app_builder.services.app_definition_compiler import compile_app_definition

        app = compile_app_definition(
            requirement="generate manufacturing monitoring app with dashboard, analytics, reports",
            plan_json={"blueprint": {"selected_domain": "manufacturing"}},
        )
        nav_labels = [n.label for n in app.navigation]
        page_ids = [p.id for p in app.pages]

        self.assertEqual(app.meta.domain, "manufacturing")
        self.assertEqual(nav_labels, ["Dashboard", "Analytics", "Reports"])
        self.assertEqual(page_ids, ["dashboard", "analytics", "reports"])
        self.assertNotIn("materials", page_ids)
        self.assertIn("kpis_dashboard", app.mock)
        self.assertIn("series_output_defects", app.mock)

    def test_inventory_prompt_still_inventory(self):
        from app_builder.services.app_definition_compiler import compile_app_definition

        app = compile_app_definition(
            requirement="Warehouse inventory and purchase orders for SKUs",
            plan_json={"blueprint": {"selected_domain": "retail"}},
        )
        page_ids = [p.id for p in app.pages]
        self.assertIn("materials", page_ids)
        self.assertIn("orders", page_ids)


if __name__ == "__main__":
    unittest.main()
