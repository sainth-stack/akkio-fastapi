import unittest


class TestPrdPromptBuilder(unittest.TestCase):
    def test_manufacturing_prompt_pages_and_product(self):
        from app_builder.services.prd_prompt_builder import (
            build_prd_prompts,
            infer_pages_from_requirement,
            is_multi_page_dashboard_prompt,
        )

        req = """
Create a Manufacturing Quality Monitoring Dashboard with 4 pages:

1. **Dashboard:** Quality KPIs like Total Inspected, First Pass Yield, Defect Rate
2. **Sessions:** Inspection sessions table
3. **Reports:** Export quality reports
4. **Analytics:** Trends and pareto analysis
"""
        self.assertTrue(is_multi_page_dashboard_prompt(req))
        pages = infer_pages_from_requirement(req)
        self.assertEqual(pages, ["Dashboard", "Sessions", "Reports", "Analytics"])

        system, user = build_prd_prompts(req)
        self.assertIn("Application Pages", system)
        self.assertIn("### Dashboard", system)
        self.assertIn("Manufacturing", user)
        self.assertNotIn("1. Product Overview", system)
        self.assertIn("First Pass Yield", req)


if __name__ == "__main__":
    unittest.main()
