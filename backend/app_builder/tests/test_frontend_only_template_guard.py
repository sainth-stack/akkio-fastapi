import unittest

from app_builder.services.frontend_only_template_guard import (
    fix_ui_barrel_index,
    restore_template_ui_kit,
)


class TestFrontendOnlyTemplateGuard(unittest.TestCase):
    def test_fix_ui_barrel_chart_paths(self):
        bad = "export { default as LineChart } from './LineChart';"
        fixed = fix_ui_barrel_index(bad)
        self.assertIn("./charts/LineChart", fixed)

    def test_restore_overwrites_broken_barrel(self):
        files = {
            "frontend/src/components/ui/index.ts": "export { default as LineChart } from './LineChart';",
            "frontend/src/pages/DashboardPage.tsx": "// page",
        }
        merged, count = restore_template_ui_kit(files)
        self.assertGreater(count, 0)
        self.assertIn("./charts/LineChart", merged["frontend/src/components/ui/index.ts"])
        self.assertEqual(merged["frontend/src/pages/DashboardPage.tsx"], "// page")


if __name__ == "__main__":
    unittest.main()
