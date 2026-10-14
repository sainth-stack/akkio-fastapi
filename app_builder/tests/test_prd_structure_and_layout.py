import unittest


class TestPrdStructureAndLayout(unittest.TestCase):
    def test_resolve_app_name_not_dashboard_colon(self):
        from app_builder.services.app_definition_generator import resolve_app_display_name

        req = """
Create a Manufacturing Quality Monitoring Dashboard with 4 pages:

1. **Dashboard:** Quality KPIs
"""
        name = resolve_app_display_name(requirement=req)
        self.assertNotEqual(name, "Dashboard:")
        self.assertIn("Manufacturing", name)

    def test_numbered_prompt_pages(self):
        from app_builder.services.app_definition_generator import extract_screens_from_plan

        req = """
Create a Manufacturing Quality Monitoring Dashboard with 4 pages:

1. **Dashboard:** Quality KPIs
2. **Sessions:** Inspection sessions
3. **Reports:** Export reports
4. **Analytics:** Trends
"""
        screens = extract_screens_from_plan(requirement=req)
        titles = [s["title"] for s in screens]
        self.assertEqual(len(screens), 4)
        self.assertEqual(titles, ["Dashboard", "Sessions", "Reports", "Analytics"])

    def test_ensure_minimum_blocks_for_sparse_page(self):
        from app_builder.schemas.app_definition_llm import PageBlocksPlan
        from app_builder.services.prd_driven_layout import ensure_minimum_blocks

        blocks = [{"type": "dataTable", "dataKey": "table_analytics", "columns": []}]
        fixed = ensure_minimum_blocks(blocks, "analytics", "Analytics")
        self.assertGreaterEqual(len(fixed), 2)
        PageBlocksPlan(blocks=fixed)

    def test_section_for_numbered_dashboard_heading(self):
        from app_builder.services.prd_structure_parser import section_for_page

        prd = """
### 1. Dashboard
- KPI: Total Units Inspected
- Line chart: Quality trends over time.

### 2. Sessions
- Table of inspection sessions
- Columns:
  - Session ID
  - Operator

### 3. Analytics
- Bar chart: Defects by category.
"""
        dash = section_for_page(prd, "Dashboard", "dashboard")
        self.assertIn("Total Units Inspected", dash)
        self.assertNotIn("Session ID", dash)

        sessions = section_for_page(prd, "Sessions", "sessions")
        self.assertIn("Session ID", sessions)

    def test_blocks_do_not_dump_entire_prd(self):
        from app_builder.services.prd_driven_layout import blocks_for_page

        huge = "## Overview\n" + "\n".join(
            [f"- Line chart: Trend {i} over time." for i in range(20)]
        )
        blocks = blocks_for_page("dashboard", "Dashboard", huge)
        charts = [b for b in blocks if b.get("type") in ("lineChart", "barChart", "pieChart")]
        self.assertLessEqual(len(charts), 6)

    def test_finalize_removes_sample_table_cells(self):
        from app_builder.schemas.app_definition import AppDefinition, AppMeta, AppTheme, NavItemDef, PageDef
        from app_builder.services.app_definition_generator import finalize_app_definition

        defn = AppDefinition(
            meta=AppMeta(
                appName="Manufacturing Quality Monitoring Dashboard",
                description="test",
                domain="manufacturing_quality",
                generatedBy="llm_pipeline",
            ),
            theme=AppTheme(),
            navigation=[NavItemDef(label="Sessions", path="/sessions", icon="Dashboard", pageId="sessions")],
            pages=[
                PageDef(
                    id="sessions",
                    title="Sessions",
                    subtitle="",
                    blocks=[
                        {
                            "type": "dataTable",
                            "dataKey": "table_sessions",
                            "columns": [
                                {"field": "session_id", "header": "Session ID"},
                                {"field": "operator", "header": "Operator"},
                            ],
                        }
                    ],
                )
            ],
            mock={
                "table_sessions": [
                    {"session_id": "Sample session_id 1", "operator": "Sample operator 1"},
                ]
            },
        )
        out = finalize_app_definition(defn)
        row = (out.mock or {}).get("table_sessions", [{}])[0]
        self.assertFalse(str(row.get("session_id", "")).startswith("Sample "))
        self.assertEqual(out.navigation[0].icon, "FactCheck")


if __name__ == "__main__":
    unittest.main()
