import unittest


class TestPageLayoutIntent(unittest.TestCase):
    def test_dashboard_overview_no_table(self):
        from app_builder.services.page_layout_intent import infer_page_layout_intent
        from app_builder.services.prd_driven_layout import blocks_for_page

        section = """### Dashboard
**Purpose:** Real-time overview and executive monitoring of quality KPIs

**KPIs:**
- Total Units Inspected
- First Pass Yield (FPY)
- Defect Rate
- Scrap Rate

**Charts:**
- Line chart: Quality trends over time
- Bar chart: Defects by category
- Donut chart: Passed vs reworked vs scrapped

**Filters:**
- Date range
- Production line
"""
        intent = infer_page_layout_intent(section, "Dashboard", page_id="dashboard")
        self.assertTrue(intent.wants_kpis)
        self.assertTrue(intent.wants_charts)
        self.assertGreaterEqual(intent.chart_cap, 2)
        self.assertFalse(intent.wants_table)
        self.assertFalse(intent.wants_filters)

        blocks = blocks_for_page("dashboard", "Dashboard", section, dashboard_page_id="dashboard")
        types = [b["type"] for b in blocks]
        self.assertIn("kpiRow", types)
        self.assertNotIn("dataTable", types)
        self.assertNotIn("filterBar", types)
        self.assertIn("lineChart", types)
        self.assertIn("barChart", types)

    def test_dashboard_kpis_only_when_no_chart_bullets(self):
        from app_builder.services.page_layout_intent import infer_page_layout_intent
        from app_builder.services.prd_driven_layout import blocks_for_page

        section = """### Dashboard
**Purpose:** Maintenance KPI overview

**KPIs:**
- Total Maintenance Work Orders
- MTBF
- MTTR
"""
        intent = infer_page_layout_intent(section, "Dashboard", page_id="dashboard")
        self.assertTrue(intent.wants_kpis)
        self.assertFalse(intent.wants_charts)
        blocks = blocks_for_page("dashboard", "Dashboard", section, dashboard_page_id="dashboard")
        types = [b["type"] for b in blocks]
        self.assertIn("kpiRow", types)
        self.assertNotIn("lineChart", types)

    def test_dashboard_include_section_charts(self):
        from app_builder.services.page_layout_intent import infer_page_layout_intent
        from app_builder.services.prd_driven_layout import blocks_for_page

        section = """### Dashboard
**Purpose:** Maintenance KPI overview

**KPIs:**
- Total Maintenance Work Orders
- Mean Time Between Failures (MTBF)
- Mean Time to Repair (MTTR)

**Include:**
- Line chart: Production output and operational downtime trends over time.
- Bar chart: Production performance by category.
"""
        intent = infer_page_layout_intent(section, "Dashboard", page_id="dashboard")
        self.assertTrue(intent.wants_kpis)
        self.assertTrue(intent.wants_charts)
        self.assertEqual(intent.chart_cap, 2)
        blocks = blocks_for_page("dashboard", "Dashboard", section, dashboard_page_id="dashboard")
        types = [b["type"] for b in blocks]
        self.assertIn("kpiRow", types)
        self.assertIn("lineChart", types)
        self.assertIn("barChart", types)

    def test_sessions_list_table_focus(self):
        from app_builder.services.prd_driven_layout import blocks_for_page

        section = """### Sessions
**Purpose:** Browse and search inspection sessions in a structured list

**Table columns:**
- Session ID
- Date & Time
- Operator
- Status

**Filters:**
- Search
- Date range
"""
        blocks = blocks_for_page("sessions", "Sessions", section, dashboard_page_id="dashboard")
        types = [b["type"] for b in blocks]
        self.assertIn("dataTable", types)
        self.assertNotIn("kpiRow", types)

    def test_custom_page_name_widgets_from_prd(self):
        from app_builder.services.prd_driven_layout import blocks_for_page

        section = """### Supplier Scorecard
**Purpose:** Compare vendor quality and delivery performance

**Charts:**
- Line chart: On-time delivery % vs target over time
- Bar chart: Defect rate by supplier

**Filters:**
- Date range
- Supplier tier
"""
        blocks = blocks_for_page("supplier_scorecard", "Supplier Scorecard", section)
        types = [b["type"] for b in blocks]
        self.assertNotIn("kpiRow", types)
        self.assertIn("filterBar", types)
        self.assertGreaterEqual(sum(1 for t in types if t in ("lineChart", "barChart")), 2)
        line = next(b for b in blocks if b["type"] == "lineChart")
        self.assertIn("target", [s["dataKey"] for s in line.get("series", [])])

    def test_analytics_charts_no_kpis(self):
        from app_builder.services.page_layout_intent import infer_page_layout_intent
        from app_builder.services.prd_driven_layout import blocks_for_page

        section = """### Analytics
**Purpose:** Deep-dive trends, pareto analysis, and yield breakdowns

**Charts:**
- Line chart: Yield trend by week
- Bar chart: Defect pareto by category
- Donut chart: Scrap vs rework mix
"""
        intent = infer_page_layout_intent(section, "Analytics")
        self.assertFalse(intent.wants_kpis)
        self.assertTrue(intent.wants_charts)
        blocks = blocks_for_page("analytics", "Analytics", section, dashboard_page_id="dashboard")
        types = [b["type"] for b in blocks]
        self.assertNotIn("kpiRow", types)
        self.assertGreaterEqual(types.count("lineChart") + types.count("barChart") + types.count("pieChart"), 1)

    def test_reports_export_one_chart_table(self):
        from app_builder.services.page_layout_intent import infer_page_layout_intent

        section = """### Reports
**Purpose:** Generate and export quality reports for stakeholders

**Charts:**
- Bar chart: Defect summary by line

**Table columns:**
- Report name
- Date range
- Status

**Actions:**
- Export PDF
- Export CSV
"""
        intent = infer_page_layout_intent(section, "Reports", page_id="reports")
        self.assertTrue(intent.wants_export_actions)
        self.assertFalse(intent.wants_table)
        self.assertFalse(intent.wants_kpis)
        self.assertFalse(intent.wants_filters)
        self.assertGreaterEqual(intent.chart_cap, 1)


    def test_requirement_pages_list_sidebar_names(self):
        from app_builder.services.app_definition_generator import extract_screens_from_plan

        req = (
            "Create a Manufacturing Maintenance Monitoring Dashboard with 3 pages: "
            "Dashboard, Reports, and Analytics."
        )
        screens = extract_screens_from_plan(requirement=req, prd="", uiux="", plan_json={})
        titles = [s["title"] for s in screens]
        self.assertEqual(titles[:3], ["Dashboard", "Reports", "Analytics"])


if __name__ == "__main__":
    unittest.main()
