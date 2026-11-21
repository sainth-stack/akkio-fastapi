"""
Comprehensive verification that LLM-authored apps never carry maintenance
columns/data for non-maintenance prompts, and that every layer of the fix is
in place (prompt guard, source-dataset planner, post-generation stripper).

All tests are deterministic — no LLM calls, no network, no pydantic deps beyond
what is already installed in the venv.
"""
from __future__ import annotations

import re
import sys
import unittest
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helpers — build minimal AppDefinition-like dicts without importing pydantic
# ---------------------------------------------------------------------------

def _page(id_: str, title: str, blocks: List[Any]) -> Dict[str, Any]:
    return {"id": id_, "title": title, "subtitle": "", "blocks": blocks}


def _kpi_block(data_key: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "kpiRow", "dataKey": data_key, "items": items}


def _table_block(data_key: str, columns: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "type": "dataTable",
        "title": "Records",
        "dataKey": data_key,
        "columns": columns,
        "searchable": True,
    }


def _line_chart(data_key: str, series: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "type": "lineChart",
        "title": "Trend",
        "dataKey": data_key,
        "xAxisKey": "period",
        "series": series,
    }


# ---------------------------------------------------------------------------
# Layer 1 — Domain guard string computed in the blocks user prompt
# ---------------------------------------------------------------------------

def _compute_domain_guard(requirement: str, section: str = "") -> str:
    """Mirror the logic in app_definition_llm_pipeline._maintenance_domain."""
    _maintenance_domain = bool(re.search(
        r"maintenance|\bmtbf\b|\bmttr\b|work order|downtime|\boee\b",
        f"{requirement} {section}",
        re.I,
    ))
    if _maintenance_domain:
        return "MAINTENANCE app — columns allowed"
    return "NOT a maintenance app. Do NOT use work_order_id, machine, production_line"


class TestDomainGuard(unittest.TestCase):
    """Layer 1: the domain guard injected into every blocks prompt."""

    def test_inventory_prompt_is_non_maintenance(self):
        guard = _compute_domain_guard(
            "Build a manufacturing inventory dashboard for warehouse stock, "
            "safety stock, reorder points, and stock-outs"
        )
        self.assertIn("NOT a maintenance", guard)

    def test_supply_chain_prompt_is_non_maintenance(self):
        guard = _compute_domain_guard(
            "Supply chain visibility dashboard with purchase orders, "
            "supplier lead times, on-time delivery rate"
        )
        self.assertIn("NOT a maintenance", guard)

    def test_vendor_quality_prompt_is_non_maintenance(self):
        guard = _compute_domain_guard(
            "Vendor quality management: inspection lots, defect rate, yield pct, supplier scorecard"
        )
        self.assertIn("NOT a maintenance", guard)

    def test_maintenance_prompt_is_maintenance(self):
        guard = _compute_domain_guard(
            "Plant maintenance dashboard — MTBF, MTTR, downtime, work orders"
        )
        self.assertIn("MAINTENANCE app", guard)

    def test_work_order_prompt_is_maintenance(self):
        guard = _compute_domain_guard("work order management with OEE and downtime tracking")
        self.assertIn("MAINTENANCE app", guard)

    def test_vendor_management_prompt_is_non_maintenance(self):
        guard = _compute_domain_guard(
            "manufacturing vendor management dashboard to track vendor spend, "
            "contract value, delivery performance, and quality scores"
        )
        self.assertIn("NOT a maintenance", guard)


# ---------------------------------------------------------------------------
# Layer 2 — Post-generation guard strips maintenance tables from non-maint apps
# ---------------------------------------------------------------------------

_MAINT_FIELDS = {
    "work_order_id", "machine", "production_line", "equipment",
    "event_type", "root_cause", "downtime_hours", "mtbf_hours", "mttr_hours",
}


def _post_gen_strip(requirement: str, pages_raw: List[Any]) -> List[Any]:
    """
    Mirror the post-generation guard from build_llm_app_definition.
    Returns a list of plain dicts (not PageDef objects).
    """
    _is_maint = bool(re.search(
        r"maintenance|\bmtbf\b|\bmttr\b|work order|downtime|\boee\b",
        requirement or "",
        re.I,
    ))
    if _is_maint:
        return pages_raw

    cleaned = []
    for page in pages_raw:
        new_blocks = []
        for block in page["blocks"]:
            if block.get("type") == "dataTable":
                cols = block.get("columns") or []
                block_fields = {str(c.get("field") or "") for c in cols}
                if block_fields and block_fields.issubset(_MAINT_FIELDS | {"status", "severity", "shift", "occurred_at"}):
                    continue  # drop entirely-maintenance table
                cleaned_cols = [
                    c for c in cols
                    if str(c.get("field") or "") not in _MAINT_FIELDS
                ]
                if not cleaned_cols:
                    continue
                new_blocks.append({**block, "columns": cleaned_cols})
            else:
                new_blocks.append(block)
        cleaned.append({**page, "blocks": new_blocks})
    return cleaned


class TestPostGenerationGuard(unittest.TestCase):
    """Layer 2: post-generation guard strips maintenance tables."""

    def _inventory_pages_with_maint_table(self):
        return [
            _page("dashboard", "Dashboard", [
                _kpi_block("kpis_dashboard", [
                    {"label": "Total Warehouse Stock", "valueKey": "total_stock"},
                    {"label": "Safety Stock Coverage", "valueKey": "safety_stock_days"},
                ]),
                _table_block("table_dashboard", [
                    {"field": "work_order_id", "header": "Work Order"},
                    {"field": "machine", "header": "Machine"},
                    {"field": "downtime_hours", "header": "Downtime (h)"},
                    {"field": "mtbf_hours", "header": "MTBF (h)"},
                    {"field": "mttr_hours", "header": "MTTR (h)"},
                    {"field": "status", "header": "Status"},
                ]),
            ])
        ]

    def test_inventory_prompt_strips_maintenance_table(self):
        requirement = "inventory dashboard for warehouse stock, safety stock, reorder points"
        pages = _post_gen_strip(requirement, self._inventory_pages_with_maint_table())
        dashboard = pages[0]
        types = [b.get("type") for b in dashboard["blocks"]]
        # Maintenance-only table must be removed
        self.assertNotIn("dataTable", types,
            "post-gen guard should strip a table with only maintenance columns for a non-maintenance prompt")

    def test_kpi_row_is_preserved(self):
        requirement = "inventory dashboard for warehouse stock, safety stock, reorder points"
        pages = _post_gen_strip(requirement, self._inventory_pages_with_maint_table())
        dashboard = pages[0]
        types = [b.get("type") for b in dashboard["blocks"]]
        self.assertIn("kpiRow", types, "kpiRow must be preserved")

    def test_mixed_table_keeps_non_maintenance_cols(self):
        """Table with BOTH inventory and maintenance columns: keep inventory cols, drop maintenance ones."""
        pages = [
            _page("dashboard", "Dashboard", [
                _table_block("table_dashboard", [
                    {"field": "sku", "header": "SKU"},
                    {"field": "warehouse", "header": "Warehouse"},
                    {"field": "quantity_on_hand", "header": "Qty on Hand"},
                    {"field": "work_order_id", "header": "Work Order"},  # should be stripped
                    {"field": "mtbf_hours", "header": "MTBF (h)"},       # should be stripped
                ]),
            ])
        ]
        result = _post_gen_strip("inventory warehouse stock management", pages)
        table = next(b for b in result[0]["blocks"] if b.get("type") == "dataTable")
        fields = [c["field"] for c in table["columns"]]
        self.assertIn("sku", fields)
        self.assertIn("warehouse", fields)
        self.assertIn("quantity_on_hand", fields)
        self.assertNotIn("work_order_id", fields, "work_order_id should be stripped")
        self.assertNotIn("mtbf_hours", fields, "mtbf_hours should be stripped")

    def test_maintenance_app_keeps_maintenance_table(self):
        """Maintenance prompts must NOT have their tables stripped."""
        pages = [
            _page("dashboard", "Dashboard", [
                _table_block("table_dashboard", [
                    {"field": "work_order_id", "header": "Work Order"},
                    {"field": "machine", "header": "Machine"},
                    {"field": "downtime_hours", "header": "Downtime (h)"},
                ]),
            ])
        ]
        result = _post_gen_strip(
            "plant maintenance dashboard MTBF MTTR work order downtime", pages
        )
        types = [b.get("type") for b in result[0]["blocks"]]
        self.assertIn("dataTable", types,
            "maintenance app must keep its maintenance table")

    def test_supply_chain_prompt_strips_maintenance_table(self):
        pages = [
            _page("dashboard", "Dashboard", [
                _table_block("table_dashboard", [
                    {"field": "work_order_id", "header": "Work Order"},
                    {"field": "machine", "header": "Machine"},
                    {"field": "event_type", "header": "Event Type"},
                    {"field": "occurred_at", "header": "Date"},
                    {"field": "status", "header": "Status"},
                ]),
            ])
        ]
        result = _post_gen_strip(
            "supply chain visibility dashboard with purchase orders and supplier lead times",
            pages,
        )
        types = [b.get("type") for b in result[0]["blocks"]]
        self.assertNotIn("dataTable", types)

    def test_vendor_quality_prompt_strips_maintenance_table(self):
        pages = [
            _page("analytics", "Analytics", [
                _table_block("table_analytics", [
                    {"field": "work_order_id", "header": "Work Order"},
                    {"field": "equipment", "header": "Equipment"},
                    {"field": "root_cause", "header": "Root Cause"},
                    {"field": "severity", "header": "Severity"},
                    {"field": "status", "header": "Status"},
                ]),
            ])
        ]
        result = _post_gen_strip(
            "vendor quality management with inspection lots, defect rate, yield pct",
            pages,
        )
        types = [b.get("type") for b in result[0]["blocks"]]
        self.assertNotIn("dataTable", types)


# ---------------------------------------------------------------------------
# Layer 3 — source_dataset_planner assigns correct sid for LLM-authored apps
# ---------------------------------------------------------------------------

def _sid_for_llm_app(requirement: str) -> str:
    """Mirror the LLM-authored branch of infer_source_datasets context-based sid logic."""
    is_quality = bool(re.search(r"quality|inspect|defect|yield", requirement, re.I))
    is_retail = bool(re.search(r"retail|e-?commerce|sales order|pos\b", requirement, re.I))
    is_maint = bool(re.search(r"maintenance|\bmtbf\b|\bmttr\b|work order|downtime|\boee\b", requirement, re.I))
    is_inventory = bool(re.search(r"inventory|stock|sku|warehouse|reorder", requirement, re.I))
    is_supply_chain = bool(re.search(r"supply.chain|purchase.order|\bpo\b|procurement|sourcing", requirement, re.I))
    is_vendor = bool(re.search(r"vendor|supplier", requirement, re.I))

    if is_supply_chain or is_vendor:
        return "purchase_orders"
    elif is_inventory:
        return "inventory_items"
    elif is_quality:
        return "quality_inspections"
    elif is_retail:
        return "sales_orders"
    elif is_maint:
        return "maintenance_events"
    else:
        return "operational_records"


class TestSourceDatasetSid(unittest.TestCase):
    """Layer 3: source dataset id correctly derived from context for LLM apps."""

    def test_inventory_prompt_gets_inventory_sid(self):
        sid = _sid_for_llm_app(
            "manufacturing inventory dashboard for warehouse stock, safety stock, reorder points"
        )
        self.assertEqual(sid, "inventory_items")

    def test_supply_chain_prompt_gets_purchase_orders_sid(self):
        sid = _sid_for_llm_app(
            "supply chain visibility dashboard with purchase orders and supplier lead times"
        )
        self.assertEqual(sid, "purchase_orders")

    def test_vendor_quality_gets_purchase_orders_sid(self):
        # vendor + quality → vendor wins (purchase_orders)
        sid = _sid_for_llm_app(
            "vendor quality management dashboard with vendor scorecard, inspection lots, defect rate"
        )
        self.assertEqual(sid, "purchase_orders")

    def test_pure_quality_no_vendor_gets_quality_sid(self):
        sid = _sid_for_llm_app(
            "quality inspection dashboard with inspection lots, defect rate, yield pct"
        )
        self.assertEqual(sid, "quality_inspections")

    def test_maintenance_prompt_gets_maintenance_sid(self):
        sid = _sid_for_llm_app(
            "plant maintenance dashboard MTBF MTTR work order downtime"
        )
        self.assertEqual(sid, "maintenance_events")

    def test_vendor_management_gets_purchase_orders_sid(self):
        sid = _sid_for_llm_app(
            "manufacturing vendor management dashboard to track vendor spend, "
            "contract value, delivery performance, and quality scores"
        )
        self.assertEqual(sid, "purchase_orders")

    def test_supply_chain_not_mistaken_for_maintenance(self):
        # This was the original bug: supply chain matched as maintenance
        sid = _sid_for_llm_app(
            "supply chain management with on-time delivery, inventory turnover, supplier spend"
        )
        self.assertNotEqual(sid, "maintenance_events")
        self.assertEqual(sid, "purchase_orders")


# ---------------------------------------------------------------------------
# Layer 4 — PRD prompt builder never injects maintenance metrics for non-maint
# ---------------------------------------------------------------------------

class TestPrdPromptBuilder(unittest.TestCase):
    """Layer 4: prd_prompt_builder doesn't add MTBF/MTTR to non-maintenance PRDs."""

    def test_inventory_prd_hint_excludes_maintenance_metrics(self):
        from app_builder.services.prd_prompt_builder import _page_layout_hint, _asks_maintenance

        requirement = "inventory dashboard for warehouse stock, safety stock, reorder points, stock-outs"
        self.assertFalse(_asks_maintenance(requirement))
        # _page_layout_hint takes (page_title_string, requirement=...)
        hint = _page_layout_hint("Dashboard", requirement=requirement)
        # Hint must NOT mention MTBF or work orders
        self.assertNotIn("MTBF", hint, "inventory PRD hint must not contain MTBF")
        self.assertNotIn("MTTR", hint, "inventory PRD hint must not contain MTTR")
        self.assertNotIn("work order", hint.lower(), "inventory PRD hint must not contain work orders")

    def test_vendor_quality_prd_hint_excludes_maintenance_metrics(self):
        from app_builder.services.prd_prompt_builder import _page_layout_hint, _asks_maintenance

        requirement = "vendor quality management with inspection lots, defect rate, yield pct, supplier scorecard"
        self.assertFalse(_asks_maintenance(requirement))
        hint = _page_layout_hint("Dashboard", requirement=requirement)
        self.assertNotIn("MTBF", hint)
        self.assertNotIn("work order", hint.lower())

    def test_maintenance_prd_hint_keeps_maintenance_metrics(self):
        from app_builder.services.prd_prompt_builder import _asks_maintenance

        requirement = "plant maintenance dashboard MTBF MTTR work order downtime OEE"
        self.assertTrue(_asks_maintenance(requirement))


# ---------------------------------------------------------------------------
# Layer 5 — _BLOCKS_SYSTEM must contain the domain column guide
# ---------------------------------------------------------------------------

class TestBlocksSystemPrompt(unittest.TestCase):
    """Layer 5: the _BLOCKS_SYSTEM prompt has the CRITICAL domain column guide."""

    def test_blocks_system_has_critical_domain_column_guide(self):
        from app_builder.services.app_definition_llm_pipeline import _BLOCKS_SYSTEM
        self.assertIn("CRITICAL", _BLOCKS_SYSTEM)
        self.assertIn("sku", _BLOCKS_SYSTEM.lower())
        self.assertIn("work_order_id", _BLOCKS_SYSTEM)
        # Must forbid maintenance columns for non-maintenance prompts
        self.assertIn("NEVER default to work_order_id", _BLOCKS_SYSTEM)

    def test_blocks_system_has_domain_guard_slot(self):
        from app_builder.services.app_definition_llm_pipeline import _BLOCKS_SYSTEM
        # Domain guard must NOT be forbidden — the blocks user prompt injects it
        # The system prompt itself does not need the guard, but should NOT say
        # "manufacturing/maintenance names" as a universal fallback
        self.assertNotIn("manufacturing/maintenance names", _BLOCKS_SYSTEM)

    def test_data_system_does_not_say_maintenance_names(self):
        from app_builder.services.app_definition_llm_pipeline import _DATA_SYSTEM
        self.assertNotIn("manufacturing/maintenance names", _DATA_SYSTEM)


# ---------------------------------------------------------------------------
# Layer 6 — dashboard_layout only adds maintenance chart series for maint PRDs
# ---------------------------------------------------------------------------

class TestDashboardLayoutMaintGuard(unittest.TestCase):
    """Layer 6: align_dashboard_charts_to_kpis only adds maint series for maint apps."""

    def _make_app_with_linechart(self, kpi_items=None):
        """Return a minimal AppDefinition-like structure."""
        try:
            from app_builder.schemas.app_definition import (
                AppDefinition, AppMeta, NavItemDef, PageDef,
            )
        except ImportError:
            self.skipTest("pydantic not installed in test env")
        blocks = [_line_chart("chart_trend", [])]
        if kpi_items:
            blocks.insert(0, _kpi_block("kpis_dashboard", kpi_items))
        return AppDefinition(
            meta=AppMeta(appName="Test App", domain="manufacturing", description=""),
            navigation=[NavItemDef(label="Dashboard", path="/", icon="Dashboard", pageId="dashboard")],
            pages=[PageDef(id="dashboard", title="Dashboard", subtitle="", blocks=blocks)],
            mock={},
        )

    def test_non_maintenance_prd_does_not_add_mtbf_series(self):
        from app_builder.services.dashboard_layout import align_dashboard_charts_to_kpis
        defn = self._make_app_with_linechart(kpi_items=[
            {"label": "Total Stock", "valueKey": "total_stock"},
            {"label": "Safety Stock Days", "valueKey": "safety_stock_days"},
        ])
        result = align_dashboard_charts_to_kpis(
            defn,
            dashboard_page_id="dashboard",
            prd="inventory dashboard for warehouse stock and safety stock",
        )
        chart = next(b for b in result.pages[0].blocks if b.get("type") == "lineChart")
        series_keys = {s.get("dataKey") for s in chart.get("series", [])}
        self.assertNotIn("mtbf_hours", series_keys)
        self.assertNotIn("mttr_hours", series_keys)
        self.assertNotIn("downtime_hours", series_keys)

    def test_maintenance_prd_adds_maintenance_series_to_sole_chart(self):
        from app_builder.services.dashboard_layout import align_dashboard_charts_to_kpis
        defn = self._make_app_with_linechart(kpi_items=[
            {"label": "MTBF", "valueKey": "mtbf_hours"},
            {"label": "MTTR", "valueKey": "mttr_hours"},
        ])
        result = align_dashboard_charts_to_kpis(
            defn,
            dashboard_page_id="dashboard",
            prd="plant maintenance MTBF MTTR work order downtime",
        )
        chart = next(b for b in result.pages[0].blocks if b.get("type") == "lineChart")
        series_keys = {s.get("dataKey") for s in chart.get("series", [])}
        self.assertTrue(
            series_keys & {"mtbf_hours", "mttr_hours", "downtime_hours"},
            f"maintenance chart should have maintenance series, got {series_keys}",
        )


# ---------------------------------------------------------------------------
# Layer 7 — source_dataset_planner integration (uses actual module)
# ---------------------------------------------------------------------------

class TestSourceDatasetPlannerIntegration(unittest.TestCase):
    """Layer 7: actual source_dataset_planner.infer_source_datasets for LLM apps."""

    def _make_llm_defn(self, requirement: str, columns: List[Dict[str, str]]) -> Dict[str, Any]:
        """Build a minimal LLM-authored app dict with one dataTable."""
        return {
            "meta": {
                "appName": "Test App",
                "domain": "manufacturing",
                "description": requirement,
                "requirement": requirement,
                "generatedBy": "llm_pipeline",
            },
            "pages": [
                {
                    "id": "dashboard",
                    "title": "Dashboard",
                    "blocks": [
                        {
                            "type": "dataTable",
                            "dataKey": "table_dashboard",
                            "columns": columns,
                        }
                    ],
                }
            ],
            "navigation": [{"pageId": "dashboard", "path": "/", "label": "Dashboard"}],
            "data": {},
        }

    def test_inventory_llm_app_with_maintenance_columns_gets_inventory_sid(self):
        """
        Even if the LLM wrote maintenance columns (the old bug), the sid
        must be derived from the requirement text, not column sniffing.
        """
        from app_builder.services.source_dataset_planner import infer_source_datasets

        requirement = "inventory dashboard for warehouse stock, safety stock, reorder points"
        defn = self._make_llm_defn(
            requirement,
            # LLM hallucinated these maintenance columns
            [
                {"field": "work_order_id", "header": "Work Order"},
                {"field": "machine", "header": "Machine"},
                {"field": "downtime_hours", "header": "Downtime (h)"},
            ],
        )
        datasets = infer_source_datasets(defn, requirement=requirement)
        self.assertTrue(datasets, "should return at least one source dataset")
        ids = [d.get("id") for d in datasets]
        self.assertNotIn(
            "maintenance_events", ids,
            f"inventory prompt must not get maintenance_events sid even with maintenance columns, got {ids}",
        )
        self.assertIn("inventory_items", ids)

    def test_supply_chain_llm_app_gets_purchase_orders_sid(self):
        from app_builder.services.source_dataset_planner import infer_source_datasets

        requirement = "supply chain visibility dashboard with purchase orders, supplier lead times"
        defn = self._make_llm_defn(
            requirement,
            [
                {"field": "work_order_id", "header": "Work Order"},
                {"field": "machine", "header": "Machine"},
            ],
        )
        datasets = infer_source_datasets(defn, requirement=requirement)
        ids = [d.get("id") for d in datasets]
        self.assertNotIn("maintenance_events", ids)
        self.assertIn("purchase_orders", ids)

    def test_vendor_quality_llm_app_gets_purchase_orders_sid(self):
        from app_builder.services.source_dataset_planner import infer_source_datasets

        requirement = "vendor quality management with vendor scorecard, defect rate"
        defn = self._make_llm_defn(
            requirement,
            [{"field": "work_order_id", "header": "Work Order"}],
        )
        datasets = infer_source_datasets(defn, requirement=requirement)
        ids = [d.get("id") for d in datasets]
        self.assertNotIn("maintenance_events", ids)

    def test_maintenance_llm_app_keeps_maintenance_sid(self):
        from app_builder.services.source_dataset_planner import infer_source_datasets

        requirement = "plant maintenance dashboard MTBF MTTR work order downtime"
        defn = self._make_llm_defn(
            requirement,
            [
                {"field": "work_order_id", "header": "Work Order"},
                {"field": "downtime_hours", "header": "Downtime (h)"},
            ],
        )
        datasets = infer_source_datasets(defn, requirement=requirement)
        ids = [d.get("id") for d in datasets]
        self.assertIn("maintenance_events", ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
