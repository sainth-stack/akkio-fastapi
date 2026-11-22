"""Dashboard vs Analytics layout: table on dashboard, charts aligned to KPIs."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app_builder.schemas.app_definition import AppDefinition


def _kpi_items_from_page(page: Any) -> List[Dict[str, Any]]:
    for block in page.blocks or []:
        if isinstance(block, dict) and block.get("type") == "kpiRow":
            items = block.get("items") or []
            return [i for i in items if isinstance(i, dict)]
    return []


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (label or "").lower()).strip("_")


def align_dashboard_charts_to_kpis(
    defn: AppDefinition,
    *,
    dashboard_page_id: str,
    prd: str = "",
) -> AppDefinition:
    """Point dashboard chart series at KPI labels that are actually on the page."""
    _maint_prd = bool(re.search(r"maintenance|\bmtbf\b|\bmttr\b|work order|downtime", prd or "", re.I))
    for page in defn.pages:
        if page.id != dashboard_page_id:
            continue
        kpis = _kpi_items_from_page(page)
        chart_keys = {
            "mtbf": ("mtbf_hours", "MTBF (h)"),
            "mttr": ("mttr_hours", "MTTR (h)"),
            "downtime": ("downtime_hours", "Downtime (h)"),
            "work_order": ("work_orders", "Work orders"),
            "output": ("production_output", "Production output"),
        }
        present_hints: Dict[str, str] = {}
        # Seed hints from actual KPI items on the page
        for item in kpis:
            label = str(item.get("label") or "")
            low = f"{label} {item.get('valueKey')}".lower()
            for hint in chart_keys:
                if hint.replace("_", " ") in low or hint in low:
                    present_hints[hint] = label
        # Also seed maintenance hints from PRD when it's a maintenance app
        if _maint_prd and not present_hints:
            for hint, (dk, default_label) in chart_keys.items():
                if hint in ("mtbf", "mttr", "downtime", "work_order"):
                    present_hints[hint] = default_label
        if not present_hints and not kpis:
            return defn

        new_blocks: List[Any] = []
        for block in page.blocks:
            if not isinstance(block, dict) or block.get("type") not in ("lineChart", "barChart"):
                new_blocks.append(block)
                continue
            b = dict(block)
            series = [dict(s) for s in (b.get("series") or []) if isinstance(s, dict)]
            keys = {str(s.get("dataKey")) for s in series}
            title_low = str(b.get("title") or "").lower()

            def _add_chart_series(hint: str) -> None:
                if hint not in chart_keys or hint not in present_hints:
                    return
                dk, default_name = chart_keys[hint]
                if dk in keys:
                    return
                series.append({"dataKey": dk, "name": present_hints.get(hint, default_name)})
                keys.add(dk)

            if b.get("type") == "lineChart":
                # Count how many lineCharts are on the page
                line_count = sum(
                    1 for blk in page.blocks
                    if isinstance(blk, dict) and blk.get("type") == "lineChart"
                )
                for hint in present_hints:
                    # Match by title OR fill in the only trend chart on the page
                    if (
                        hint in title_low
                        or hint.replace("_", " ") in title_low
                        or line_count == 1  # sole trend chart gets all KPI hints
                    ):
                        _add_chart_series(hint)
            if b.get("type") == "barChart" and re.search(
                r"category|performance|by\s+\w+|pareto|breakdown", title_low
            ):
                if "work_order" in present_hints and re.search(r"work order|wo\b", title_low):
                    _add_chart_series("work_order")

            if series:
                b["series"] = series[:6]
            new_blocks.append(b)
        page.blocks = new_blocks
    return defn


def consolidate_events_table_on_dashboard(
    defn: AppDefinition, *, prd: str = "", requirement: str = ""
) -> AppDefinition:
    """Move dataTable from Analytics to Dashboard; Analytics stays charts-only."""
    from app_builder.services.page_layout_intent import resolve_dashboard_page_id
    from app_builder.services.prd_driven_layout import _maintenance_analytics_columns

    pages = list(defn.pages)
    nav = list(defn.navigation or [])
    dash_id = resolve_dashboard_page_id([(p.id, p.title) for p in pages], [n.pageId for n in nav])
    if not dash_id:
        return defn

    dash_page = next((p for p in pages if p.id == dash_id), None)
    if dash_page is None:
        return defn

    pulled_tables: List[Dict[str, Any]] = []
    new_pages = []
    for page in pages:
        if page.id == dash_id:
            new_pages.append(page)
            continue
        blob = f"{page.id} {page.title}".lower()
        if "analytic" not in blob:
            new_pages.append(page)
            continue
        kept: List[Any] = []
        for block in page.blocks or []:
            if isinstance(block, dict) and block.get("type") == "dataTable":
                pulled_tables.append(dict(block))
            else:
                kept.append(block)
        new_pages.append(type(page)(id=page.id, title=page.title, subtitle=page.subtitle, blocks=kept))

    dash_page = next(p for p in new_pages if p.id == dash_id)
    dash_blocks = list(dash_page.blocks or [])
    from app_builder.services.block_normalizer import _columns_are_invalid, repair_data_table_columns

    from app_builder.services.block_normalizer import _prompt_requests_maintenance

    maintenance_prd = _prompt_requests_maintenance(requirement, prd)
    repaired_blocks: List[Any] = []
    for b in dash_blocks:
        if isinstance(b, dict) and b.get("type") == "dataTable":
            b = repair_data_table_columns(b)
            if maintenance_prd and _columns_are_invalid(b.get("columns") or []):
                b = {**b, "columns": _maintenance_analytics_columns(), "title": "Recent maintenance events"}
        repaired_blocks.append(b)
    dash_blocks = repaired_blocks
    has_table = any(isinstance(b, dict) and b.get("type") == "dataTable" for b in dash_blocks)
    if not has_table:
        table_block: Optional[Dict[str, Any]] = None
        if pulled_tables:
            table_block = pulled_tables[0]
        elif maintenance_prd:
            table_block = {
                "type": "dataTable",
                "title": "Recent maintenance events",
                "dataKey": f"table_{dash_id}",
                "columns": _maintenance_analytics_columns(),
                "searchable": True,
                "detailView": True,
            }
        if table_block:
            table_block = {
                **table_block,
                "dataKey": f"table_{dash_id}",
                "title": table_block.get("title") or ("Recent maintenance events" if maintenance_prd else "Records"),
                "detailView": True,
                "searchable": True,
            }
            if not table_block.get("columns") and maintenance_prd:
                table_block["columns"] = _maintenance_analytics_columns()
            if table_block.get("columns"):
                dash_blocks.append(table_block)

    dash_page = type(dash_page)(
        id=dash_page.id,
        title=dash_page.title,
        subtitle=dash_page.subtitle,
        blocks=dash_blocks,
    )
    final_pages = [dash_page if p.id == dash_id else p for p in new_pages]
    defn.pages = final_pages
    return align_dashboard_charts_to_kpis(defn, dashboard_page_id=dash_id, prd=prd)


def apply_dashboard_analytics_product_rules(
    defn: AppDefinition,
    *,
    prd: str = "",
    requirement: str = "",
) -> AppDefinition:
    from app_builder.services.page_layout_intent import resolve_dashboard_page_id

    dash_id = resolve_dashboard_page_id(
        [(p.id, p.title) for p in defn.pages],
        [n.pageId for n in (defn.navigation or [])],
    )
    defn = consolidate_events_table_on_dashboard(defn, prd=prd, requirement=requirement)
    if dash_id:
        defn = align_dashboard_charts_to_kpis(defn, dashboard_page_id=dash_id, prd=prd)
    return defn
