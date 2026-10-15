"""Build page blocks by parsing PRD section text (no fixed domain templates)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

_JUNK_LINE = re.compile(
    r"use case\s*\d|^\*\*|charts?\s+and\s+tables|clear kpi|cards displaying|displays\s+",
    re.I,
)


def _bullets(text: str) -> List[str]:
    lines: List[str] = []
    for line in text.splitlines():
        m = re.match(r"^[\s]*[-*•]\s+(.+)$", line.strip())
        if m:
            lines.append(m.group(1).strip())
    return lines


def _is_chart_bullet(line: str) -> bool:
    if _JUNK_LINE.search(line):
        return False
    return bool(
        re.search(
            r"^(line|bar|donut|pie)\s+chart\b|pareto\s+chart|^\*\*(line|bar|donut|pie)\s+chart",
            line,
            re.I,
        )
    )


def _is_kpi_candidate(line: str) -> bool:
    if _JUNK_LINE.search(line) or _is_chart_bullet(line):
        return False
    if re.search(r"\bchart\b|\btable\b|\bfilters?\b|\bexport\b|\bscreen\b", line, re.I):
        return False
    if re.search(
        r"\bkpi\b|yield|defect rate|scrap rate|rework|copq|units inspected|pass rate|open quality|"
        r"work order|mtbf|mttr|mean time|downtime|maintenance|availability|oee",
        line,
        re.I,
    ):
        return True
    return False


def _section_looks_like_full_document(section: str, page_id: str) -> bool:
    if not section:
        return False
    headings = len(re.findall(r"(?m)^#{1,4}\s+", section))
    if headings >= 3:
        return True
    if len(section) > 3500 and section.lower().count("chart") > 6:
        return True
    page_hits = len(re.findall(rf"\b{re.escape(page_id)}\b", section, re.I))
    if len(section) > 6000 and page_hits < 2:
        return True
    return False


def _dedupe_columns(cols: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: Set[str] = set()
    out: List[Dict[str, str]] = []
    for col in cols:
        field = str(col.get("field") or "")
        if not field or field in seen:
            continue
        seen.add(field)
        out.append(col)
    return out


def _default_maintenance_kpi_items() -> List[Dict[str, str]]:
    labels = (
        "Total Maintenance Work Orders",
        "Mean Time Between Failures (MTBF)",
        "Mean Time to Repair (MTTR)",
    )
    colors = ("primary", "success", "warning", "error", "info", "default")
    out: List[Dict[str, str]] = []
    for i, label in enumerate(labels):
        vk = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:40]
        out.append({"label": label, "valueKey": vk, "color": colors[i % len(colors)]})
    return out


def _extract_kpi_items(section: str, page_id: str) -> List[Dict[str, str]]:
    from app_builder.services.page_layout_intent import _subsection_text

    items: List[Dict[str, str]] = []
    colors = ("primary", "success", "warning", "error", "info", "default")
    seen_labels: Set[str] = set()

    def add(label: str) -> None:
        label = re.sub(r"^\*+|\*+$", "", label).strip()
        label = re.sub(r"^[-*•\s]+", "", label)
        if len(label) < 3 or len(label) > 56:
            return
        if _JUNK_LINE.search(label):
            return
        key = label.lower()
        if key in seen_labels:
            return
        seen_labels.add(key)
        vk = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:40]
        items.append(
            {
                "label": label,
                "valueKey": vk or f"metric_{len(items)}",
                "color": colors[len(items) % len(colors)],
            }
        )

    kpi_block = _subsection_text(section, "kpis") or _subsection_text(section, "kpi")
    for line in _bullets(kpi_block):
        parts = re.split(r"[:,]", line, maxsplit=1)
        label = parts[-1].strip() if len(parts) > 1 else line
        add(label)

    for line in _bullets(section):
        if _is_kpi_candidate(line):
            parts = re.split(r"[:,]", line, maxsplit=1)
            label = parts[-1].strip() if len(parts) > 1 else line
            add(label)

    if not items and re.search(r"maintenance|mtbf|mttr|work order", section, re.I):
        return _default_maintenance_kpi_items()
    return items[:8]


def _maintenance_analytics_columns() -> List[Dict[str, str]]:
    return [
        {"field": "work_order_id", "header": "Work order"},
        {"field": "machine", "header": "Machine"},
        {"field": "production_line", "header": "Production line"},
        {"field": "equipment", "header": "Equipment"},
        {"field": "shift", "header": "Shift"},
        {"field": "event_type", "header": "Event type"},
        {"field": "root_cause", "header": "Root cause"},
        {"field": "severity", "header": "Severity"},
        {"field": "status", "header": "Status"},
        {"field": "downtime_hours", "header": "Downtime (h)", "align": "right"},
        {"field": "mtbf_hours", "header": "MTBF (h)", "align": "right"},
        {"field": "mttr_hours", "header": "MTTR (h)", "align": "right"},
        {"field": "occurred_at", "header": "Occurred at"},
    ]


def _extract_table_columns(section: str) -> List[Dict[str, str]]:
    from app_builder.services.prd_compliance import extract_expected_table_columns

    cols: List[Dict[str, str]] = []
    for name in extract_expected_table_columns(section):
        field = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        if field == "actions":
            continue
        if not field:
            field = f"field_{len(cols)}"
        cols.append(
            {
                "field": field,
                "header": name,
                **({"align": "right"} if "rate" in field or "units" in field or "hours" in field else {}),
            }
        )
    for line in section.splitlines():
        if re.search(r"columns?\s*:", line, re.I) and "-" not in line:
            rest = line.split(":", 1)[-1]
            for part in re.split(r",|;|\|", rest):
                name = part.strip()
                if name and name.lower() != "actions":
                    field = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
                    if not field:
                        field = f"field_{len(cols)}"
                    cols.append({"field": field, "header": name})
    return _dedupe_columns(cols)[:16]


def _extract_charts(section: str, page_id: str, max_charts: int) -> List[Dict[str, Any]]:
    from app_builder.services.chart_spec_parser import chart_block_from_bullet

    blocks: List[Dict[str, Any]] = []
    chart_idx = 0
    seen_titles: Set[str] = set()

    for line in _bullets(section):
        if not _is_chart_bullet(line):
            continue
        chart_idx += 1
        block = chart_block_from_bullet(line, page_id, chart_idx)
        norm = str(block.get("title") or "").lower()
        if norm in seen_titles:
            continue
        seen_titles.add(norm)
        blocks.append(block)
        if len(blocks) >= max_charts:
            break
    return blocks


def pad_blocks_to_minimum(
    blocks: List[Dict[str, Any]],
    page_id: str,
    title: str,
    section: str,
    *,
    dashboard_page_id: Optional[str] = None,
    min_count: int = 2,
) -> List[Dict[str, Any]]:
    """Ensure len(blocks) >= min_count without breaking route layout policies."""
    from app_builder.services.page_layout_intent import infer_page_layout_intent

    out = list(blocks or [])
    if len(out) >= min_count:
        return out
    intent = infer_page_layout_intent(
        section,
        title,
        page_id=page_id,
        dashboard_page_id=dashboard_page_id,
    )
    while len(out) < min_count:
        types = {b.get("type") for b in out}
        if intent.wants_charts:
            for ch in _extract_charts(section, page_id, intent.chart_cap or 2):
                out.append(ch)
                if len(out) >= min_count:
                    return out[:24]
        if intent.wants_kpis and "kpiRow" not in types:
            items = _extract_kpi_items(section, page_id)[: intent.kpi_cap or 4]
            if items:
                out.insert(0, {"type": "kpiRow", "dataKey": f"kpis_{page_id}", "items": items})
                if len(out) >= min_count:
                    return out[:24]
        if intent.wants_table and "dataTable" not in types:
            cols = _extract_table_columns(section) or [
                {"field": "id", "header": "ID"},
                {"field": "name", "header": "Name"},
                {"field": "status", "header": "Status"},
            ]
            out.append(
                {
                    "type": "dataTable",
                    "title": f"{title} records",
                    "dataKey": f"table_{page_id}",
                    "columns": cols,
                    "searchable": True,
                    "detailView": intent.table_detail,
                }
            )
            if len(out) >= min_count:
                return out[:24]
        out.append({"type": "text", "content": "", "hidden": True})
    return out[:24]


def ensure_minimum_blocks(
    blocks: List[Dict[str, Any]],
    page_id: str,
    title: str,
    *,
    section: str = "",
    dashboard_page_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Guarantee at least two blocks so PageBlocksPlan validation never fails."""
    from app_builder.services.page_layout_intent import infer_page_layout_intent

    intent = infer_page_layout_intent(
        section,
        title,
        page_id=page_id,
        dashboard_page_id=dashboard_page_id,
    )
    out = list(blocks or [])
    if len(out) >= 2:
        return out
    types = {b.get("type") for b in out}
    if intent.wants_filters and "filterBar" not in types:
        out.insert(
            0,
            {
                "type": "filterBar",
                "filters": [
                    {"id": "search", "label": "Search", "filterType": "search"},
                    {"id": "date", "label": "Date range", "filterType": "dateRange"},
                ],
            },
        )
    types = {b.get("type") for b in out}
    if len(out) < 2 and intent.wants_charts and not ({b.get("type") for b in out} & {"lineChart", "barChart", "pieChart"}):
        out.extend(_extract_charts(section, page_id, 1))
    if len(out) < 2 and intent.wants_table and "dataTable" not in {b.get("type") for b in out}:
        out.append(
            {
                "type": "dataTable",
                "title": f"{title} records",
                "dataKey": f"table_{page_id}",
                "columns": [
                    {"field": "id", "header": "ID"},
                    {"field": "name", "header": "Name"},
                    {"field": "status", "header": "Status"},
                ],
                "searchable": True,
                "detailView": intent.table_detail,
            }
        )
    if len(out) < 2:
        out = _outline_saas_blocks(page_id, title, section or "", intent)
    return pad_blocks_to_minimum(
        out,
        page_id,
        title,
        section or "",
        dashboard_page_id=dashboard_page_id,
        min_count=2,
    )


def blocks_for_page(
    page_id: str,
    title: str,
    prd_section: str,
    *,
    dashboard_page_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from app_builder.services.page_layout_intent import (
        ensure_blocks_match_intent,
        infer_page_layout_intent,
        is_dashboard_like,
    )

    section = prd_section or ""
    if dashboard_page_id is None and is_dashboard_like(page_id, title):
        dashboard_page_id = page_id
    intent = infer_page_layout_intent(
        section, title, page_id=page_id, dashboard_page_id=dashboard_page_id
    )

    if _section_looks_like_full_document(section, page_id):
        blocks = _outline_saas_blocks(page_id, title, section[:1200], intent)
        return ensure_minimum_blocks(
            ensure_blocks_match_intent(
                blocks,
                intent,
                page_id=page_id,
                title=title,
                section=section,
                dashboard_page_id=dashboard_page_id,
            ),
            page_id,
            title,
            section=section,
        )

    blocks: List[Dict[str, Any]] = []
    low = section.lower()

    if intent.wants_filters:
        filters: List[Dict[str, Any]] = [{"id": "search", "label": "Search", "filterType": "search"}]
        if "date" in low:
            filters.append({"id": "date", "label": "Date range", "filterType": "dateRange"})
        if "line" in low or "production" in low:
            ok = f"options_{page_id}_lines"
            filters.append({"id": "line", "label": "Production line", "filterType": "select", "optionsKey": ok})
        if "status" in low:
            ok = f"options_{page_id}_status"
            filters.append({"id": "status", "label": "Status", "filterType": "select", "optionsKey": ok})
        blocks.append({"type": "filterBar", "filters": filters})

    if intent.wants_export_actions or intent.wants_create_action or re.search(
        r"export|create|new session|pdf|csv", low
    ):
        actions: List[Dict[str, Any]] = []
        if re.search(r"export|csv", low):
            actions.append({"id": "export_csv", "label": "Export CSV", "actionType": "exportCsv"})
        if "pdf" in low:
            actions.append({"id": "export_pdf", "label": "Export PDF", "actionType": "exportPdf"})
        if re.search(r"create|new", low):
            actions.append({"id": "create", "label": "Create", "actionType": "primary"})
        if actions:
            blocks.append({"type": "actionBar", "actions": actions})

    from app_builder.services.prd_compliance import extract_expected_kpis

    kpis = _extract_kpi_items(section, page_id)
    for label in extract_expected_kpis(section):
        if _JUNK_LINE.search(label) or _is_chart_bullet(label):
            continue
        vk = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:40]
        if not any(i.get("label", "").lower() == label.lower() for i in kpis):
            kpis.append(
                {
                    "label": label,
                    "valueKey": vk,
                    "color": ("primary", "success", "warning", "error", "info", "default")[len(kpis) % 6],
                }
            )
    if intent.wants_kpis and kpis:
        cap = intent.kpi_cap or 8
        blocks.append({"type": "kpiRow", "dataKey": f"kpis_{page_id}", "items": kpis[:cap]})

    if intent.wants_charts:
        blocks.extend(_extract_charts(section, page_id, intent.chart_cap or 3))

    cols = _extract_table_columns(section)
    if intent.wants_table and (cols or re.search(r"table", low)):
        if not cols and re.search(r"\banalytics?\b", f"{page_id} {title}".lower()):
            if re.search(r"maintenance|mtbf|mttr|equipment|work order", low):
                cols = _maintenance_analytics_columns()
        if not cols:
            cols = [
                {"field": "id", "header": "ID"},
                {"field": "name", "header": "Name"},
                {"field": "status", "header": "Status"},
                {"field": "updated_at", "header": "Updated"},
            ]
        blocks.append(
            {
                "type": "dataTable",
                "title": f"{title} records",
                "dataKey": f"table_{page_id}",
                "columns": cols,
                "searchable": True,
                "detailView": intent.table_detail,
            }
        )

    structural = [b for b in blocks if b.get("type") not in ("text",)]
    if not structural:
        blocks = _outline_saas_blocks(page_id, title, section, intent)
    shaped = ensure_blocks_match_intent(
        blocks,
        intent,
        page_id=page_id,
        title=title,
        section=section,
        dashboard_page_id=dashboard_page_id,
    )
    return ensure_minimum_blocks(shaped, page_id, title, section=section)


def _outline_saas_blocks(
    page_id: str,
    title: str,
    section: str,
    intent: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Fallback layout driven by PRD intent (purpose/widgets), not a fixed page-id template."""
    from app_builder.services.page_layout_intent import PageLayoutIntent, infer_page_layout_intent

    if intent is None:
        intent = infer_page_layout_intent(section, title)
    blocks: List[Dict[str, Any]] = []

    if intent.wants_filters:
        blocks.append(
            {
                "type": "filterBar",
                "filters": [
                    {"id": "search", "label": "Search", "filterType": "search"},
                    {"id": "date", "label": "Date range", "filterType": "dateRange"},
                ],
            }
        )
    if intent.wants_export_actions or intent.wants_create_action:
        actions: List[Dict[str, Any]] = []
        if intent.wants_export_actions:
            actions.extend(
                [
                    {"id": "export_csv", "label": "Export CSV", "actionType": "exportCsv"},
                    {"id": "export_pdf", "label": "Export PDF", "actionType": "exportPdf"},
                ]
            )
        if intent.wants_create_action:
            actions.append({"id": "create", "label": "Create", "actionType": "primary"})
        blocks.append({"type": "actionBar", "actions": actions})

    if intent.wants_kpis:
        items = _extract_kpi_items(section, page_id)[: intent.kpi_cap or 4]
        if items:
            blocks.append({"type": "kpiRow", "dataKey": f"kpis_{page_id}", "items": items})

    if intent.wants_charts:
        blocks.extend(_extract_charts(section, page_id, intent.chart_cap or 2))

    if intent.wants_table:
        cols = _extract_table_columns(section) or [
            {"field": "id", "header": "ID"},
            {"field": "name", "header": "Name"},
            {"field": "status", "header": "Status"},
        ]
        blocks.append(
            {
                "type": "dataTable",
                "title": f"{title} — records",
                "dataKey": f"table_{page_id}",
                "columns": cols,
                "searchable": True,
                "detailView": intent.table_detail,
            }
        )
    return blocks
