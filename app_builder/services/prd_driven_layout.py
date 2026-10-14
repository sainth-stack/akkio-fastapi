"""Build page blocks by parsing PRD section text (no fixed domain templates)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Set

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
    if re.search(r"\bkpi\b|yield|defect rate|scrap rate|rework|copq|units inspected|pass rate|open quality", line, re.I):
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


def _extract_kpi_items(section: str, page_id: str) -> List[Dict[str, str]]:
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

    for line in _bullets(section):
        if _is_kpi_candidate(line):
            parts = re.split(r"[:,]", line, maxsplit=1)
            label = parts[-1].strip() if len(parts) > 1 else line
            add(label)

    return items[:8]


def _extract_table_columns(section: str) -> List[Dict[str, str]]:
    from app_builder.services.prd_compliance import extract_expected_table_columns

    cols: List[Dict[str, str]] = []
    for name in extract_expected_table_columns(section):
        field = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        if field == "actions":
            continue
        cols.append(
            {
                "field": field or f"col_{len(cols)}",
                "header": name,
                **({"align": "right"} if "rate" in field or "units" in field else {}),
            }
        )
    for line in section.splitlines():
        if re.search(r"columns?\s*:", line, re.I) and "-" not in line:
            rest = line.split(":", 1)[-1]
            for part in re.split(r",|;|\|", rest):
                name = part.strip()
                if name and name.lower() != "actions":
                    field = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
                    cols.append({"field": field or f"col_{len(cols)}", "header": name})
    return _dedupe_columns(cols)[:16]


def _extract_charts(section: str, page_id: str, max_charts: int) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    chart_idx = 0
    seen_titles: Set[str] = set()

    for line in _bullets(section):
        if not _is_chart_bullet(line):
            continue
        title = re.sub(r"^\*+|\*+$", "", line)
        title = re.sub(r"^(line|bar|donut|pie)\s+chart\s*:\s*", "", title, flags=re.I).strip()
        title = title[:72] or "Trend"
        norm = title.lower()
        if norm in seen_titles:
            continue
        seen_titles.add(norm)
        chart_idx += 1
        if re.search(r"donut|pie", line, re.I):
            blocks.append(
                {
                    "type": "pieChart",
                    "title": title,
                    "dataKey": f"pie_{page_id}_{chart_idx}",
                    "height": 280,
                }
            )
        elif re.search(r"bar chart|pareto|comparison", line, re.I):
            blocks.append(
                {
                    "type": "barChart",
                    "title": title,
                    "dataKey": f"series_{page_id}_bar_{chart_idx}",
                    "xAxisKey": "period",
                    "series": [
                        {"dataKey": "actual", "name": "Actual"},
                        {"dataKey": "benchmark", "name": "Benchmark"},
                    ],
                    "height": 300,
                }
            )
        else:
            blocks.append(
                {
                    "type": "lineChart",
                    "title": title,
                    "dataKey": f"series_{page_id}_line_{chart_idx}",
                    "xAxisKey": "period",
                    "series": [
                        {"dataKey": "value", "name": "Value"},
                        {"dataKey": "target", "name": "Target"},
                    ],
                    "height": 300,
                }
            )
        if len(blocks) >= max_charts:
            break
    return blocks


def ensure_minimum_blocks(blocks: List[Dict[str, Any]], page_id: str, title: str) -> List[Dict[str, Any]]:
    """Guarantee at least two blocks so PageBlocksPlan validation never fails."""
    out = list(blocks or [])
    if len(out) >= 2:
        return out
    types = {b.get("type") for b in out}
    if "filterBar" not in types:
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
    if len(out) < 2 and "kpiRow" not in types:
        out.insert(
            1 if out and out[0].get("type") == "filterBar" else 0,
            {
                "type": "kpiRow",
                "dataKey": f"kpis_{page_id}",
                "items": [
                    {"label": f"{title} — primary", "valueKey": "primary_metric", "color": "primary"},
                    {"label": f"{title} — volume", "valueKey": "volume_metric", "color": "info"},
                ],
            },
        )
    if len(out) < 2:
        return _outline_saas_blocks(page_id, title, "")
    return out


def _max_charts_for_page(page_id: str) -> int:
    low = page_id.lower()
    if low in ("dashboard", "home", "overview"):
        return 5
    if "analytic" in low:
        return 4
    if "report" in low:
        return 3
    if "session" in low:
        return 2
    return 3


def blocks_for_page(page_id: str, title: str, prd_section: str) -> List[Dict[str, Any]]:
    section = prd_section or ""
    if _section_looks_like_full_document(section, page_id):
        return _outline_saas_blocks(page_id, title, section[:1200])

    blocks: List[Dict[str, Any]] = []
    low = section.lower()

    if re.search(r"filter|date.range|export|search", low):
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

    if re.search(r"export|create|new session|pdf|csv", low):
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
    if kpis:
        blocks.append({"type": "kpiRow", "dataKey": f"kpis_{page_id}", "items": kpis[:8]})

    blocks.extend(_extract_charts(section, page_id, _max_charts_for_page(page_id)))

    cols = _extract_table_columns(section)
    if cols or re.search(r"table", low):
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
                "detailView": page_id in ("sessions", "session") or "session" in page_id,
            }
        )

    structural = [b for b in blocks if b.get("type") not in ("text",)]
    if not structural:
        blocks = _outline_saas_blocks(page_id, title, section)
    return ensure_minimum_blocks(blocks, page_id, title)


def _outline_saas_blocks(page_id: str, title: str, section: str) -> List[Dict[str, Any]]:
    """PRD-agnostic layout when section has no parseable widgets (uses page title + excerpt only)."""
    low = f"{title} {section} {page_id}".lower()
    blocks: List[Dict[str, Any]] = [
        {
            "type": "filterBar",
            "filters": [
                {"id": "search", "label": "Search", "filterType": "search"},
                {"id": "date", "label": "Date range", "filterType": "dateRange"},
            ],
        },
    ]
    if "report" in low:
        blocks.append(
            {
                "type": "actionBar",
                "actions": [
                    {"id": "export_csv", "label": "Export CSV", "actionType": "exportCsv"},
                    {"id": "export_pdf", "label": "Export PDF", "actionType": "exportPdf"},
                ],
            }
        )
    if "session" in low:
        blocks.append(
            {
                "type": "actionBar",
                "actions": [{"id": "create", "label": "New session", "actionType": "create"}],
            }
        )

    blocks.append(
        {
            "type": "kpiRow",
            "dataKey": f"kpis_{page_id}",
            "items": [
                {"label": f"{title} — primary", "valueKey": "primary_metric", "color": "primary"},
                {"label": f"{title} — secondary", "valueKey": "secondary_metric", "color": "success"},
                {"label": f"{title} — risk", "valueKey": "risk_metric", "color": "warning"},
                {"label": f"{title} — volume", "valueKey": "volume_metric", "color": "info"},
            ],
        }
    )
    blocks.append(
        {
            "type": "lineChart",
            "title": f"{title} — trend",
            "dataKey": f"series_{page_id}_trend",
            "xAxisKey": "period",
            "series": [
                {"dataKey": "actual", "name": "Actual", "color": "#1976d2"},
                {"dataKey": "target", "name": "Target", "color": "#94a3b8"},
            ],
            "height": 300,
        }
    )
    if "analytic" in low or "report" in low:
        blocks.append(
            {
                "type": "barChart",
                "title": f"{title} — comparison",
                "dataKey": f"series_{page_id}_compare",
                "xAxisKey": "period",
                "series": [
                    {"dataKey": "value_a", "name": "Series A", "color": "#1976d2"},
                    {"dataKey": "value_b", "name": "Series B", "color": "#d32f2f"},
                ],
                "height": 280,
            }
        )
    if "dashboard" in low or page_id == "dashboard":
        blocks.append(
            {
                "type": "pieChart",
                "title": f"{title} — distribution",
                "dataKey": f"pie_{page_id}_mix",
                "height": 260,
            }
        )
    blocks.append(
        {
            "type": "dataTable",
            "title": f"{title} — records",
            "dataKey": f"table_{page_id}",
            "columns": [
                {"field": "id", "header": "ID"},
                {"field": "name", "header": "Name"},
                {"field": "status", "header": "Status"},
                {"field": "updated_at", "header": "Updated"},
            ],
            "searchable": True,
            "detailView": "session" in low,
        }
    )
    return blocks
