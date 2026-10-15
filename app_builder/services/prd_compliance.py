"""PRD checklist extraction and gap detection (domain-agnostic)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from app_builder.schemas.app_definition import AppDefinition, PageDef


def _bullets(text: str) -> List[str]:
    out: List[str] = []
    for line in text.splitlines():
        m = re.match(r"^[\s]*[-*•]\s+(.+)$", line.strip())
        if m:
            out.append(m.group(1).strip())
    return out


def extract_expected_kpis(section: str) -> List[str]:
    labels: List[str] = []
    for line in _bullets(section):
        if re.search(r"\bchart\b|use case|cards displaying|^\*\*", line, re.I):
            continue
        if re.search(r"line chart|bar chart|donut|pie chart|pareto", line, re.I):
            continue
        if re.search(r"kpi|yield|defect rate|scrap rate|rework|copq|units inspected|pass rate|open quality", line, re.I):
            label = line.split(":")[-1].strip() if ":" in line else line
            label = re.sub(r"^\d+[\).\s]+", "", label)
            label = re.sub(r"^\*+|\*+$", "", label).strip()
            if 2 < len(label) < 80:
                labels.append(label)
    block = re.search(r"quality kpis?:\s*\n((?:\s*[-*•].+\n?)+)", section, re.I)
    if block:
        for line in block.group(1).splitlines():
            m = re.match(r"^[\s]*[-*•]\s+(.+)$", line.strip())
            if m:
                labels.append(m.group(1).strip())
    return list(dict.fromkeys(labels))


def extract_expected_table_columns(section: str) -> List[str]:
    cols: List[str] = []
    in_cols = False
    for line in section.splitlines():
        stripped = line.strip()
        if re.match(r"^columns?\s*:?\s*$", stripped, re.I) or re.match(r"^columns?\s*:\s*$", stripped, re.I):
            in_cols = True
            continue
        if re.search(r"columns?\s*:", stripped, re.I) and "-" not in stripped:
            rest = stripped.split(":", 1)[-1]
            for part in re.split(r",|;", rest):
                name = part.strip()
                if name:
                    cols.append(name)
            continue
        if in_cols:
            bm = re.match(r"^[\s]*[-*•]\s+(.+)$", stripped)
            if bm:
                cols.append(bm.group(1).strip())
            elif stripped and not stripped.startswith("#"):
                in_cols = False
    m = re.search(r"columns?:\s*\n((?:\s*[-*•].+\n?)+)", section, re.I)
    if m:
        for line in m.group(1).splitlines():
            bm = re.match(r"^[\s]*[-*•]\s+(.+)$", line.strip())
            if bm:
                cols.append(bm.group(1).strip())
    return [c for c in dict.fromkeys(cols) if c.lower() != "actions"]


def page_gaps(page: PageDef, section: str) -> Dict[str, Any]:
    """Return missing KPI labels, table columns, and block count issues."""
    from app_builder.services.page_layout_intent import infer_page_layout_intent

    gaps: Dict[str, Any] = {"missing_kpis": [], "missing_columns": [], "block_types": []}
    from app_builder.services.page_layout_intent import resolve_dashboard_page_id

    dash_id = resolve_dashboard_page_id([(page.id, page.title)], None)
    layout_intent = infer_page_layout_intent(
        section,
        page.title,
        page_id=page.id,
        dashboard_page_id=dash_id,
    )
    expected_kpis = extract_expected_kpis(section) if layout_intent.wants_kpis else []
    expected_cols = extract_expected_table_columns(section)

    found_kpi_labels: Set[str] = set()
    found_fields: Set[str] = set()
    types: List[str] = []

    for block in page.blocks:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type") or "")
        types.append(btype)
        if btype == "kpiRow":
            for item in block.get("items") or []:
                if isinstance(item, dict) and item.get("label"):
                    found_kpi_labels.add(str(item["label"]).lower())
        if btype == "dataTable":
            for col in block.get("columns") or []:
                if isinstance(col, dict):
                    if col.get("header"):
                        found_fields.add(str(col["header"]).lower())
                    if col.get("field"):
                        found_fields.add(str(col["field"]).lower())

    for kpi in expected_kpis:
        if kpi.lower() not in found_kpi_labels and not any(kpi.lower() in f for f in found_kpi_labels):
            gaps["missing_kpis"].append(kpi)

    for col in expected_cols:
        slug = re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")
        if col.lower() not in found_fields and slug not in found_fields:
            gaps["missing_columns"].append(col)

    gaps["block_types"] = types
    gaps["needs_charts"] = bool(layout_intent.wants_charts)
    gaps["has_chart"] = any(t in types for t in ("lineChart", "barChart", "pieChart"))
    gaps["needs_filters"] = bool(layout_intent.wants_filters)
    gaps["has_filter"] = "filterBar" in types
    return gaps


def compliance_report(defn: AppDefinition, section_map: Dict[str, str]) -> List[Tuple[str, Dict[str, Any]]]:
    issues: List[Tuple[str, Dict[str, Any]]] = []
    for page in defn.pages:
        section = section_map.get(page.id, "")
        gaps = page_gaps(page, section)
        if (
            gaps.get("missing_kpis")
            or gaps.get("missing_columns")
            or (gaps.get("needs_charts") and not gaps.get("has_chart"))
            or (gaps.get("needs_filters") and not gaps.get("has_filter"))
        ):
            issues.append((page.id, gaps))
    return issues
