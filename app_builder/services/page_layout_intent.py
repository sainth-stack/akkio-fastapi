"""Infer per-page SaaS layout from PRD section text (widget declarations + purpose semantics)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class PageLayoutIntent:
    wants_filters: bool = False
    wants_export_actions: bool = False
    wants_create_action: bool = False
    wants_kpis: bool = False
    kpi_cap: int = 0
    wants_charts: bool = False
    chart_cap: int = 0
    wants_table: bool = False
    table_detail: bool = False


_SUBSECTION = re.compile(
    r"(?im)^\s*\*{0,2}\s*(purpose|kpis?|charts?|filters?|actions?|table\s*columns?|columns?)\s*:?\s*\*{0,2}\s*$"
)


def _subsection_text(section: str, name: str) -> str:
    if not section:
        return ""
    lines = section.splitlines()
    name_re = re.compile(rf"(?i)^\s*\*{{0,2}}\s*{name}\s*:?\s*\*{{0,2}}\s*$")
    start = None
    for i, line in enumerate(lines):
        if name_re.match(line.strip()):
            start = i + 1
            break
    if start is None:
        return ""
    out: List[str] = []
    for line in lines[start:]:
        if _SUBSECTION.match(line.strip()):
            break
        if re.match(r"^#{1,4}\s+", line.strip()):
            break
        out.append(line)
    return "\n".join(out).strip()


def _has_subsection_header(section: str, name: str) -> bool:
    name_re = re.compile(rf"(?i)^\s*\*{{0,2}}\s*{name}\s*:?\s*\*{{0,2}}\s*$")
    return any(name_re.match(line.strip()) for line in (section or "").splitlines())


def _bullets(text: str) -> List[str]:
    lines: List[str] = []
    for line in (text or "").splitlines():
        m = re.match(r"^[\s]*[-*•]\s+(.+)$", line.strip())
        if m:
            lines.append(m.group(1).strip())
    return lines


def _purpose_text(section: str, title: str) -> str:
    block = _subsection_text(section, "purpose")
    if block:
        return block.split("\n")[0].strip()
    m = re.search(r"(?im)purpose\s*:\s*(.+)$", section or "")
    if m:
        return m.group(1).strip()
    return (title or "").strip()


def _subsection_declares_omit(block: str) -> bool:
    if not block:
        return True
    low = block.lower()
    return bool(re.search(r"\bomit\b|\bnone\b|\bn/?a\b|not applicable|leave empty", low))


def is_dashboard_like(page_id: str, title: str = "") -> bool:
    blob = f"{page_id} {title}".lower().replace("_", " ").replace("-", " ")
    return bool(re.search(r"\b(dashboard|home|overview)\b", blob))


def resolve_dashboard_page_id(
    pages: Sequence[Tuple[str, str]],
    nav_page_ids: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Single route that may show top KPI cards (Dashboard / Home / Overview, else first nav page)."""
    for pid, title in pages:
        if is_dashboard_like(pid, title):
            return pid
    if nav_page_ids:
        for pid in nav_page_ids:
            if pid:
                return str(pid)
    if pages:
        return pages[0][0]
    return None


def _apply_kpi_dashboard_gate(intent: PageLayoutIntent, page_id: str, dashboard_page_id: Optional[str]) -> None:
    if dashboard_page_id and page_id and page_id != dashboard_page_id:
        intent.wants_kpis = False
        intent.kpi_cap = 0


def _page_role(page_id: str, title: str) -> str:
    if is_dashboard_like(page_id, title):
        return "dashboard"
    blob = f"{page_id} {title}".lower().replace("_", " ").replace("-", " ")
    if re.search(r"\breports?\b", blob):
        return "reports"
    if re.search(r"\banalytics?\b", blob):
        return "analytics"
    return "other"


def apply_route_layout_policies(
    intent: PageLayoutIntent,
    *,
    page_id: str,
    title: str,
    section: str,
    dashboard_page_id: Optional[str],
) -> None:
    """Enforce common SaaS page roles (dashboard = KPIs only; reports = charts + export; no default filters)."""
    filter_block = _subsection_text(section, "filters") or _subsection_text(section, "filter")
    filter_bullets = _bullets(filter_block)
    if not filter_bullets:
        intent.wants_filters = False

    role = _page_role(page_id, title)
    dash = dashboard_page_id and page_id == dashboard_page_id

    if dash or role == "dashboard":
        intent.wants_charts = False
        intent.chart_cap = 0
        intent.wants_table = False
        intent.wants_filters = False
        if not intent.wants_kpis:
            from app_builder.services.prd_compliance import extract_expected_kpis

            kpis = extract_expected_kpis(section)
            if kpis:
                intent.wants_kpis = True
                intent.kpi_cap = max(intent.kpi_cap, min(8, len(kpis)))
    elif role == "reports":
        intent.wants_kpis = False
        intent.kpi_cap = 0
        intent.wants_table = False
        intent.wants_filters = False
        intent.wants_export_actions = True
        if not intent.wants_charts:
            intent.wants_charts = True
            intent.chart_cap = max(intent.chart_cap, 2)
    elif role == "analytics":
        intent.wants_kpis = False
        intent.kpi_cap = 0
        intent.wants_filters = False
        if not intent.wants_table:
            intent.wants_table = True
        if not intent.wants_charts:
            intent.wants_charts = True
            intent.chart_cap = max(intent.chart_cap, 2)


def _infer_from_purpose_when_sparse(intent: PageLayoutIntent, purpose: str) -> None:
    """Soft hints when PRD page section has no widget subsections — any screen name is allowed."""
    p = (purpose or "").lower()
    if not p:
        return
    if re.search(r"overview|at a glance|executive|command center|real.?time.{0,12}monitor", p):
        intent.wants_kpis = True
        intent.kpi_cap = max(intent.kpi_cap, 4)
        intent.wants_charts = True
        intent.chart_cap = max(intent.chart_cap, 2)
    if re.search(r"\b(search|browse|list|register|catalog|log|directory|grid of)\b", p):
        intent.wants_table = True
        intent.table_detail = bool(re.search(r"detail|drill|row|click", p))
    if re.search(r"\b(export|report|publish|print|download|share)\b", p):
        intent.wants_export_actions = True
    if re.search(r"\b(analy|trend|compare|pareto|insight|breakdown|distribution|forecast|correlat)\b", p):
        intent.wants_charts = True
        intent.chart_cap = max(intent.chart_cap, 3)
    if re.search(r"\b(create|add new|submit|intake)\b", p):
        intent.wants_create_action = True


def infer_page_layout_intent(
    section: str,
    title: str = "",
    *,
    page_id: str = "",
    dashboard_page_id: Optional[str] = None,
) -> PageLayoutIntent:
    """Widgets declared in THIS page's PRD section are authoritative; purpose fills gaps only."""
    from app_builder.services.prd_compliance import extract_expected_table_columns

    intent = PageLayoutIntent()
    purpose = _purpose_text(section, title)

    kpi_block = _subsection_text(section, "kpis") or _subsection_text(section, "kpi")
    chart_block = _subsection_text(section, "charts") or _subsection_text(section, "chart")
    filter_block = _subsection_text(section, "filters") or _subsection_text(section, "filter")
    actions_block = _subsection_text(section, "actions") or _subsection_text(section, "action")
    table_block = (
        _subsection_text(section, "table columns")
        or _subsection_text(section, "table column")
        or _subsection_text(section, "columns")
    )

    kpi_bullets = _bullets(kpi_block)
    chart_bullets = _bullets(chart_block)
    table_bullets = _bullets(table_block)
    table_cols = extract_expected_table_columns(section)

    has_kpi_header = _has_subsection_header(section, "kpis") or _has_subsection_header(section, "kpi")
    has_chart_header = _has_subsection_header(section, "charts") or _has_subsection_header(section, "chart")
    has_table_header = _has_subsection_header(section, "table columns") or _has_subsection_header(section, "columns")
    has_filter_header = _has_subsection_header(section, "filters") or _has_subsection_header(section, "filter")
    has_actions_header = _has_subsection_header(section, "actions") or _has_subsection_header(section, "action")

    if kpi_bullets:
        intent.wants_kpis = True
        intent.kpi_cap = min(8, len(kpi_bullets))
    elif has_kpi_header and _subsection_declares_omit(kpi_block):
        intent.wants_kpis = False
        intent.kpi_cap = 0

    if chart_bullets:
        intent.wants_charts = True
        intent.chart_cap = min(6, len(chart_bullets))
    elif has_chart_header and _subsection_declares_omit(chart_block):
        intent.wants_charts = False
        intent.chart_cap = 0

    if table_bullets or table_cols:
        intent.wants_table = True
    elif has_table_header and _subsection_declares_omit(table_block):
        intent.wants_table = False

    filter_bullets = _bullets(filter_block)
    if filter_bullets:
        intent.wants_filters = True

    action_bullets = _bullets(actions_block)
    if action_bullets:
        intent.wants_export_actions = bool(
            re.search(r"export|pdf|csv|download|print", " ".join(action_bullets), re.I)
        )
        intent.wants_create_action = bool(re.search(r"create|add|new", " ".join(action_bullets), re.I))
    elif has_actions_header and not _subsection_declares_omit(actions_block):
        intent.wants_export_actions = True

    intent.table_detail = bool(re.search(r"detail|drill|row click|session", purpose + section, re.I))

    has_widget_headers = any(
        [has_kpi_header, has_chart_header, has_table_header, has_filter_header, has_actions_header]
    )

    if not has_widget_headers:
        _infer_from_purpose_when_sparse(intent, purpose)

    # Cross-widget hygiene (real products: don't stack everything unless PRD asks)
    if intent.wants_table and intent.wants_charts and not chart_bullets and not has_chart_header:
        if re.search(r"\b(search|browse|list|register)\b", purpose.lower()):
            intent.wants_charts = False
            intent.chart_cap = 0

    if intent.wants_kpis and not kpi_bullets and not has_kpi_header:
        if not re.search(r"overview|executive|kpi|metric|at a glance", purpose, re.I):
            intent.wants_kpis = False
            intent.kpi_cap = 0

    _apply_kpi_dashboard_gate(intent, page_id or title, dashboard_page_id)
    apply_route_layout_policies(
        intent,
        page_id=page_id or title,
        title=title,
        section=section,
        dashboard_page_id=dashboard_page_id,
    )
    return intent


def strip_misplaced_kpi_blocks(
    blocks: List[Dict[str, Any]],
    *,
    title: str,
    page_id: str,
    purpose: str = "",
    section: str = "",
    dashboard_page_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Remove kpiRow from non-dashboard routes; keep only on the dashboard page."""
    if dashboard_page_id and page_id != dashboard_page_id:
        return [b for b in (blocks or []) if isinstance(b, dict) and b.get("type") != "kpiRow"]
    body = section.strip() if section else f"### {title}\n**Purpose:** {purpose or title}\n"
    intent = infer_page_layout_intent(
        body, title, page_id=page_id, dashboard_page_id=dashboard_page_id
    )
    if intent.wants_kpis:
        return _kpi_row_first(list(blocks or []))
    return [b for b in (blocks or []) if isinstance(b, dict) and b.get("type") != "kpiRow"]


def _kpi_row_first(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kpis = [b for b in blocks if isinstance(b, dict) and b.get("type") == "kpiRow"]
    rest = [b for b in blocks if not (isinstance(b, dict) and b.get("type") == "kpiRow")]
    return kpis + rest if kpis else blocks


def enforce_page_role_blocks(defn: Any, *, prd: str = "") -> Any:
    """Strip widgets that contradict route roles (dashboard KPI-only, reports no table, no filter bars unless PRD lists filters)."""
    from app_builder.services.prd_structure_parser import section_for_page

    pages = list(getattr(defn, "pages", []) or [])
    nav = list(getattr(defn, "navigation", []) or [])
    dash_id = resolve_dashboard_page_id(
        [(p.id, p.title) for p in pages],
        [n.pageId for n in nav],
    )
    new_pages = []
    for page in pages:
        section = section_for_page(prd, page.title, page.id) if prd else ""
        intent = infer_page_layout_intent(
            section or f"### {page.title}\n{page.subtitle or ''}",
            page.title,
            page_id=page.id,
            dashboard_page_id=dash_id,
        )
        blocks = apply_layout_intent_to_blocks(
            list(page.blocks or []),
            intent,
            page_id=page.id,
            title=page.title,
        )
        new_pages.append(
            type(page)(id=page.id, title=page.title, subtitle=page.subtitle, blocks=blocks)
        )
    defn.pages = new_pages
    return defn


def enforce_kpis_on_dashboard_only(defn: Any) -> Any:
    """Final guard: KPI cards only on the dashboard route."""
    pages = list(getattr(defn, "pages", []) or [])
    nav = list(getattr(defn, "navigation", []) or [])
    dash_id = resolve_dashboard_page_id(
        [(p.id, p.title) for p in pages],
        [n.pageId for n in nav],
    )
    if not dash_id:
        return defn
    new_pages = []
    for page in pages:
        blocks = list(page.blocks or [])
        if page.id != dash_id:
            blocks = [b for b in blocks if not (isinstance(b, dict) and b.get("type") == "kpiRow")]
        else:
            blocks = _kpi_row_first(blocks)
        new_pages.append(
            type(page)(id=page.id, title=page.title, subtitle=page.subtitle, blocks=blocks)
        )
    defn.pages = new_pages
    return defn


def apply_layout_intent_to_blocks(
    blocks: List[Dict[str, Any]],
    intent: PageLayoutIntent,
    *,
    page_id: str,
    title: str,
) -> List[Dict[str, Any]]:
    """Remove or trim blocks that contradict PRD-derived intent."""
    out: List[Dict[str, Any]] = []
    charts_kept = 0

    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")

        if btype == "filterBar":
            if intent.wants_filters:
                out.append(block)
            continue
        if btype == "actionBar":
            if intent.wants_export_actions or intent.wants_create_action:
                out.append(block)
            continue
        if btype == "kpiRow":
            if not intent.wants_kpis:
                continue
            items = block.get("items") or []
            if intent.kpi_cap and isinstance(items, list):
                block = {**block, "items": items[: intent.kpi_cap]}
            out.append(block)
            continue
        if btype in ("lineChart", "barChart", "pieChart"):
            if not intent.wants_charts:
                continue
            if intent.chart_cap and charts_kept >= intent.chart_cap:
                continue
            charts_kept += 1
            out.append(block)
            continue
        if btype == "dataTable":
            if intent.wants_table:
                block = {**block, "detailView": bool(block.get("detailView") or intent.table_detail)}
                out.append(block)
            continue
        out.append(block)

    return out


def ensure_blocks_match_intent(
    blocks: List[Dict[str, Any]],
    intent: PageLayoutIntent,
    *,
    page_id: str,
    title: str,
    section: str,
    dashboard_page_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """After trimming, add minimal widgets declared in PRD but missing from LLM output."""
    from app_builder.services.prd_driven_layout import (
        _extract_charts,
        _extract_kpi_items,
        _extract_table_columns,
    )

    gated = PageLayoutIntent(**intent.__dict__)
    _apply_kpi_dashboard_gate(gated, page_id, dashboard_page_id)
    shaped = apply_layout_intent_to_blocks(blocks, gated, page_id=page_id, title=title)
    types = {b.get("type") for b in shaped}

    if intent.wants_filters and "filterBar" not in types:
        shaped.insert(
            0,
            {
                "type": "filterBar",
                "filters": [
                    {"id": "search", "label": "Search", "filterType": "search"},
                    {"id": "date", "label": "Date range", "filterType": "dateRange"},
                ],
            },
        )
    if (intent.wants_export_actions or intent.wants_create_action) and "actionBar" not in types:
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
        if actions:
            shaped.insert(1 if shaped and shaped[0].get("type") == "filterBar" else 0, {"type": "actionBar", "actions": actions})

    if gated.wants_kpis and "kpiRow" not in {b.get("type") for b in shaped}:
        items = _extract_kpi_items(section, page_id)[: gated.kpi_cap or 4]
        if items:
            shaped.insert(0, {"type": "kpiRow", "dataKey": f"kpis_{page_id}", "items": items})

    chart_types = {b.get("type") for b in shaped} & {"lineChart", "barChart", "pieChart"}
    if intent.wants_charts and not chart_types:
        cap = intent.chart_cap or 2
        shaped.extend(_extract_charts(section, page_id, cap))

    if intent.wants_table and "dataTable" not in {b.get("type") for b in shaped}:
        cols = _extract_table_columns(section)
        if not cols and _page_role(page_id, title) == "analytics":
            from app_builder.services.prd_driven_layout import _maintenance_analytics_columns

            if re.search(r"maintenance|mtbf|mttr|equipment|work order", section, re.I):
                cols = _maintenance_analytics_columns()
        if not cols:
            cols = [
                {"field": "id", "header": "ID"},
                {"field": "name", "header": "Name"},
                {"field": "status", "header": "Status"},
            ]
        shaped.append(
            {
                "type": "dataTable",
                "title": title,
                "dataKey": f"table_{page_id}",
                "columns": cols,
                "searchable": True,
                "detailView": intent.table_detail,
            }
        )

    return shaped
