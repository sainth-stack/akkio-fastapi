"""End-to-end LLM AppDefinition: shell → per-page blocks → per-page data."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app_builder.schemas.app_definition import AppDefinition, AppMeta, AppTheme, NavItemDef, PageDef
from app_builder.schemas.app_definition_llm import AppShellPlan, PageBlocksPlan, PageDataPlan
from app_builder.services.app_definition_compiler import _theme_from_tokens
from app_builder.services.app_definition_generator import (
    align_shell_plan_to_screens,
    extract_screens_from_plan,
    resolve_app_display_name,
)
from app_builder.services.prd_compliance import page_gaps
from app_builder.services.page_layout_intent import (
    apply_layout_intent_to_blocks,
    ensure_blocks_match_intent,
    infer_page_layout_intent,
    resolve_dashboard_page_id,
)
from app_builder.services.prd_driven_layout import blocks_for_page as prd_blocks_for_page
from app_builder.services.prd_driven_layout import ensure_minimum_blocks
from app_builder.services.prd_structure_parser import build_planning_context, section_for_page
from pydantic import ValidationError

logger = logging.getLogger("app_builder")

_VISIBLE_BLOCK_TYPES = frozenset({"kpiRow", "lineChart", "barChart", "pieChart", "dataTable"})

_SHELL_SYSTEM = """You are a principal product architect. Output ONLY valid JSON for AppShellPlan.

Derive navigation and page outlines from the user's requirement, PRD, and UI/UX docs.
- navigation.length MUST equal pages.length; every pageId must match a page id.
- First navigation path MUST be "/".
- Use 2–8 pages; navigation.label and pages[].title MUST match the Suggested screens list exactly when provided (same spelling and order).
- Icons: MUI names (Dashboard, FactCheck, Assessment, ShowChart, People, Settings, PrecisionManufacturing).
- subtitle: one sentence describing what the page shows (from PRD).
- domain: short slug inferred from the product (e.g. manufacturing_quality), not a hardcoded list.
- appName: SHORT brand label for the sidebar — at most 2 words (e.g. "Quality Pulse", "Fleet Ops"), never the full sentence from the prompt.
"""

_BLOCKS_SYSTEM = """You are a UX architect. Output ONLY valid JSON for PageBlocksPlan.

Design the FULL page layout from the PRD section — production SaaS density, not a minimal demo.

Allowed block types:
- filterBar: { type, filters: [{ id, label, filterType: search|dateRange|select, optionsKey? }] }
- actionBar: { type, actions: [{ id, label, actionType: primary|exportCsv|exportPdf|create }] }
- kpiRow: { type, dataKey: "kpis_<pageId>", items: [{ label, valueKey, color, trendKey, targetKey }] } — include EVERY KPI listed in PRD
- lineChart / barChart: { type, title, dataKey (unique per chart), xAxisKey: "period", series: [{ dataKey, name, color }], height }
- pieChart: { type, title, dataKey, height }
- dataTable: { type, title, dataKey (unique), columns: [{ field, header, align?, variant?: "status" }], searchable: true, detailView?: true for session tables }
- text: { type, content } — avoid unless PRD needs intro copy

Rules:
- dataKey must be unique across the app; prefix with page id (e.g. table_sessions_inspections).
- Read ONLY the PRD section for THIS page. Layout must differ per page like a real SaaS product.
- Layout is driven ONLY by this page's PRD subsections (**Purpose**, **KPIs**, **Charts**, **Filters**, **Actions**, **Table columns**) — page names can be anything the user asked for.
- kpiRow ONLY on the Dashboard page (route id/title dashboard or home). NEVER on Analytics, Reports, Sessions, or any other page — even if PRD lists KPIs there.
- Include charts for bullets under **Charts:** or **Include:** on that page (use distinct titles and series names from the bullet text).
- On Dashboard: include KPI row when **KPIs:** exist; include line/bar charts only when the page section lists chart bullets (do not add charts if none are listed).
- Include dataTable ONLY when **Table columns:** explicitly lists columns in the PRD section above. Column field names and headers MUST be taken verbatim from those bullets.
- CRITICAL — table columns must match the user's business domain. For inventory: use sku, product_name, warehouse, quantity_on_hand, reorder_point, unit_cost, status. For vendor quality: use vendor, inspection_lot, defect_rate_pct, yield_pct, status. For procurement: use po_number, vendor, order_value, lead_time_days, status. For supply chain: use po_number, supplier, item, quantity, delivery_date, status. NEVER default to work_order_id, machine, production_line, mtbf_hours, mttr_hours, downtime_hours, root_cause unless the prompt is explicitly about plant maintenance.
- dataTable columns MUST use shape {{ "field": "snake_case_key", "header": "Human label" }} — never empty field/header; do not use title/dataKey on column objects.
- Include filterBar only when **Filters:** lists filters. Do NOT add actionBar for export, download, PDF, or CSV — the UI is view-only.
- actionBar only for explicit in-app **create** flows (rare); never exportCsv/exportPdf.
- Include widgets ONLY when the PRD subsection lists them (**KPIs**, **Charts**, **Table columns**, **Filters**, **Actions**).
- Do NOT output mock/data — blocks only.
"""

_DATA_SYSTEM = """You are a data engineer. Output ONLY valid JSON for PageDataPlan.

Generate realistic operational sample data for ONE page. Every name, metric, and number must come from THIS app's requirement and PRD — not from a generic factory template.

Rules:
- Keys MUST include every dataKey referenced in the blocks (KPI bundles, chart arrays, table arrays, filter option lists).
- KPI bundles: objects whose values match the KPI labels in the prompt (currency, %, days, counts). Do not invent MTBF, MTTR, OEE, or work orders unless the prompt asks for maintenance.
- Also output kpis_<pageId>_meta with <valueKey>_trend_pct (number -5..8) and <valueKey>_target for each KPI.
- Tables: arrays of 12–24 objects. Keys MUST match column field names exactly (never col1, col2, or field_0).
- Row text must look like a real business in the prompt: vendor names, SKUs, warehouses, purchase orders, inspection lots, or suppliers — whichever the prompt is about.
- Do not reuse a default plant sheet (Line A, Press 1, CNC Mill, Bearing wear, WO-24000) unless the prompt is about shop-floor maintenance.
- Charts: arrays of 8–16 objects; xAxis field must match block xAxisKey. Series numbers must be plausible for those KPI labels.
- pieChart data: [{ name, value }, ...] (4–7 segments) named for the prompt's categories, not "Segment 1".
- filter select options: [{ value, label }, ...] using the same categories as the table.
- No duplicate keys; no commentary outside JSON.
"""


def _coerce_page_blocks_plan(blocks: List[Dict[str, Any]], *, page_id: str) -> PageBlocksPlan:
    """Build PageBlocksPlan without failing codegen when the LLM returns sparse JSON."""
    try:
        return PageBlocksPlan(blocks=blocks)
    except ValidationError as exc:
        logger.warning("[app_definition_llm] PageBlocksPlan validation failed for %s: %s", page_id, exc)
        padded = list(blocks or [])
        while len(padded) < 2:
            padded.append({"type": "text", "content": "", "hidden": True})
        return PageBlocksPlan(blocks=padded[:24])


def _page_has_visible_widgets(blocks: List[Dict[str, Any]]) -> int:
    return sum(
        1
        for b in blocks or []
        if isinstance(b, dict) and str(b.get("type") or "") in _VISIBLE_BLOCK_TYPES
    )


async def _llm_validated(llm: Any, system: str, user: str, schema: type, label: str) -> Optional[Any]:
    from app_builder.agents.structured_output import extract_validated

    return await extract_validated(
        llm=llm,
        system_prompt=system,
        user_prompt=user,
        schema=schema,
        max_retries=2,
        context_label=label,
    )


async def generate_app_definition_llm(
    *,
    requirement: str = "",
    prd: str = "",
    uiux: str = "",
    plan_json: Optional[Dict[str, Any]] = None,
    app_name: Optional[str] = None,
    on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
) -> Optional[AppDefinition]:
    try:
        from llm_helper import get_llm_for_user
    except ImportError:
        logger.warning("[app_definition_llm] llm_helper unavailable")
        return None

    ctx = build_planning_context(requirement=requirement, prd=prd, uiux=uiux, plan_json=plan_json)
    llm = get_llm_for_user(None, temperature=0.2, streaming=False)

    screens = extract_screens_from_plan(
        requirement=requirement, prd=prd, uiux=uiux, plan_json=plan_json or {},
    )
    screen_hint = json.dumps(screens, indent=2) if screens else "[]"

    if on_progress:
        await on_progress("LLM phase 1/3: app shell (navigation + page outlines)…")

    shell_user = f"""Requirement:
{ctx['requirement']}

PRD:
{ctx['prd'][:32000]}

UI/UX:
{ctx['uiux'][:16000]}

Suggested screens:
{screen_hint}
"""
    shell: Optional[AppShellPlan] = await _llm_validated(llm, _SHELL_SYSTEM, shell_user, AppShellPlan, "app_shell")
    if shell is None:
        return None
    if screens:
        shell = align_shell_plan_to_screens(shell, screens)

    from app_builder.services.app_theme_generator import (
        ensure_theme_variation_seed,
        generate_app_theme,
        merge_design_tokens_into_theme,
    )

    plan = plan_json or {}
    variation_seed = ensure_theme_variation_seed(plan)
    plan["theme_variation_seed"] = variation_seed
    tokens = plan.get("design_tokens") if isinstance(plan.get("design_tokens"), dict) else None
    display_name_early = resolve_app_display_name(
        requirement=requirement,
        prd=prd,
        plan_json=plan_json,
        app_name=app_name,
    )
    theme = generate_app_theme(
        app_name=display_name_early,
        domain=str(plan.get("blueprint", {}).get("domain", "") if isinstance(plan.get("blueprint"), dict) else "generic"),
        requirement=requirement,
        variation_seed=variation_seed,
    )
    theme = merge_design_tokens_into_theme(theme, tokens, preserve_generated_colors=True)
    display_name = resolve_app_display_name(
        requirement=requirement,
        prd=prd,
        plan_json=plan_json,
        app_name=app_name or shell.appName,
    )

    mock: Dict[str, Any] = {}
    pages: List[PageDef] = []
    navigation = [
        NavItemDef(label=n.label, path=n.path, icon=n.icon, pageId=n.pageId) for n in shell.navigation
    ]

    dashboard_page_id = resolve_dashboard_page_id(
        [(o.id, o.title) for o in shell.pages],
        [n.pageId for n in shell.navigation],
    )

    for i, outline in enumerate(shell.pages):
        if on_progress:
            await on_progress(f"LLM phase 2/3: blocks for «{outline.title}» ({i + 1}/{len(shell.pages)})…")

        section = section_for_page(ctx["prd"], outline.title, outline.id)
        if not section.strip():
            section = section_for_page(ctx["requirement"], outline.title, outline.id)
        section_body_preview = section or f"## {outline.title}\n{outline.subtitle or outline.title}"
        page_intent_hint = infer_page_layout_intent(
            section_body_preview,
            outline.title,
            page_id=outline.id,
            dashboard_page_id=dashboard_page_id,
        )

        _maintenance_domain = bool(re.search(
            r"maintenance|\bmtbf\b|\bmttr\b|work order|downtime|\boee\b",
            f"{ctx['requirement']} {section}",
            re.I,
        ))
        _domain_guard = (
            "This is a MAINTENANCE app — work_order_id, machine, downtime_hours, mtbf_hours, mttr_hours columns are allowed."
            if _maintenance_domain
            else
            "This is NOT a maintenance app. Do NOT use work_order_id, machine, production_line, equipment, root_cause, downtime_hours, mtbf_hours, mttr_hours, or any maintenance/plant column. Table columns must match the entities in the user's requirement (e.g. SKU, vendor, supplier, purchase order, inspection lot — whatever the prompt describes)."
        )
        blocks_user = f"""App: {display_name}
Domain guard: {_domain_guard}

Page id: {outline.id}
Page title: {outline.title}
Page subtitle: {outline.subtitle}

Layout intent (from PRD — follow this; do not duplicate other pages):
- include KPI row: {page_intent_hint.wants_kpis} (cap {page_intent_hint.kpi_cap or 0})
- include charts: {page_intent_hint.wants_charts} (cap {page_intent_hint.chart_cap or 0})
- include table: {page_intent_hint.wants_table} (detail rows: {page_intent_hint.table_detail})
- include filters: {page_intent_hint.wants_filters}
- include export/create actions: {page_intent_hint.wants_export_actions or page_intent_hint.wants_create_action}

PRD section for this page (table column names and KPI names MUST come from here):
{section or '(use requirement below)'}

Full requirement (source of truth for all labels and column names):
{ctx['requirement'][:8000]}

UI/UX excerpt:
{ctx['uiux'][:6000]}
"""
        blocks_plan: Optional[PageBlocksPlan] = await _llm_validated(
            llm, _BLOCKS_SYSTEM, blocks_user, PageBlocksPlan, f"page_blocks_{outline.id}"
        )
        section_body = section or f"## {outline.title}\n{outline.subtitle or outline.title}"

        page_intent = infer_page_layout_intent(
            section_body,
            outline.title,
            page_id=outline.id,
            dashboard_page_id=dashboard_page_id,
        )

        def _finalize_blocks(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            shaped = apply_layout_intent_to_blocks(
                raw, page_intent, page_id=outline.id, title=outline.title
            )
            shaped = ensure_blocks_match_intent(
                shaped,
                page_intent,
                page_id=outline.id,
                title=outline.title,
                section=section_body,
                dashboard_page_id=dashboard_page_id,
            )
            return ensure_minimum_blocks(
                shaped,
                outline.id,
                outline.title,
                section=section_body,
                dashboard_page_id=dashboard_page_id,
            )

        def _plan_from_prd() -> PageBlocksPlan:
            return _coerce_page_blocks_plan(
                _finalize_blocks(
                    prd_blocks_for_page(
                        outline.id,
                        outline.title,
                        section_body,
                        dashboard_page_id=dashboard_page_id,
                    )
                ),
                page_id=outline.id,
            )

        try:
            if blocks_plan is None or not blocks_plan.blocks:
                blocks_plan = _plan_from_prd()
            else:
                blocks_plan = _coerce_page_blocks_plan(
                    _finalize_blocks(blocks_plan.blocks),
                    page_id=outline.id,
                )
                gaps = page_gaps(
                    PageDef(
                        id=outline.id,
                        title=outline.title,
                        subtitle=outline.subtitle,
                        blocks=blocks_plan.blocks,
                    ),
                    section_body,
                )
                if (
                    gaps.get("missing_kpis")
                    or gaps.get("missing_columns")
                    or (page_intent.wants_charts and gaps.get("needs_charts") and not gaps.get("has_chart"))
                ):
                    repair_user = f"""Fix blocks to satisfy PRD gaps on page {outline.id}.
Domain guard: {_domain_guard}

Layout intent (do not exceed — page must look different from other routes):
- KPI row: {page_intent.wants_kpis} (max {page_intent.kpi_cap or 0} items)
- Charts: {page_intent.wants_charts} (max {page_intent.chart_cap or 0})
- Table: {page_intent.wants_table}
- Filters: {page_intent.wants_filters}
- Export/create: {page_intent.wants_export_actions or page_intent.wants_create_action}

Current blocks:
{json.dumps(blocks_plan.blocks, indent=2)[:12000]}

Gaps:
{json.dumps(gaps, indent=2)}

PRD:
{section_body[:14000]}
"""
                    repaired = await _llm_validated(
                        llm, _BLOCKS_SYSTEM, repair_user, PageBlocksPlan, f"page_blocks_repair_{outline.id}"
                    )
                    if repaired and repaired.blocks:
                        blocks_plan = _coerce_page_blocks_plan(
                            _finalize_blocks(repaired.blocks),
                            page_id=outline.id,
                        )
                    else:
                        blocks_plan = _plan_from_prd()

            if _page_has_visible_widgets(blocks_plan.blocks) < 1:
                logger.warning(
                    "[app_definition_llm] page %s had no visible widgets after finalize — PRD fallback",
                    outline.id,
                )
                blocks_plan = _plan_from_prd()
        except Exception as exc:
            logger.warning(
                "[app_definition_llm] blocks phase failed for %s (%s) — PRD fallback",
                outline.id,
                exc,
            )
            blocks_plan = _plan_from_prd()

        if on_progress:
            await on_progress(f"LLM phase 3/3: datasets for «{outline.title}»…")

        data_user = f"""Page: {outline.title} ({outline.id})
Domain guard: {_domain_guard}

User requirement — every label, number, and row must come from THIS business:
{ctx['requirement'][:8000]}

Blocks JSON (generate data for every dataKey here):
{json.dumps(blocks_plan.blocks, indent=2)[:24000]}

PRD section:
{section[:12000]}

Rules:
- KPI string values and table rows must use the nouns in the requirement above.
- If the domain guard says NOT maintenance, then table rows must NOT contain work_order_id, machine, CNC Mill, Conveyor A, Hydraulic Press, WO-24000, downtime_hours, mtbf_hours, or mttr_hours.
- Use realistic names for the actual domain: SKU codes, vendor names, warehouse codes, inspection lot IDs, supplier names, purchase order numbers — whatever the requirement describes.
"""
        data_plan: Optional[PageDataPlan] = await _llm_validated(
            llm, _DATA_SYSTEM, data_user, PageDataPlan, f"page_data_{outline.id}"
        )
        if data_plan and isinstance(data_plan.data, dict):
            mock.update(data_plan.data)

        pages.append(
            PageDef(
                id=outline.id,
                title=outline.title,
                subtitle=outline.subtitle,
                blocks=blocks_plan.blocks,
            )
        )

    if not pages:
        return None

    # Post-generation guard: strip maintenance columns from non-maintenance apps.
    _is_maint_app = bool(re.search(
        r"maintenance|\bmtbf\b|\bmttr\b|work order|downtime|\boee\b",
        requirement or "",
        re.I,
    ))
    if not _is_maint_app:
        _MAINT_FIELDS = {
            "work_order_id", "machine", "production_line", "equipment",
            "event_type", "root_cause", "downtime_hours", "mtbf_hours", "mttr_hours",
        }
        cleaned_pages: List[PageDef] = []
        for page in pages:
            new_blocks: List[Any] = []
            for block in page.blocks:
                if isinstance(block, dict) and block.get("type") == "dataTable":
                    cols = block.get("columns") or []
                    block_fields = {str(c.get("field") or "") for c in cols if isinstance(c, dict)}
                    if block_fields and block_fields.issubset(_MAINT_FIELDS | {"status", "severity", "shift", "occurred_at"}):
                        # This table is entirely maintenance columns — skip it for non-maintenance apps
                        continue
                    # Keep the table but drop individual maintenance columns
                    cleaned_cols = [
                        c for c in cols
                        if isinstance(c, dict) and str(c.get("field") or "") not in _MAINT_FIELDS
                    ]
                    if not cleaned_cols:
                        continue
                    new_blocks.append({**block, "columns": cleaned_cols})
                else:
                    new_blocks.append(block)
            cleaned_pages.append(PageDef(
                id=page.id, title=page.title, subtitle=page.subtitle, blocks=new_blocks
            ))
        pages = cleaned_pages
        # Also drop maintenance keys from mock
        for mf in _MAINT_FIELDS:
            mock.pop(mf, None)
        for key in list(mock):
            if isinstance(mock.get(key), list) and mock[key]:
                first = mock[key][0]
                if isinstance(first, dict) and set(first.keys()).issubset(
                    _MAINT_FIELDS | {"status", "severity", "shift", "occurred_at"}
                ):
                    mock.pop(key, None)

    meta = AppMeta(
        appName=display_name,
        description=shell.description or ctx["requirement"][:200],
        domain=shell.domain or "generic",
        dataSource="static",
        generatedBy="llm_pipeline",
    )
    # extra meta fields via model — add generatedBy on dump
    defn = AppDefinition(
        meta=meta,
        theme=theme,
        navigation=navigation[: len(pages)],
        pages=pages,
        mock=mock,
    )
    defn = _finalize_llm_definition(
        defn,
        prd=prd,
        requirement=requirement,
        plan_json=plan_json,
    )
    from app_builder.services.json_app_generator import log_generated_app_schema

    log_generated_app_schema(defn)
    logger.info(
        "[app_definition_llm] done | pages=%d blocks=%d data_keys=%d",
        len(defn.pages),
        sum(len(p.blocks) for p in defn.pages),
        len(defn.mock or {}),
    )
    return defn


def _finalize_llm_definition(
    defn: AppDefinition,
    *,
    prd: str = "",
    requirement: str = "",
    plan_json: Optional[Dict[str, Any]] = None,
) -> AppDefinition:
    from app_builder.services.app_definition_generator import finalize_app_definition

    defn.meta.generatedBy = defn.meta.generatedBy or "llm_pipeline"
    plan = plan_json or {}
    from app_builder.services.app_theme_generator import ensure_theme_variation_seed

    tokens = plan.get("design_tokens") if isinstance(plan.get("design_tokens"), dict) else None
    seed = str(plan.get("theme_variation_seed") or "") or ensure_theme_variation_seed(plan)
    return finalize_app_definition(
        defn,
        prd=prd,
        requirement=requirement,
        design_tokens=tokens,
        theme_variation_seed=seed,
    )
