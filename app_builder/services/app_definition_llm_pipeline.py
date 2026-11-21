"""End-to-end LLM AppDefinition: shell → per-page blocks → per-page data."""
from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app_builder.schemas.app_definition import AppDefinition, AppMeta, AppTheme, NavItemDef, PageDef
from app_builder.schemas.app_definition_llm import AppShellPlan, PageBlocksPlan, PageDataPlan
from app_builder.services.app_definition_generator import (
    align_shell_plan_to_screens,
    extract_screens_from_plan,
    resolve_app_display_name,
)
from app_builder.services.prd_compliance import page_gaps
from app_builder.services.prd_structure_parser import build_planning_context, section_for_page

logger = logging.getLogger("app_builder")

_SHELL_SYSTEM = """You are a principal product architect. Output ONLY valid JSON for AppShellPlan.

Derive navigation and page outlines from the user's requirement, PRD, and UI/UX docs.
- navigation.length MUST equal pages.length; every pageId must match a page id.
- First navigation path MUST be "/".
- Use the page count requested by the user; navigation.label and pages[].title MUST match the Suggested screens list exactly when provided (same spelling and order).
- Icons: MUI names (Dashboard, FactCheck, Assessment, ShowChart, People, Settings, PrecisionManufacturing).
- subtitle: one sentence describing what the page shows (from PRD).
- domain: short slug inferred from the product (e.g. manufacturing_quality), not a hardcoded list.
- appName: SHORT brand label for the sidebar — at most 2 words (e.g. "Quality Pulse", "Fleet Ops"), never the full sentence from the prompt.
"""

_BLOCKS_SYSTEM = """You are a UX architect. Output ONLY valid JSON for PageBlocksPlan.

Design the FULL page layout from the PRD section — production SaaS density, not a minimal demo.

Allowed block types:
- filterBar: { type, filters: [{ id, label, field, filterType: search|dateRange|select, optionsKey? }] }
- actionBar: { type, actions: [{ id, label, actionType: exportCsv|exportPdf|create, dataKey (required for create, references a dataTable) }] }
- kpiRow: { type, dataKey: "kpis_<pageId>", items: [{ label, valueKey, color, trendKey, targetKey }] } — include EVERY KPI listed in PRD
- lineChart / barChart: { type, title, dataKey (unique per chart), xAxisKey: "period", series: [{ dataKey, name, color }], height }
- pieChart: { type, title, dataKey, height }
- dataTable: { type, title, dataKey (unique), columns: [{ field, header, align?, variant?: "status" }], searchable: true, detailView?: true for session tables }
- text: { type, content } — avoid unless PRD needs intro copy

Rules:
- dataKey must be unique across the app; prefix with page id (e.g. table_sessions_inspections).
- Read ONLY the PRD section for THIS page. Layout must differ per page like a real SaaS product.
- Layout is driven ONLY by this page's PRD subsections (**Purpose**, **KPIs**, **Charts**, **Filters**, **Actions**, **Table columns**) — page names can be anything the user asked for.
- Include KPIs on any page where the requirement or PRD requests them. Never override a requirement based on its page name.
- Include charts for bullets under **Charts:** or **Include:** on that page (use distinct titles and series names from the bullet text).
- On Dashboard: include KPI row when **KPIs:** exist; include line/bar charts only when the page section lists chart bullets (do not add charts if none are listed).
- Include dataTable only when **Table columns:** exist or Purpose describes a searchable list/register.
- dataTable columns MUST use shape {{ "field": "snake_case_key", "header": "Human label" }} — never empty field/header; do not use title/dataKey on column objects.
- Include requested filters, create forms and CSV/PDF exports. Select/date filters require the exact dataset field name.
- Create actions must reference a dataTable dataKey; table columns should declare inputType (text|number|date|datetime-local) and required when applicable.
- For operational KPIs/charts, bind to canonical dataTable records using sourceDataKey. A KPI item/series declares aggregation: count|sum|avg|min|max|ratio, field (numeric source column except count), where (optional equality filters). ratio requires numeratorField, denominatorField and scale (e.g. 100 for percentage). Charts additionally require groupBy (source column). These bindings are calculated from source records after uploads and record changes; never guess formulas.
- Use only aggregations that correctly represent the user metric. For a metric that needs an unsupported formula, do not pretend it is source-computed; describe that limitation in a text block.
- Include widgets ONLY when the PRD subsection lists them (**KPIs**, **Charts**, **Table columns**, **Filters**, **Actions**).
- Do NOT output mock/data — blocks only.
"""

_DATA_SYSTEM = """You are a data engineer. Output ONLY valid JSON for PageDataPlan.

Generate prompt-specific synthetic operational data for ONE page's blocks. Never claim it is real factory data.

Rules:
- Keys MUST include every dataKey referenced in the blocks (KPI bundles, chart arrays, table arrays, filter option lists).
- KPI bundles: objects with string values matching labels (%, counts, currency as in PRD).
- Also output kpis_<pageId>_meta with <valueKey>_trend_pct (number -5..8) and <valueKey>_target for each KPI.
- Tables: keys MUST match column field names exactly. Honor user-provided names, dates, units, values, and row counts. Infer other synthetic values from the exact manufacturing process, not a canned domain dataset.
- When the user requests real data only or an empty app, output empty tables/charts and null KPI values until a data source is supplied.
- Charts: arrays of 14–20 objects; xAxis field must match block xAxisKey (period, category, shift, etc.).
- Every series[].dataKey in blocks must exist on each chart row with realistic numeric values.
- Match chart semantics from titles (OEE %, defects by line, shift comparison, etc.) — not generic value/target only.
- pieChart data: [{ name, value }, ...] (5–8 segments).
- filter select options: [{ value, label }, ...].
- Use realistic labels from the PRD domain (product names, lines, metrics) — infer from context, no placeholder "Asset 1".
- No duplicate keys; no commentary outside JSON.
"""


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
    user_email: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Optional[AppDefinition]:
    try:
        from llm_helper import get_llm_for_user
    except ImportError:
        logger.warning("[app_definition_llm] llm_helper unavailable")
        return None

    ctx = build_planning_context(requirement=requirement, prd=prd, uiux=uiux, plan_json=plan_json)
    llm = get_llm_for_user(user_email, model_name=model_name, temperature=0.2, streaming=False)

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

Architecture and planning contracts:
{json.dumps((plan_json or {}).get('architecture', {}), ensure_ascii=False)[:16000]}
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

    for i, outline in enumerate(shell.pages):
        if on_progress:
            await on_progress(f"LLM phase 2/3: blocks for «{outline.title}» ({i + 1}/{len(shell.pages)})…")

        section = section_for_page(ctx["prd"], outline.title, outline.id)
        if not section.strip():
            section = section_for_page(ctx["requirement"], outline.title, outline.id)
        blocks_user = f"""App: {display_name}
Page id: {outline.id}
Page title: {outline.title}
Page subtitle: {outline.subtitle}

PRD section for this page:
{section or '(use requirement + UI docs below)'}

Full requirement excerpt:
{ctx['requirement'][:8000]}

UI/UX excerpt:
{ctx['uiux'][:6000]}

Architecture:
{json.dumps((plan_json or {}).get('architecture', {}), ensure_ascii=False)[:12000]}

Existing page blocks (use these exact dataTable dataKeys and columns for cross-page source bindings):
{json.dumps([p.model_dump() for p in pages], ensure_ascii=False)[:16000]}
"""
        blocks_plan: Optional[PageBlocksPlan] = await _llm_validated(
            llm, _BLOCKS_SYSTEM, blocks_user, PageBlocksPlan, f"page_blocks_{outline.id}"
        )
        section_body = section or f"## {outline.title}\n{outline.subtitle or outline.title}"

        if blocks_plan is None or not blocks_plan.blocks:
            raise ValueError(f"LLM failed to generate blocks for {outline.id}")
        else:
            gaps = page_gaps(
                PageDef(id=outline.id, title=outline.title, subtitle=outline.subtitle, blocks=blocks_plan.blocks),
                section_body,
            )
            if (
                gaps.get("missing_kpis")
                or gaps.get("missing_columns")
                or (gaps.get("needs_charts") and not gaps.get("has_chart"))
            ):
                repair_user = f"""Fix blocks to satisfy PRD gaps on page {outline.id}.
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
                    blocks_plan = repaired
                    remaining = page_gaps(PageDef(id=outline.id, title=outline.title, blocks=repaired.blocks), section_body)
                    if remaining.get("missing_kpis") or remaining.get("missing_columns") or (remaining.get("needs_charts") and not remaining.get("has_chart")):
                        raise ValueError(f"LLM repair still misses PRD requirements for {outline.id}: {remaining}")
                else:
                    raise ValueError(f"LLM could not repair PRD gaps for {outline.id}")

        if on_progress:
            await on_progress(f"LLM phase 3/3: datasets for «{outline.title}»…")

        data_user = f"""Page: {outline.title} ({outline.id})
Blocks JSON:
{json.dumps(blocks_plan.blocks, indent=2)[:24000]}

PRD section:
{section[:12000]}

Full requirement:
{ctx['requirement']}

Previously generated datasets (preserve shared entity identities):
{json.dumps(mock, ensure_ascii=False)[:16000]}

Generate complete data for every dataKey in these blocks.
"""
        data_plan: Optional[PageDataPlan] = await _llm_validated(
            llm, _DATA_SYSTEM, data_user, PageDataPlan, f"page_data_{outline.id}"
        )
        from app_builder.services.definition_contract import validate_page_data

        last_error = "No page data generated"
        for attempt in range(3):
            try:
                if data_plan is None:
                    raise ValueError(last_error)
                validate_page_data(blocks_plan.blocks, data_plan.data)
                if set(mock) & set(data_plan.data):
                    raise ValueError("Dataset keys must be unique across pages")
                break
            except ValueError as exc:
                last_error = str(exc)
                if attempt == 2:
                    raise ValueError(f"Invalid LLM data for {outline.id}: {last_error}") from exc
                data_plan = await _llm_validated(llm, _DATA_SYSTEM, f"{data_user}\nRepair error: {last_error}", PageDataPlan, f"page_data_repair_{outline.id}")
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

    meta = AppMeta(
        appName=display_name,
        description=shell.description or ctx["requirement"][:200],
        domain=shell.domain or "generic",
        dataSource="llm_generated",
        generatedBy="llm_pipeline",
        generationPolicy="strict_llm_v1",
        requirement=requirement,
        prd=prd,
        dataModel="definition_v1",
        containsSyntheticData=True,
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
