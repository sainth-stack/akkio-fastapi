"""Build AppDefinition from user requirement + plan artifacts (generic, not fixed templates)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app_builder.schemas.app_definition import (
    AppDefinition,
    AppMeta,
    AppTheme,
    NavItemDef,
    PageDef,
)
from app_builder.services.app_definition_compiler import _theme_from_tokens
from app_builder.services.block_normalizer import normalize_app_definition_blocks
from app_builder.services.mvp_mock_enricher import enrich_mvp_mock

logger = logging.getLogger("app_builder")

_ICON_HINTS = (
    (("dashboard", "overview", "home"), "Dashboard"),
    (("analytic", "insight", "metric", "chart"), "ShowChart"),
    (("report", "export"), "Assessment"),
    (("alert", "notification"), "NotificationsActive"),
    (("order", "purchase", "cart"), "ShoppingCart"),
    (("supplier", "vendor", "shipping"), "LocalShipping"),
    (("inventory", "material", "stock", "sku", "warehouse"), "Inventory"),
    (("user", "customer", "contact", "team"), "People"),
    (("setting", "config", "admin"), "Settings"),
    (("quality", "inspection"), "FactCheck"),
    (("machine", "equipment", "plant", "production"), "PrecisionManufacturing"),
)


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return s or "page"


def _icon_for_label(label: str) -> str:
    low = (label or "").lower()
    for keys, icon in _ICON_HINTS:
        if any(k in low for k in keys):
            return icon
    return "Article"


def _title_from_requirement(requirement: str, fallback: str = "Workspace") -> str:
    line = (requirement or "").strip().split("\n")[0].strip()
    line = re.sub(r"^(build|create|generate|make)\s+(a|an|the|product)?\s*", "", line, flags=re.I).strip()
    line = re.sub(r"\s+app(lication)?\s*$", "", line, flags=re.I).strip()
    if 4 < len(line) < 100:
        return line[:90].title() if line.islower() or "monitoring" in line.lower() else line[:90]
    return fallback


def _screens_from_requirement_phrase(text: str) -> List[Dict[str, str]]:
    """Parse '… with dashboard, analytics and reports' style lists."""
    if not text:
        return []
    low = text.lower()
    for pattern in (
        r"(?:with|including|screens?)\s+([^.;\n]+)",
        r"(?:pages?)\s*:\s*([^.;\n]+)",
    ):
        m = re.search(pattern, low, re.I)
        if not m:
            continue
        chunk = m.group(1)
        parts = re.split(r",|;|/|\band\b", chunk)
        out: List[Dict[str, str]] = []
        for p in parts:
            name = re.sub(r"^\d+[\).\s]+", "", p.strip())
            name = re.sub(r"\s+screen(s)?$", "", name, flags=re.I).strip()
            if 2 < len(name) < 48:
                title = name[0].upper() + name[1:]
                out.append({"id": _slug(title), "title": title, "purpose": title})
        if out:
            return out
    return []


def _screens_from_prd_dict(prd: Any) -> List[Dict[str, str]]:
    if not isinstance(prd, dict):
        return []
    screens = prd.get("screens") or []
    out: List[Dict[str, str]] = []
    for s in screens:
        if isinstance(s, dict) and s.get("name"):
            title = str(s["name"])
            out.append({
                "id": _slug(s.get("id") or title),
                "title": title,
                "purpose": str(s.get("purpose") or title),
            })
    return out


def _screens_from_architecture(arch: Any) -> List[Dict[str, str]]:
    if not isinstance(arch, dict):
        return []
    out: List[Dict[str, str]] = []
    for s in arch.get("screens") or []:
        if isinstance(s, dict) and s.get("name"):
            title = str(s["name"])
            out.append({
                "id": _slug(title),
                "title": title,
                "purpose": str(s.get("description") or s.get("purpose") or title),
            })
        elif isinstance(s, str) and s.strip():
            title = s.strip()
            out.append({"id": _slug(title), "title": title, "purpose": title})
    fs = arch.get("frontend_structure") or {}
    for p in fs.get("pages") or []:
        if isinstance(p, dict):
            title = str(p.get("name") or p.get("component") or "").replace("Page", "").strip()
            if title:
                out.append({"id": _slug(title), "title": title, "purpose": title})
    return out


def _screens_from_blueprint_nav(blueprint: Any) -> List[Dict[str, str]]:
    if not isinstance(blueprint, dict):
        return []
    nav = blueprint.get("nav_items") or []
    out: List[Dict[str, str]] = []
    for i, item in enumerate(nav):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or f"Page {i + 1}")
        out.append({"id": _slug(label) if item.get("path") != "/" else "dashboard", "title": label, "purpose": label})
    return out


def _dedupe_screens(screens: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set = set()
    out: List[Dict[str, str]] = []
    for s in screens:
        sid = s.get("id") or _slug(s.get("title", "page"))
        if sid in seen:
            continue
        seen.add(sid)
        out.append({**s, "id": sid})
    return out


def extract_screens_from_plan(
    *,
    requirement: str = "",
    prd: str = "",
    uiux: str = "",
    plan_json: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    plan_json = plan_json or {}
    collected: List[Dict[str, str]] = []

    collected.extend(_screens_from_prd_dict(plan_json.get("prd")))
    collected.extend(_screens_from_architecture(plan_json.get("architecture")))
    bp = plan_json.get("blueprint")
    if isinstance(bp, dict):
        collected.extend(_screens_from_blueprint_nav(bp))

    combined = "\n".join([requirement, prd, uiux])
    collected.extend(_screens_from_requirement_phrase(combined))

    # Markdown PRD headings: "## Dashboard" or "Screen: Contacts"
    for m in re.finditer(r"(?m)^#+\s*(.+)$", prd or ""):
        t = m.group(1).strip()
        if 2 < len(t) < 60 and not t.lower().startswith(("prd", "overview", "goal", "feature")):
            collected.append({"id": _slug(t), "title": t, "purpose": t})

    if not collected:
        collected.extend(_screens_from_requirement_phrase(requirement))

    screens = _dedupe_screens(collected)
    if screens and screens[0]["id"] != "dashboard" and "dashboard" in screens[0]["title"].lower():
        screens[0]["id"] = "dashboard"
    return screens


def resolve_app_display_name(
    *,
    requirement: str = "",
    prd: str = "",
    plan_json: Optional[Dict[str, Any]] = None,
    app_name: Optional[str] = None,
) -> str:
    """Human-readable product name for meta.appName and shell header."""
    if app_name and not re.match(r"^(fullstack_app_|generated_app_|Workspace)$", app_name):
        if not app_name.startswith("fullstack_app_") and not app_name.startswith("generated_app_"):
            return app_name[:80]

    plan_json = plan_json or {}
    blueprint = plan_json.get("blueprint") if isinstance(plan_json.get("blueprint"), dict) else {}
    if isinstance(blueprint, dict) and blueprint.get("app_name"):
        return str(blueprint["app_name"])[:80]

    prd_obj = plan_json.get("prd")
    if isinstance(prd_obj, dict):
        for key in ("app_name", "product_name", "title", "name"):
            if prd_obj.get(key):
                return str(prd_obj[key])[:80]

    try:
        from app_builder.services.fullstack_app_generator import extract_app_title

        title = extract_app_title(requirement, prd)
        if title and title != "Enterprise Application":
            return title[:80]
    except ImportError:
        pass

    title = _title_from_requirement(requirement, "My Application")
    return title[:80]


def _infer_blocks_for_screen(screen: Dict[str, str], index: int, total: int) -> List[Dict[str, Any]]:
    title = (screen.get("title") or "").lower()
    purpose = (screen.get("purpose") or "").lower()
    ctx = f"{title} {purpose}"
    sid = screen.get("id") or "page"

    is_dashboard = index == 0 or "dashboard" in ctx or "overview" in ctx
    is_analytics = "analytic" in ctx or "insight" in ctx or "metric" in ctx
    is_reports = "report" in ctx
    is_table = any(k in ctx for k in ("list", "table", "catalog", "inventory", "order", "user", "contact", "ticket"))

    blocks: List[Dict[str, Any]] = []

    if is_dashboard:
        blocks.append({
            "type": "kpiRow",
            "items": [
                {"label": "OEE", "valueKey": "oee", "color": "primary"},
                {"label": "Throughput", "valueKey": "throughput", "color": "success"},
                {"label": "Downtime (today)", "valueKey": "downtime", "color": "warning"},
                {"label": "Active alerts", "valueKey": "active_alerts", "color": "error"},
            ],
        })
        blocks.append({
            "type": "lineChart",
            "title": f"{screen.get('title', 'Operations')} — OEE trend",
            "dataKey": f"series_{sid}_trend",
            "xAxisKey": "label",
            "series": [
                {"dataKey": "oee", "name": "OEE %", "color": "#1976d2"},
                {"dataKey": "target", "name": "Target", "color": "#94a3b8"},
            ],
            "height": 300,
        })
        return blocks

    if is_analytics:
        blocks.append({
            "type": "kpiRow",
            "items": [
                {"label": "MTBF", "valueKey": "mtbf", "color": "primary"},
                {"label": "First-pass yield", "valueKey": "first_pass_yield", "color": "success"},
                {"label": "Scrap rate", "valueKey": "scrap_rate", "color": "warning"},
            ],
        })
        blocks.append({
            "type": "barChart",
            "title": f"{screen.get('title', 'Analytics')} comparison",
            "dataKey": f"series_{sid}_bars",
            "xAxisKey": "label",
            "series": [
                {"dataKey": "output", "name": "Output (units)", "color": "#1976d2"},
                {"dataKey": "defects", "name": "Defects", "color": "#d32f2f"},
            ],
            "height": 300,
        })
        blocks.append({
            "type": "dataTable",
            "title": "Machine performance",
            "dataKey": f"table_{sid}",
            "columns": [
                {"field": "id", "header": "ID"},
                {"field": "name", "header": "Machine"},
                {"field": "value", "header": "OEE %", "align": "right"},
                {"field": "status", "header": "Status"},
            ],
            "searchable": True,
        })
        return blocks

    if is_reports:
        blocks.append({
            "type": "kpiRow",
            "items": [
                {"label": "Reports (30d)", "valueKey": "reports_count", "color": "primary"},
                {"label": "On-time delivery", "valueKey": "on_time_delivery", "color": "success"},
            ],
        })
        blocks.append({
            "type": "dataTable",
            "title": "Report library",
            "dataKey": f"table_{sid}",
            "columns": [
                {"field": "name", "header": "Report"},
                {"field": "period", "header": "Period"},
                {"field": "owner", "header": "Owner"},
                {"field": "status", "header": "Status"},
            ],
            "searchable": True,
        })
        return blocks

    if is_table or index > 0:
        entity = screen.get("title") or "Records"
        blocks.append({
            "type": "dataTable",
            "title": entity,
            "dataKey": f"table_{sid}",
            "columns": [
                {"field": "id", "header": "ID"},
                {"field": "name", "header": "Name"},
                {"field": "status", "header": "Status"},
                {"field": "updated", "header": "Updated"},
            ],
            "searchable": True,
        })
        return blocks

    # Default middle page
    blocks.append({
        "type": "text",
        "content": screen.get("purpose") or screen.get("title") or "",
    })
    return blocks


def _collect_mock_keys(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    mock: Dict[str, Any] = {}

    def ensure_kpi(key: str) -> None:
        if key in mock:
            return
        mock[key] = "—"

    def ensure_series(key: str, kind: str) -> None:
        if key in mock:
            return
        if kind == "line":
            mock[key] = [
                {"label": f"W{i+1}", "value": 70 + i * 3, "target": 85}
                for i in range(6)
            ]
        else:
            mock[key] = [
                {"label": f"W{i+1}", "actual": 40 + i * 5, "benchmark": 50 + i * 2}
                for i in range(6)
            ]

    def ensure_table(key: str, columns: List[Dict[str, Any]]) -> None:
        if key in mock:
            return
        fields = [c.get("field") for c in columns if c.get("field")]
        if not fields:
            fields = ["id", "name", "status"]
        rows = []
        for i in range(4):
            row: Dict[str, Any] = {}
            for f in fields:
                if f == "id":
                    row[f] = f"ROW-{100 + i}"
                elif f in ("value", "lines"):
                    row[f] = (i + 1) * 1000
                elif f == "status":
                    row[f] = ["Active", "Pending", "Complete", "Draft"][i % 4]
                elif f == "updated" or f == "when":
                    row[f] = f"{i + 1}d ago"
                else:
                    row[f] = f"Sample {f} {i + 1}"
            rows.append(row)
        mock[key] = rows

    for block in blocks:
        btype = block.get("type")
        if btype == "kpiRow":
            for item in block.get("items") or []:
                if isinstance(item, dict) and item.get("valueKey"):
                    ensure_kpi(str(item["valueKey"]))
        elif btype in ("lineChart", "barChart"):
            dk = block.get("dataKey")
            if dk:
                ensure_series(str(dk), "line" if btype == "lineChart" else "bar")
        elif btype == "dataTable":
            dk = block.get("dataKey")
            cols = block.get("columns") or []
            if dk:
                ensure_table(str(dk), cols if isinstance(cols, list) else [])

    # Fill KPI keys with plausible demo values
    for k, v in list(mock.items()):
        if v == "—":
            if "rate" in k or "pct" in k:
                mock[k] = "94.2%"
            elif "count" in k or "open" in k or k.endswith("_d"):
                mock[k] = 12
            else:
                mock[k] = "1,240"

    return mock


def build_app_definition_from_screens(
    screens: List[Dict[str, str]],
    *,
    app_name: str,
    description: str,
    domain: str,
    theme: Optional[AppTheme] = None,
) -> AppDefinition:
    if not screens:
        screens = [
            {"id": "dashboard", "title": "Dashboard", "purpose": "Overview"},
            {"id": "details", "title": "Details", "purpose": "Records"},
        ]

    navigation: List[NavItemDef] = []
    pages: List[PageDef] = []
    mock: Dict[str, Any] = {}

    for i, screen in enumerate(screens):
        sid = screen["id"]
        path = "/" if i == 0 else f"/{sid.replace('_', '-')}"
        label = screen.get("title") or sid
        navigation.append(
            NavItemDef(
                label=label,
                path=path,
                icon=_icon_for_label(label),
                pageId=sid if i > 0 or sid == "dashboard" else (sid if path == "/" else "dashboard"),
            )
        )
        if i == 0 and path == "/":
            page_id = sid if sid != "page" else "dashboard"
            if page_id != navigation[-1].pageId:
                navigation[-1].pageId = page_id
        else:
            page_id = sid
            navigation[-1].pageId = page_id

        blocks = _infer_blocks_for_screen(screen, i, len(screens))
        mock.update(_collect_mock_keys(blocks))
        pages.append(
            PageDef(
                id=page_id,
                title=label,
                subtitle=str(screen.get("purpose") or ""),
                blocks=blocks,
            )
        )

    return AppDefinition(
        meta=AppMeta(appName=app_name, description=description, domain=domain or "generic"),
        theme=theme or AppTheme(),
        navigation=navigation,
        pages=pages,
        mock=mock,
    )


def build_app_definition_from_plan(
    *,
    requirement: str = "",
    prd: str = "",
    uiux: str = "",
    plan_json: Optional[Dict[str, Any]] = None,
    app_name: Optional[str] = None,
) -> AppDefinition:
    plan_json = plan_json or {}
    blueprint = plan_json.get("blueprint") if isinstance(plan_json.get("blueprint"), dict) else {}
    domain = str(
        blueprint.get("selected_domain")
        or blueprint.get("domain")
        or "generic"
    ).lower()

    name = resolve_app_display_name(
        requirement=requirement,
        prd=prd,
        plan_json=plan_json,
        app_name=app_name,
    )

    screens = extract_screens_from_plan(
        requirement=requirement,
        prd=prd,
        uiux=uiux,
        plan_json=plan_json,
    )
    desc = (requirement or "").strip().split("\n")[0][:200] or "Generated from your requirement"
    theme = _theme_from_tokens(plan_json.get("design_tokens") if isinstance(plan_json.get("design_tokens"), dict) else None)

    return build_app_definition_from_screens(
        screens,
        app_name=name,
        description=desc,
        domain=domain,
        theme=theme,
    )


def hydrate_plan_json(
    plan_json: Optional[Dict[str, Any]],
    *,
    architecture: Optional[Any] = None,
    design_tokens: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge codegen request fields into plan_json for screen extraction + LLM context."""
    plan = dict(plan_json or {})
    if architecture and not plan.get("architecture"):
        if isinstance(architecture, dict):
            plan["architecture"] = architecture
    if design_tokens and not plan.get("design_tokens"):
        plan["design_tokens"] = design_tokens
    return plan


def finalize_app_definition(defn: AppDefinition) -> AppDefinition:
    """Ensure mock data exists for every block reference; align nav pageIds."""
    mock = dict(defn.mock or {})
    for page in defn.pages:
        mock.update(_collect_mock_keys(page.blocks))
    defn.mock = mock
    defn = normalize_app_definition_blocks(defn)
    defn = enrich_mvp_mock(defn)
    mock = dict(defn.mock or {})
    page_ids = {p.id for p in defn.pages}
    navigation: List[NavItemDef] = []
    for i, nav in enumerate(defn.navigation):
        page_id = nav.pageId if nav.pageId in page_ids else (defn.pages[i].id if i < len(defn.pages) else nav.pageId)
        navigation.append(
            NavItemDef(label=nav.label, path=nav.path, icon=nav.icon, pageId=page_id)
        )
    defn.navigation = navigation
    defn.mock = mock
    return defn


def _gen_to_app_definition(gen: Any, theme: AppTheme) -> AppDefinition:
    from app_builder.schemas.app_definition_gen import AppDefinitionGen

    if not isinstance(gen, AppDefinitionGen):
        gen = AppDefinitionGen.model_validate(gen)

    navigation = [
        NavItemDef(label=n.label, path=n.path, icon=n.icon, pageId=n.pageId)
        for n in gen.navigation
    ]
    pages = [
        PageDef(id=p.id, title=p.title, subtitle=p.subtitle, blocks=p.blocks)
        for p in gen.pages
    ]
    mock = dict(gen.mock or {})
    for page in pages:
        mock.update(_collect_mock_keys(page.blocks))

    return finalize_app_definition(
        AppDefinition(
            meta=AppMeta(appName=gen.appName, description=gen.description, domain=gen.domain),
            theme=theme,
            navigation=navigation,
            pages=pages,
            mock=mock,
        )
    )


_APP_DEF_LLM_SYSTEM = """You are an app UX architect. Output ONLY JSON for AppDefinitionGen.

Design a multi-page app that matches the user's requirement and planning docs.
Rules:
- navigation.length must equal pages.length; pageId must match page id.
- First nav item path should be "/".
- Use block types: kpiRow, lineChart, barChart, dataTable, text only.
- Every dataKey in blocks must have matching entries in mock (arrays for charts/tables, strings/numbers for KPI valueKeys).
- KPI rows: use semantic valueKeys (oee, throughput) and store values in mock["kpis_<pageId>"] = { "oee": "87.4%", ... } plus scalars if needed.
- Mock data must look like a real MVP product (realistic names, numbers, 5+ table rows, 6+ chart points).
- Charts need series[].dataKey matching keys in mock row objects.
- 2–8 pages typical; labels must reflect the user's domain (not generic inventory unless they asked).
- Icons: MUI icon names like Dashboard, ShowChart, Assessment, People, Inventory, Settings.

Schema:
{
  "appName": string,
  "description": string,
  "domain": string,
  "navigation": [{"label", "path", "icon", "pageId"}],
  "pages": [{"id", "title", "subtitle", "blocks": [...]}],
  "mock": { ... }
}
"""


async def generate_app_definition_with_llm(
    *,
    requirement: str,
    prd: str = "",
    uiux: str = "",
    plan_json: Optional[Dict[str, Any]] = None,
) -> Optional[AppDefinition]:
    try:
        from llm_helper import get_llm_for_user
        from app_builder.agents.structured_output import extract_validated
        from app_builder.schemas.app_definition_gen import AppDefinitionGen
    except ImportError:
        return None

    plan_json = plan_json or {}
    screens = extract_screens_from_plan(
        requirement=requirement, prd=prd, uiux=uiux, plan_json=plan_json,
    )
    screen_hint = json.dumps(screens, indent=2) if screens else "[]"
    prd_excerpt = (prd or "")[:12000]
    uiux_excerpt = (uiux or "")[:8000]
    arch = plan_json.get("architecture")
    arch_excerpt = json.dumps(arch, indent=2)[:8000] if arch else ""

    user_prompt = f"""Requirement:
{requirement}

Suggested screens (use or refine):
{screen_hint}

PRD excerpt:
{prd_excerpt}

UI/UX excerpt:
{uiux_excerpt}

Architecture JSON:
{arch_excerpt}
"""

    llm = get_llm_for_user(None, temperature=0.25, streaming=False)
    gen = await extract_validated(
        llm=llm,
        system_prompt=_APP_DEF_LLM_SYSTEM,
        user_prompt=user_prompt,
        schema=AppDefinitionGen,
        max_retries=2,
        context_label="app_definition_gen",
    )
    if gen is None:
        return None

    theme = _theme_from_tokens(
        plan_json.get("design_tokens") if isinstance(plan_json.get("design_tokens"), dict) else None
    )
    try:
        return _gen_to_app_definition(gen, theme)
    except Exception as exc:
        logger.warning("[app_definition] LLM gen conversion failed: %s", exc)
        return None


async def resolve_app_definition(
    *,
    requirement: str = "",
    prd: str = "",
    uiux: str = "",
    plan_json: Optional[Dict[str, Any]] = None,
    app_name: Optional[str] = None,
    use_llm: bool = True,
    prefer_stored: bool = False,
    on_progress: Optional[Any] = None,
) -> AppDefinition:
    plan_json = plan_json or {}

    if prefer_stored:
        stored = plan_json.get("app_definition")
        if isinstance(stored, dict) and stored.get("pages"):
            try:
                return finalize_app_definition(AppDefinition.model_validate(stored))
            except Exception:
                pass

    if use_llm:
        if on_progress:
            await on_progress("LLM: structuring navigation and page blocks from PRD…")
        llm_def = await generate_app_definition_with_llm(
            requirement=requirement,
            prd=prd,
            uiux=uiux,
            plan_json=plan_json,
        )
        if on_progress and llm_def is not None:
            await on_progress("LLM: validating schema and enriching MVP mock datasets…")
        if llm_def is not None:
            llm_def.meta.appName = resolve_app_display_name(
                requirement=requirement,
                prd=prd,
                plan_json=plan_json,
                app_name=app_name or llm_def.meta.appName,
            )
            logger.info(
                "[app_definition] LLM | pages=%d nav=%d",
                len(llm_def.pages),
                len(llm_def.navigation),
            )
            return llm_def

    if on_progress:
        await on_progress("Deterministic fallback: building AppDefinition from PRD screen map…")
    built = build_app_definition_from_plan(
        requirement=requirement,
        prd=prd,
        uiux=uiux,
        plan_json=plan_json,
        app_name=app_name,
    )
    if on_progress:
        await on_progress("Enriching production-style sample data for charts and tables…")
    return finalize_app_definition(built)


def resolve_app_definition_sync(
    *,
    requirement: str = "",
    prd: str = "",
    uiux: str = "",
    plan_json: Optional[Dict[str, Any]] = None,
    app_name: Optional[str] = None,
) -> AppDefinition:
    """Sync path for tests / scaffold — PRD screen extraction, no LLM."""
    plan_json = plan_json or {}
    stored = plan_json.get("app_definition")
    if isinstance(stored, dict) and stored.get("pages"):
        try:
            return finalize_app_definition(AppDefinition.model_validate(stored))
        except Exception:
            pass
    return finalize_app_definition(
        build_app_definition_from_plan(
            requirement=requirement,
            prd=prd,
            uiux=uiux,
            plan_json=plan_json,
            app_name=app_name,
        )
    )
