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
from app_builder.services.generic_dataset_builder import ensure_datasets_from_blocks
from app_builder.services.prd_structure_parser import section_for_page
from app_builder.services.prd_driven_layout import blocks_for_page as prd_blocks_for_page

logger = logging.getLogger("app_builder")

_ICON_HINTS = (
    (("dashboard", "overview", "home"), "Dashboard"),
    (("session", "inspection"), "FactCheck"),
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


def _normalize_prompt_text(text: str) -> str:
    """Strip zero-width chars and normalize bullet glyphs for screen parsing."""
    if not text:
        return ""
    t = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    t = re.sub(r"•\s*", "- ", t)
    return t


def _screens_from_numbered_prompt_list(text: str) -> List[Dict[str, str]]:
    """Parse '1. **Dashboard:** …' style page lists in the user prompt."""
    text = _normalize_prompt_text(text)
    if not text:
        return []
    out: List[Dict[str, str]] = []
    for m in re.finditer(
        r"(?m)^\s*\d+[\.\)]\s*\**\s*([A-Za-z][A-Za-z0-9\s/&-]{1,40}?)\s*\**\s*:",
        text,
    ):
        title = m.group(1).strip()
        if _is_nav_screen_title(title):
            out.append({"id": _slug(title), "title": title, "purpose": title})
    return out


_PRD_DOC_HEADING = re.compile(
    r"^(pages?|product overview|business requirements?|functional requirements?|"
    r"non[- ]functional|appendix|glossary|introduction|objectives?|scope|"
    r"assumptions?|constraints?|success metrics?|user stories?|wireframes?)\b",
    re.I,
)


def _is_nav_screen_title(title: str) -> bool:
    t = re.sub(r"^\d+[\.\)]\s*", "", (title or "").strip()).rstrip(":").strip()
    if not t or len(t) > 56:
        return False
    if _PRD_DOC_HEADING.match(t):
        return False
    if re.match(r"^\d+\.\d+", t):
        return False
    low = t.lower()
    if low in ("pages", "page", "screens", "screen", "prompt"):
        return False
    return True


def _screens_from_requirement_phrase(text: str) -> List[Dict[str, str]]:
    """Parse '… with dashboard, analytics and reports' style lists."""
    text = _normalize_prompt_text(text)
    if not text:
        return []
    numbered = _screens_from_numbered_prompt_list(text)
    if len(numbered) >= 2:
        return numbered
    pages_list = re.search(r"with\s+\d+\s+pages?\s*:\s*([^.;\n]+)", text, re.I)
    if pages_list:
        chunk = pages_list.group(1)
        parts = re.split(r",|\band\b", chunk)
        out: List[Dict[str, str]] = []
        for p in parts:
            name = p.strip()
            if 2 < len(name) < 48 and _is_nav_screen_title(name):
                title = name[0].upper() + name[1:] if name.islower() else name
                out.append({"id": _slug(title), "title": title, "purpose": title})
        if len(out) >= 2:
            return out
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
            # "dashboard sessions reports analytics" without commas
            if " " in name and len(name) > 24:
                tokens = re.findall(
                    r"\b(dashboard|overview|home|sessions?|reports?|analytics|insights?|metrics?|settings|users?|inventory)\b",
                    name,
                    re.I,
                )
                if len(tokens) >= 2:
                    for tok in tokens:
                        title = tok[0].upper() + tok[1:].lower()
                        if title.lower() == "session":
                            title = "Sessions"
                        out.append({"id": _slug(title), "title": title, "purpose": title})
                    continue
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
    combined = "\n".join([requirement, prd, uiux])

    req_from_requirement = _dedupe_screens(_screens_from_requirement_phrase(requirement or ""))
    if len(req_from_requirement) >= 2:
        screens = req_from_requirement
    else:
        arch_screens = _dedupe_screens(_screens_from_architecture(plan_json.get("architecture")))
        req_screens = _dedupe_screens(_screens_from_requirement_phrase(combined))
        if len(req_screens) >= 2:
            screens = req_screens
        elif len(arch_screens) >= 2:
            screens = arch_screens
        else:
            collected: List[Dict[str, str]] = []
            collected.extend(_screens_from_prd_dict(plan_json.get("prd")))
            collected.extend(arch_screens)
            bp = plan_json.get("blueprint")
            if isinstance(bp, dict):
                collected.extend(_screens_from_blueprint_nav(bp))
            collected.extend(req_screens)

            for m in re.finditer(r"(?m)^#+\s*(.+)$", prd or ""):
                t = re.sub(r"^\d+[\.\)]\s*", "", m.group(1).strip())
                if _is_nav_screen_title(t) and not t.lower().startswith(("prd", "goal", "feature")):
                    collected.append({"id": _slug(t), "title": t, "purpose": t})

            if not collected:
                collected.extend(_screens_from_requirement_phrase(requirement))
            screens = _dedupe_screens(collected)

    screens = screens[:8]
    if screens and screens[0]["id"] != "dashboard" and "dashboard" in screens[0]["title"].lower():
        screens[0]["id"] = "dashboard"
    return screens


def align_shell_plan_to_screens(shell: Any, screens: List[Dict[str, str]]) -> Any:
    """Force LLM shell page ids/titles and nav labels to match parsed requirement screens."""
    from app_builder.schemas.app_definition_llm import AppShellPlan, NavItemPlan, PageOutlinePlan

    if not isinstance(shell, AppShellPlan) or not screens or not shell.pages:
        return shell
    n = min(len(screens), len(shell.pages))
    if n < 1:
        return shell
    pages: List[PageOutlinePlan] = []
    for i in range(len(shell.pages)):
        if i < n:
            s = screens[i]
            old = shell.pages[i]
            pages.append(
                PageOutlinePlan(
                    id=str(s.get("id") or old.id),
                    title=str(s.get("title") or old.title),
                    subtitle=old.subtitle or str(s.get("purpose") or s.get("title") or ""),
                )
            )
        else:
            pages.append(shell.pages[i])
    navigation: List[NavItemPlan] = []
    for i, nav in enumerate(shell.navigation):
        page = pages[i] if i < len(pages) else None
        if page is not None:
            navigation.append(
                NavItemPlan(
                    label=page.title,
                    path="/" if i == 0 else (nav.path if nav.path and nav.path != "/" else f"/{page.id}"),
                    icon=nav.icon or "Article",
                    pageId=page.id,
                )
            )
        else:
            navigation.append(nav)
    return shell.model_copy(update={"pages": pages, "navigation": navigation})


_BRAND_STOP = frozenset(
    {
        "dashboard",
        "platform",
        "application",
        "app",
        "system",
        "suite",
        "professional",
        "create",
        "build",
        "generate",
        "the",
        "and",
        "for",
        "with",
    }
)


def shorten_brand_name(long_name: str, max_words: int = 2) -> str:
    """Sidebar-friendly product name (1–2 words), derived from the full title."""
    t = _clean_product_title(long_name or "")
    t = re.sub(
        r"\b(dashboard|platform|application|monitoring|system|suite)\b",
        " ",
        t,
        flags=re.I,
    )
    words = [w for w in re.findall(r"[A-Za-z]{3,}", t) if w.lower() not in _BRAND_STOP]
    if not words:
        return "Workspace"
    if len(words) == 1:
        return words[0].title()[:32]
    # Prefer a meaningful pair (e.g. Quality + Manufacturing → "Quality Ops")
    low = [w.lower() for w in words]
    if "quality" in low:
        other = next((w for w in words if w.lower() not in ("quality", "manufacturing")), words[0])
        return f"Quality {other.title()}"[:32]
    return f"{words[0].title()} {words[1].title()}"[:32]


def _clean_product_title(raw: str) -> str:
    """Turn chat-style prompts into product titles."""
    t = (raw or "").strip()
    t = re.sub(r"^(build|create|generate|make)\s+(a|an|the|product)?\s*", "", t, flags=re.I).strip()
    t = re.sub(r"\s+app(lication)?\s*$", "", t, flags=re.I).strip()
    if not t:
        return "Operations Workspace"
    # Title-case if mostly lowercase prompt
    if t.islower() or sum(1 for c in t if c.isupper()) < 2:
        t = t.title()
    t = t.replace("Monitering", "Monitoring").replace("monitering", "monitoring")
    # Drop trailing screen lists: "Transformer Monitoring With Dashboard …"
    t = re.sub(
        r"\s+with\s+(dashboard|sessions?|reports?|analytics).*$",
        "",
        t,
        flags=re.I,
    ).strip()
    if len(t) > 48:
        t = re.sub(r"\s+with\s+.*$", "", t, flags=re.I).strip()
    return t[:80]


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
            return shorten_brand_name(app_name)

    _page_bold = frozenset({"dashboard", "sessions", "reports", "analytics", "prompt"})
    for match in re.findall(r"\*\*([^*]+)\*\*", requirement or ""):
        candidate = match.strip().rstrip(":").strip()
        if len(candidate) < 12 or candidate.lower() in _page_bold:
            continue
        if "dashboard" in candidate.lower() or "monitoring" in candidate.lower():
            return shorten_brand_name(_clean_product_title(candidate))
    mfg = re.search(
        r"(?:create|build)\s+(?:a\s+)?(?:professional\s+)?([A-Za-z0-9][^\n*]{8,80}?Dashboard)",
        requirement or "",
        re.I,
    )
    if mfg:
        return shorten_brand_name(_clean_product_title(mfg.group(1).strip()))
    quoted = re.search(r'"([^"]{6,90})"', requirement or "")
    if quoted:
        return shorten_brand_name(_clean_product_title(quoted.group(1).strip()))

    plan_json = plan_json or {}
    blueprint = plan_json.get("blueprint") if isinstance(plan_json.get("blueprint"), dict) else {}
    if isinstance(blueprint, dict) and blueprint.get("app_name"):
        return shorten_brand_name(str(blueprint["app_name"]))

    prd_obj = plan_json.get("prd")
    if isinstance(prd_obj, dict):
        for key in ("app_name", "product_name", "title", "name"):
            if prd_obj.get(key):
                return shorten_brand_name(str(prd_obj[key]))

    title = _clean_product_title(_title_from_requirement(requirement, "My Application"))
    if title and title not in ("My Application", "Workspace"):
        return shorten_brand_name(title)

    try:
        from app_builder.services.fullstack_app_generator import extract_app_title

        extracted = extract_app_title(requirement, prd)
        if extracted and extracted not in ("Enterprise Application", "Anomaly Detection Platform"):
            return shorten_brand_name(_clean_product_title(extracted))
    except ImportError:
        pass

    return shorten_brand_name(title)


def _infer_blocks_for_screen(
    screen: Dict[str, str],
    index: int,
    total: int,
    *,
    domain: str = "",
    meta_description: str = "",
    prd: str = "",
    dashboard_page_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    sid = screen.get("id") or "page"
    title = screen.get("title") or sid
    section = section_for_page(prd, title, sid)
    if not section:
        section = f"{screen.get('purpose') or ''}\n{meta_description}"
    return prd_blocks_for_page(sid, title, section, dashboard_page_id=dashboard_page_id)


def _collect_mock_keys(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    mock: Dict[str, Any] = {}

    def ensure_series(key: str, kind: str) -> None:
        if key in mock:
            return
        if kind == "line":
            mock[key] = [
                {"period": f"2024-06-{(i % 28) + 1:02d}", "value": 70 + i * 3, "target": 85}
                for i in range(6)
            ]
        else:
            mock[key] = [
                {"period": f"2024-06-{(i % 28) + 1:02d}", "actual": 40 + i * 5, "benchmark": 50 + i * 2}
                for i in range(6)
            ]

    def ensure_table(key: str, columns: List[Dict[str, Any]]) -> None:
        if key in mock:
            existing = mock[key]
            if isinstance(existing, list) and existing:
                from app_builder.services.generic_dataset_builder import _table_has_placeholders

                if not _table_has_placeholders(existing):
                    return
        from app_builder.services.generic_dataset_builder import _cell_value, _seed

        from app_builder.services.block_normalizer import _columns_are_invalid, _infer_table_columns

        if isinstance(columns, list) and _columns_are_invalid(columns):
            columns = _infer_table_columns({"dataKey": key}, "page", "Page")
        fields = [c.get("field") for c in columns if c.get("field")]
        if not fields:
            fields = ["id", "name", "status"]
        rng = _seed(str(key))
        mock[key] = [{str(f): _cell_value(str(f), i, rng) for f in fields} for i in range(18)]

    for block in blocks:
        btype = block.get("type")
        if btype in ("lineChart", "barChart"):
            dk = block.get("dataKey")
            if dk:
                ensure_series(str(dk), "line" if btype == "lineChart" else "bar")
        elif btype == "dataTable":
            dk = block.get("dataKey")
            cols = block.get("columns") or []
            if dk:
                ensure_table(str(dk), cols if isinstance(cols, list) else [])
        elif btype == "pieChart":
            dk = block.get("dataKey")
            if dk and dk not in mock:
                mock[str(dk)] = []

    return mock


def build_app_definition_from_screens(
    screens: List[Dict[str, str]],
    *,
    app_name: str,
    description: str,
    domain: str,
    theme: Optional[AppTheme] = None,
    prd: str = "",
) -> AppDefinition:
    if not screens:
        screens = [
            {"id": "dashboard", "title": "Dashboard", "purpose": "Overview"},
            {"id": "details", "title": "Details", "purpose": "Records"},
        ]

    from app_builder.services.page_layout_intent import resolve_dashboard_page_id

    navigation: List[NavItemDef] = []
    pages: List[PageDef] = []
    mock: Dict[str, Any] = {}
    dash_page_id = resolve_dashboard_page_id(
        [(s.get("id") or "page", s.get("title") or s.get("id") or "page") for s in screens],
        None,
    )

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

        blocks = _infer_blocks_for_screen(
            screen,
            i,
            len(screens),
            domain=domain,
            meta_description=description,
            prd=prd,
            dashboard_page_id=dash_page_id,
        )
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
        prd=prd,
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
    from app_builder.services.app_theme_generator import ensure_theme_variation_seed

    plan.setdefault("theme_variation_seed", ensure_theme_variation_seed(plan))
    return plan


def finalize_app_definition(
    defn: AppDefinition,
    *,
    prd: str = "",
    requirement: str = "",
    design_tokens: Optional[Dict[str, Any]] = None,
    theme_variation_seed: str = "",
) -> AppDefinition:
    """Ensure mock data exists for every block reference; align nav pageIds."""
    from app_builder.services.page_layout_intent import strip_misplaced_kpi_blocks
    from app_builder.services.prd_structure_parser import section_for_page
    from app_builder.services.app_theme_generator import ensure_app_visual_identity

    defn.meta.appName = shorten_brand_name(defn.meta.appName or "Workspace")
    defn = ensure_app_visual_identity(
        defn,
        requirement=requirement,
        design_tokens=design_tokens,
        variation_seed=theme_variation_seed,
    )
    from app_builder.services.page_layout_intent import (
        enforce_kpis_on_dashboard_only,
        enforce_page_role_blocks,
        resolve_dashboard_page_id,
    )

    dash_page_id = resolve_dashboard_page_id(
        [(p.id, p.title) for p in defn.pages],
        [n.pageId for n in defn.navigation],
    )
    cleaned_pages: List[PageDef] = []
    for page in defn.pages:
        section = section_for_page(prd, page.title, page.id) if prd else ""
        blocks = strip_misplaced_kpi_blocks(
            page.blocks,
            title=page.title,
            page_id=page.id,
            purpose=page.subtitle or "",
            section=section,
            dashboard_page_id=dash_page_id,
        )
        cleaned_pages.append(
            PageDef(id=page.id, title=page.title, subtitle=page.subtitle, blocks=blocks)
        )
    defn.pages = cleaned_pages
    from app_builder.services.page_layout_intent import repair_pages_with_no_blocks

    defn = repair_pages_with_no_blocks(defn, prd=prd, requirement=requirement)
    defn = enforce_page_role_blocks(defn, prd=prd)
    from app_builder.services.page_layout_intent import strip_export_download_ui

    defn = strip_export_download_ui(defn)
    defn = enforce_kpis_on_dashboard_only(defn, prd=prd)
    from app_builder.services.dashboard_layout import apply_dashboard_analytics_product_rules

    defn = apply_dashboard_analytics_product_rules(defn, prd=prd)
    from app_builder.services.intelligence_pages import ensure_intelligence_features

    defn = ensure_intelligence_features(defn, prd=prd, requirement=requirement)
    from app_builder.services.block_normalizer import repair_broken_data_table_blocks

    defn = repair_broken_data_table_blocks(defn, prd=prd)
    mock = dict(defn.mock or {})
    for page in defn.pages:
        mock.update(_collect_mock_keys(page.blocks))
    defn.mock = mock
    defn = normalize_app_definition_blocks(defn, prd=prd)
    defn = ensure_datasets_from_blocks(defn)
    from app_builder.services.block_normalizer import repair_table_mock_keys

    defn = repair_table_mock_keys(defn)
    defn = repair_broken_data_table_blocks(defn, prd=prd)
    defn = repair_table_mock_keys(defn)
    from app_builder.services.production_dataset_enhancer import enhance_production_datasets
    from app_builder.services.production_data_polish import polish_production_data_bag

    defn = enhance_production_datasets(defn)
    from app_builder.services.block_normalizer import align_table_datasets_to_columns

    defn = align_table_datasets_to_columns(defn)
    defn.mock = polish_production_data_bag(defn)
    from app_builder.services.app_theme_generator import apply_theme_to_chart_blocks

    defn = apply_theme_to_chart_blocks(defn, replace_generic_colors=True)
    defn = strip_export_download_ui(defn)
    mock = dict(defn.mock or {})
    page_ids = {p.id for p in defn.pages}
    navigation: List[NavItemDef] = []
    for i, nav in enumerate(defn.navigation):
        page_id = nav.pageId if nav.pageId in page_ids else (defn.pages[i].id if i < len(defn.pages) else nav.pageId)
        page_match = next((p for p in defn.pages if p.id == page_id), None)
        label = page_match.title if page_match else nav.label
        icon = nav.icon or "Article"
        if icon == "Dashboard" and label.lower() != "dashboard":
            icon = _icon_for_label(label)
        elif not nav.icon or nav.icon == "Article":
            icon = _icon_for_label(label)
        navigation.append(
            NavItemDef(label=label, path=nav.path, icon=icon, pageId=page_id)
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
- Use block types: filterBar, actionBar, kpiRow, lineChart, barChart, pieChart, dataTable, text.
- Every dataKey in blocks must have matching entries in mock (arrays for charts/tables, strings/numbers for KPI valueKeys).
- KPI rows: store values only in data["kpis_<pageId>"] = { "oee": "87.4%", ... } (no duplicate flat keys).
- Mock data must look like a real MVP product: use proper domain names (transformer units, sessions, metrics), 15+ table rows, 10+ chart points.
- Each page's mock keys must match that page (sessions → session_data with operators; analytics → metrics; reports → report titles).
- kpiRow: use dataKey kpis_<pageId> with semantic keys (load_mva, oil_temp, oee) not generic a/b/c.
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
  "data": { ... operational datasets only — no placeholder scalars }
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
        from app_builder.services.app_definition_llm_pipeline import generate_app_definition_llm

        async def _progress(msg: str) -> None:
            if on_progress:
                await on_progress(msg)

        llm_def = await generate_app_definition_llm(
            requirement=requirement,
            prd=prd,
            uiux=uiux,
            plan_json=plan_json,
            app_name=app_name,
            on_progress=_progress if on_progress else None,
        )
        if llm_def is None:
            llm_def = await generate_app_definition_with_llm(
                requirement=requirement,
                prd=prd,
                uiux=uiux,
                plan_json=plan_json,
            )
            if llm_def is not None:
                llm_def = finalize_app_definition(
                    llm_def,
                    prd=prd,
                    requirement=requirement,
                    design_tokens=plan_json.get("design_tokens")
                    if isinstance(plan_json.get("design_tokens"), dict)
                    else None,
                    theme_variation_seed=str(plan_json.get("theme_variation_seed") or ""),
                )
        if llm_def is not None:
            llm_def.meta.appName = resolve_app_display_name(
                requirement=requirement,
                prd=prd,
                plan_json=plan_json,
                app_name=app_name or llm_def.meta.appName,
            )
            llm_def.meta.generatedBy = llm_def.meta.generatedBy or "llm_pipeline"
            logger.info(
                "[app_definition] LLM | pages=%d nav=%d keys=%d",
                len(llm_def.pages),
                len(llm_def.navigation),
                len(llm_def.mock or {}),
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
    return finalize_app_definition(
        built,
        prd=prd,
        requirement=requirement,
        design_tokens=plan_json.get("design_tokens") if isinstance(plan_json.get("design_tokens"), dict) else None,
        theme_variation_seed=str(plan_json.get("theme_variation_seed") or ""),
    )


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
        ),
        prd=prd,
        requirement=requirement,
        design_tokens=plan_json.get("design_tokens") if isinstance(plan_json.get("design_tokens"), dict) else None,
        theme_variation_seed=str(plan_json.get("theme_variation_seed") or ""),
    )
