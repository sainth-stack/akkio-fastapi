"""Normalize sidebar labels for json_app shells."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

_NAV_ICONS = (
    (("dashboard", "overview", "home"), "Dashboard"),
    (("analytic", "insight", "metric", "chart"), "ShowChart"),
    (("report", "export"), "Assessment"),
    (("predict", "forecast"), "TrendingUp"),
    (("quality", "inspection"), "FactCheck"),
)


def _icon_for_label(label: str) -> str:
    low = (label or "").lower()
    for keys, icon in _NAV_ICONS:
        if any(k in low for k in keys):
            return icon
    return "Article"


def friendly_nav_label(
    page_id: str,
    title: str,
    *,
    dashboard_page_id: Optional[str] = None,
) -> str:
    pid = (page_id or "").lower()
    t = (title or "").lower()
    dash = (dashboard_page_id or "").lower()
    if dash and pid == dash:
        return "Dashboard"
    if pid in ("dashboard", "home"):
        return "Dashboard"
    if pid == "analytics" or re.search(r"\banalytics?\b", t) or re.search(r"\bcharts?\b", t):
        return "Analytics"
    if pid == "reports" or re.search(r"\breports?\b", t):
        return "Reports"
    if pid == "predict":
        return "Predict"
    if pid == "observability":
        return "Observability"
    if pid in ("forecasting", "forecast") or re.search(r"\bforecasting\b", t):
        return "Forecasting"
    if re.search(r"\bpredict\b", t) and pid != "forecasting":
        return "Predict"
    if re.match(r"^(dashboard|home|overview|executive overview)\b", t.strip()):
        return "Dashboard"
    return (title or page_id or "Page").strip()


def _prune_junk_nav_pages(defn: Dict[str, Any]) -> Dict[str, Any]:
    """Drop LLM capability phrases mistaken for nav pages (e.g. 'Tokens', 'RAG chat')."""
    from app_builder.services.app_definition_generator import (
        _RESERVED_SCREEN_IDS,
        _is_plausible_app_page_name,
    )

    out = dict(defn)
    keep_ids = {"predict", "observability", "create_ticket", "similar_tickets"}
    pages = [
        p
        for p in (out.get("pages") or [])
        if isinstance(p, dict)
        and (
            str(p.get("id") or "") in keep_ids
            or (
                str(p.get("id") or "") not in _RESERVED_SCREEN_IDS
                and _is_plausible_app_page_name(str(p.get("title") or p.get("id") or ""))
            )
        )
    ]
    page_ids = {str(p.get("id") or "") for p in pages}
    nav = [
        n
        for n in (out.get("navigation") or [])
        if isinstance(n, dict)
        and (
            str(n.get("pageId") or "") in keep_ids
            or (
                str(n.get("pageId") or "") in page_ids
                and str(n.get("pageId") or "") not in _RESERVED_SCREEN_IDS
                and _is_plausible_app_page_name(str(n.get("label") or n.get("pageId") or ""))
            )
        )
    ]
    out["pages"] = pages
    out["navigation"] = nav
    return out


def normalize_definition_navigation(defn: Dict[str, Any]) -> Dict[str, Any]:
    from app_builder.services.page_layout_intent import resolve_dashboard_page_id

    out = _prune_junk_nav_pages(defn)
    pages = out.get("pages") or []
    nav = out.get("navigation") or []
    if not nav:
        return out

    page_tuples: List[Tuple[str, str]] = [
        (str(p.get("id") or ""), str(p.get("title") or ""))
        for p in pages
        if isinstance(p, dict) and p.get("id")
    ]
    nav_ids: Sequence[str] = [
        str(n.get("pageId") or "") for n in nav if isinstance(n, dict) and n.get("pageId")
    ]
    dash_id = resolve_dashboard_page_id(page_tuples, nav_ids)

    page_ids = {pt[0] for pt in page_tuples}
    seen_page_ids: set = set()
    seen_paths: set = set()
    new_nav: List[Dict[str, Any]] = []

    for i, item in enumerate(nav):
        if not isinstance(item, dict):
            continue
        page_id = str(item.get("pageId") or "")
        if page_id not in page_ids and i < len(pages):
            page_id = str(pages[i].get("id") or page_id)
        if not page_id or page_id in seen_page_ids:
            continue
        seen_page_ids.add(page_id)

        path = str(item.get("path") or "/")
        if path in seen_paths:
            path = "/" if page_id == dash_id else f"/{page_id.replace('_', '-')}"
        if page_id == "predict" and path != "/predict":
            path = "/predict"
        if page_id == "observability" and path != "/observability":
            path = "/observability"
        if page_id == "create_ticket" and path != "/create-ticket":
            path = "/create-ticket"
        if page_id == "similar_tickets" and path != "/similar-tickets":
            path = "/similar-tickets"
        seen_paths.add(path)

        page_match = next((p for p in pages if isinstance(p, dict) and str(p.get("id")) == page_id), None)
        raw = str((page_match or {}).get("title") or item.get("label") or page_id)
        label = friendly_nav_label(page_id, raw, dashboard_page_id=dash_id)
        icon = str(item.get("icon") or "")
        if not icon or icon == "Article":
            icon = _icon_for_label(label)

        new_nav.append(
            {
                **item,
                "pageId": page_id,
                "path": path,
                "label": label,
                "icon": icon,
            }
        )

    out["navigation"] = new_nav
    return out
