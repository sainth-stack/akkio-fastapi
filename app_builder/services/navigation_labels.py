"""Normalize sidebar labels for json_app shells."""
from __future__ import annotations

from typing import Any, Dict, List


def friendly_nav_label(page_id: str, title: str) -> str:
    pid = (page_id or "").lower()
    t = (title or "").lower()
    if pid in ("dashboard", "home") or "dashboard" in t or ("kpi" in t and "card" in t):
        return "Dashboard"
    if pid == "analytics" or "analytics" in t or "chart" in t:
        return "Analytics"
    if pid == "reports" or "report" in t:
        return "Reports"
    if pid == "predict":
        return "Predict"
    return (title or page_id or "Page").strip()


def normalize_definition_navigation(defn: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defn)
    pages = out.get("pages") or []
    nav = out.get("navigation") or []
    if not nav:
        return out
    page_ids = {str(p.get("id")) for p in pages if isinstance(p, dict)}
    new_nav: List[Dict[str, Any]] = []
    for i, item in enumerate(nav):
        if not isinstance(item, dict):
            continue
        page_id = str(item.get("pageId") or "")
        if page_id not in page_ids and i < len(pages):
            page_id = str(pages[i].get("id") or page_id)
        page_match = next((p for p in pages if isinstance(p, dict) and str(p.get("id")) == page_id), None)
        raw = str((page_match or {}).get("title") or item.get("label") or page_id)
        new_nav.append({**item, "pageId": page_id, "label": friendly_nav_label(page_id, raw)})
    out["navigation"] = new_nav
    return out
