"""Route multi-page dashboard prompts to json_app (data.json + PageRenderer), not legacy templates."""
from __future__ import annotations

import re

from app_builder.services.prd_prompt_builder import infer_pages_from_requirement, is_multi_page_dashboard_prompt


def is_operations_dashboard_prompt(requirement: str, prd: str = "", uiux: str = "") -> bool:
    """Manufacturing / maintenance style dashboards — never the anomaly SaaS template."""
    text = f"{requirement or ''}\n{prd or ''}\n{uiux or ''}".lower()
    ops = bool(
        re.search(
            r"maintenance|manufacturing|mtbf|mttr|work order|downtime|oee|plant ops|operations",
            text,
        )
    )
    pages = bool(re.search(r"dashboard|analytics|reports?", text))
    explicit = bool(
        re.search(r"dashboard,\s*analytics,\s*reports?", text)
        or re.search(r"3\s+pages?:\s*dashboard", text)
    )
    pages_list = infer_pages_from_requirement(requirement or prd)
    named = {"dashboard", "analytics", "reports"}.issubset({p.lower() for p in pages_list})
    return (ops and pages) or explicit or named


def should_force_json_app_track(requirement: str, prd: str = "", uiux: str = "") -> bool:
    """
    Use json_app codegen when the user asked for a data-driven multi-page dashboard
    (RAG chat, Predict, KPIs in JSON) — not fullstack AI pages or anomaly_detection legacy.
    """
    text = f"{requirement or ''}\n{prd or ''}\n{uiux or ''}".lower()
    if re.search(r"\b(ecommerce|shopping cart|online store|product catalog)\b", text):
        return False
    if re.search(r"\b(document upload|doc chat|rag over documents)\b", text) and not re.search(
        r"dashboard|analytics|reports?", text
    ):
        return False

    if is_operations_dashboard_prompt(requirement, prd, uiux):
        return True

    if is_multi_page_dashboard_prompt(requirement or prd):
        if re.search(r"\b(rag|data assistant|predict|forecast|what.?if)\b", text):
            return True
        if re.search(r"maintenance|manufacturing|kpi|chart", text):
            return True

    return False
