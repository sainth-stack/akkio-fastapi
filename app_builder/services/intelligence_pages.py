"""Inject Predict page + capabilities for RAG export and synthetic prediction."""
from __future__ import annotations

import re
from typing import Any, List

from app_builder.schemas.app_definition import AppDefinition, NavItemDef, PageDef


def _wants_intelligence(prd: str, requirement: str) -> bool:
    blob = f"{prd}\n{requirement}".lower()
    if re.search(r"\b(predict|forecast|rag|data agent|chat|assistant)\b", blob):
        return True
    if re.search(r"\b(maintenance|manufacturing|operations|dashboard|monitoring)\b", blob):
        return True
    return False


def ensure_intelligence_features(defn: AppDefinition, *, prd: str = "", requirement: str = "") -> AppDefinition:
    if defn.meta.domain == "ecommerce" or defn.meta.appName.lower().find("shop") >= 0:
        return defn
    if not _wants_intelligence(prd, requirement):
        return defn

    existing = list(defn.meta.capabilities or [])
    for c in ("rag_chat", "rag_export", "predict"):
        if c not in existing:
            existing.append(c)
    defn.meta = defn.meta.model_copy(update={"capabilities": existing})

    nav = list(defn.navigation or [])
    pages = list(defn.pages or [])
    if any(n.pageId == "predict" for n in nav):
        return defn

    nav.append(
        NavItemDef(label="Predict", path="/predict", icon="TrendingUp", pageId="predict")
    )
    pages.append(
        PageDef(
            id="predict",
            title="Predict",
            subtitle="Run what-if forecasts on synthetic operational inputs",
            blocks=[{"type": "text", "content": "", "hidden": True}],
        )
    )
    defn.navigation = nav
    defn.pages = pages
    return defn
