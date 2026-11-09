"""Inject Observability nav page when the user asks for LLM usage / token tracking in the prompt."""
from __future__ import annotations

import re

from app_builder.schemas.app_definition import AppDefinition, NavItemDef, PageDef


def wants_observability_screen(prd: str, requirement: str) -> bool:
    blob = f"{prd}\n{requirement}".lower()
    if re.search(
        r"\b(observability|llm usage|token usage|tokens consumed|model cost|ai usage|"
        r"usage monitoring|llm observability|prompt.{0,20}output|track.{0,12}tokens)\b",
        blob,
    ):
        return True
    if re.search(r"\bobservability\b", blob) and re.search(
        r"\b(chat|assistant|llm|model|tokens|prompt)\b", blob
    ):
        return True
    return False


def ensure_observability_features(
    defn: AppDefinition, *, prd: str = "", requirement: str = ""
) -> AppDefinition:
    if not wants_observability_screen(prd, requirement):
        return defn

    existing = list(defn.meta.capabilities or [])
    for cap in ("llm_observability", "rag_chat", "rag_export"):
        if cap not in existing:
            existing.append(cap)
    defn.meta = defn.meta.model_copy(update={"capabilities": existing})

    nav = list(defn.navigation or [])
    pages = list(defn.pages or [])
    if any(n.pageId == "observability" for n in nav):
        return defn

    nav.append(
        NavItemDef(
            label="Observability",
            path="/observability",
            icon="Insights",
            pageId="observability",
        )
    )
    pages.append(
        PageDef(
            id="observability",
            title="Observability",
            subtitle="Prompts, model outputs, token usage, and estimated cost per LLM call",
            blocks=[{"type": "text", "content": "", "hidden": True}],
        )
    )
    defn.navigation = nav
    defn.pages = pages
    return defn
