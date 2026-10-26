"""Chat over generated app data (RAG context + server LLM API key)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app_builder.services.app_data_rag import (
    CHAT_SYSTEM_TEMPLATE,
    ECOMMERCE_CHAT_SYSTEM_TEMPLATE,
    build_rag_context,
    load_project_app_definition,
    _is_ecommerce_definition,
)

logger = logging.getLogger("app_builder")


async def answer_app_chat(
    project_id: str,
    messages: List[Dict[str, str]],
    *,
    page_id: Optional[str] = None,
    app_definition: Optional[Dict[str, Any]] = None,
) -> str:
    defn = app_definition or load_project_app_definition(project_id)
    if not defn:
        raise ValueError("App definition not found for this project.")

    last_user = ""
    for m in reversed(messages):
        if (m.get("role") or "").lower() == "user":
            last_user = str(m.get("content") or "")
            break

    import re

    low = (last_user or "").lower()
    if re.search(r"\b(download|export)\b.*\b(rag|context|data)\b", low) or re.search(
        r"\brag data\b", low
    ):
        return (
            "Use **Download RAG data (PDF)** in this chat panel to export the grounded context "
            "(KPIs, charts, tables, and catalog data) used for answers."
        )

    if re.search(r"\b(predict|forecast|what.?if|projection)\b", low):
        from app_builder.services.synthetic_predict import (
            default_features_from_definition,
            predict_from_features,
        )

        feats = default_features_from_definition(defn)
        pred = predict_from_features(feats, app_definition=defn)
        return (
            f"**Synthetic forecast** (from current app data):\n"
            f"- Predicted downtime (next period): **{pred['predicted_downtime_hours_next_period']} h**\n"
            f"- Predicted MTBF: **{pred['predicted_mtbf_hours']} h**\n"
            f"- Risk level: **{pred['risk_level']}** (confidence {pred['confidence']:.0%})\n"
            f"- Drivers: {', '.join(pred['drivers'])}\n"
            f"- Recommendation: {pred['recommendation']}\n\n"
            "Open **Predict** in the sidebar to change inputs manually or run another scenario."
        )

    context = build_rag_context(defn, last_user or "overview", focus_page_id=page_id)
    meta = defn.get("meta") or {}
    app_name = str(meta.get("appName") or "Dashboard")
    if _is_ecommerce_definition(defn):
        system = ECOMMERCE_CHAT_SYSTEM_TEMPLATE.format(app_name=app_name, context=context)
    else:
        system = CHAT_SYSTEM_TEMPLATE.format(app_name=app_name, context=context)

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from llm_config import get_llm_config
    from llm_helper import get_llm_for_user

    llm_cfg = get_llm_config(None)
    if not (llm_cfg.get("api_key") or "").strip():
        raise ValueError(
            "No LLM API key on the server. Set OPENAI_API_KEY (or ANTHROPIC_API_KEY / GOOGLE_API_KEY) "
            "in akkio-fastapi/.env and restart the API process."
        )
    llm = get_llm_for_user(user_email=None, temperature=0.25)
    lc_messages: List[Any] = [SystemMessage(content=system)]
    for m in messages[-12:]:
        role = (m.get("role") or "user").lower()
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))

    response = await llm.ainvoke(lc_messages)
    return str(getattr(response, "content", response) or "").strip()
