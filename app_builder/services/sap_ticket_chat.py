"""Similarity Tickets assistant — structured answers over source_sap_tickets."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app_builder.services.sap_similar_tickets import find_similar_tickets

_TICKET_ID_RE = re.compile(r"\b(RETMFG\d+|SCN-\d+|TKT-[A-Z0-9]+)\b", re.I)

_SOLUTION_PLAYBOOK = [
    "Applied configuration correction and completed regression validation.",
    "Cleared application cache and verified the workflow end to end.",
    "Restored the interface and replayed messages from the error queue.",
    "Updated user authorization and confirmed access with the requester.",
]

_COMPONENT_CAUSE = {
    "idoc": "IDoc processing failed due to partner profile, segment mapping, or application status 51/64.",
    "bapi": "BAPI wrapper or RFC destination timeout blocked the synchronous call.",
    "rfc": "RFC authorization or destination configuration is incomplete for the interface user.",
    "authorization": "Missing authorization object or role assignment for the executing user.",
    "workflow": "Workflow task / agent assignment does not match the approving user group.",
    "user exit": "Custom user exit logic changed standard pricing or posting behavior.",
    "background job": "Background job variant or schedule ended in CANCELLED with a functional dump.",
}


def is_sap_ticket_app(defn: Dict[str, Any]) -> bool:
    meta = defn.get("meta") or {}
    caps = meta.get("capabilities") or []
    if not isinstance(caps, list):
        caps = []
    domain = str(meta.get("domain") or "").lower()
    return domain == "sap" or "sap_similar_tickets" in caps


def _ticket_rows(defn: Dict[str, Any]) -> List[Dict[str, Any]]:
    bag = defn.get("data") or defn.get("mock") or {}
    rows = bag.get("source_sap_tickets")
    return rows if isinstance(rows, list) else []


def _last_user(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages):
        if (m.get("role") or "").lower() == "user":
            return str(m.get("content") or "").strip()
    return ""


def _sap_area(row: Dict[str, Any]) -> str:
    cat = str(row.get("category") or "").lower()
    if "finance" in cat or "fi" in cat:
        return "SAP ECC (FI)"
    if "material" in cat or "mm" in cat:
        return "SAP ECC (MM)"
    if "sales" in cat or "sd" in cat:
        return "SAP ECC (SD)"
    return "SAP ECC"


def _cause_text(row: Dict[str, Any], query: str) -> str:
    comp = str(row.get("component") or "").lower()
    for key, txt in _COMPONENT_CAUSE.items():
        if key in comp:
            return txt
    desc = str(row.get("description") or "")
    if desc:
        return f"Based on historical tickets, the likely cause is: {desc[:280]}"
    low = f"{query} {row.get('title')}".lower()
    if "account determination" in low or "reconciliation" in low:
        return (
            "The absence of the vendor reconciliation account was confirmed in MM as well as "
            "SD and FI, due to missing or incomplete account determination configuration."
        )
    return "Root cause aligns with the referenced historical incidents and master-data gaps described in similar tickets."


def _problem_description(query: str, anchor: Dict[str, Any]) -> str:
    title = str(anchor.get("title") or "").strip()
    if title:
        base = title
    else:
        base = query.strip().rstrip(".")
    low = f"{query} {title} {anchor.get('description')}".lower()
    if "account determination" in low:
        if "missing" not in base.lower() and "reconciliation" not in base.lower():
            return (
                "Invoice posting failed due to the vendor reconciliation account being missing "
                f"— related to: {query}."
            )
    if "reconciliation" in low or "invoice posting" in low:
        return "Invoice posting failed due to the vendor reconciliation account being missing."
    if len(base) > 12:
        return base if base.endswith(".") else f"{base}."
    return f"User-reported issue: {query}."


def _format_similarity_answer(
    query: str,
    anchor: Dict[str, Any],
    similar: List[Dict[str, Any]],
    *,
    suggested_block: str = "",
) -> str:
    problem = _problem_description(query, anchor)
    area = _sap_area(anchor)
    cause = _cause_text(anchor, query)
    if suggested_block.strip():
        solutions = suggested_block.strip()
    else:
        solutions = "\n".join(f"• {line}" for line in _SOLUTION_PLAYBOOK)
    ref_parts: List[str] = []
    for row in similar[:8]:
        tid = str(row.get("ticket_id") or "")
        pct = round(float(row.get("similarity_score") or 0) * 100, 2)
        ref_parts.append(f"{tid} ({pct:.2f}%)")
    refs = ", ".join(ref_parts) if ref_parts else "No close historical matches in the ticket store."

    return (
        f"**Problem Description:** {problem}\n\n"
        f"**SAP Area:** {area}\n\n"
        f"**Cause of the problem:** {cause}\n\n"
        f"**Suggested answer (AI-generated):**\n{solutions}\n\n"
        f"**Referenced Ticket IDs:** {refs}"
    )


def _answer_count(query: str, rows: List[Dict[str, Any]]) -> Optional[str]:
    low = query.lower()
    if not re.search(r"\b(how many|count|number of|total)\b", low):
        return None
    if "ticket" not in low and "incident" not in low:
        return None
    year_m = re.search(r"\b(20\d{2})\b", low)
    year = year_m.group(1) if year_m else None
    pool = rows
    if year:
        pool = [r for r in rows if year in str(r.get("created_at") or "")]
    prio = None
    if re.search(r"\bcritical\b", low):
        prio = ("critical", "high")
    elif re.search(r"\bhigh\b", low):
        prio = ("high",)
    elif re.search(r"\bmedium\b", low):
        prio = ("medium",)
    elif re.search(r"\blow\b", low):
        prio = ("low",)
    if prio:
        pool = [r for r in pool if str(r.get("priority") or "").lower() in prio]
    if re.search(r"\bopen\b", low):
        pool = [
            r
            for r in pool
            if str(r.get("status") or "").lower() in ("open", "in progress", "scheduled")
        ]
    elif re.search(r"\bclosed\b", low):
        pool = [r for r in pool if str(r.get("status") or "").lower() == "closed"]
    n = len(pool)
    label = "tickets"
    if prio and "critical" in low:
        label = "critical/high-priority tickets"
    elif prio:
        label = f"{prio[0]}-priority tickets"
    yr = f" in {year}" if year else ""
    return (
        f"**Summary:** There are **{n}** matching {label}{yr} in the historical ticket data "
        f"(total catalog: {len(rows)} tickets)."
    )


def _pick_anchor(
    query: str, rows: List[Dict[str, Any]], ticket_id: Optional[str]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if ticket_id:
        ref = next((r for r in rows if str(r.get("ticket_id") or "").upper() == ticket_id.upper()), None)
        if ref:
            similar = find_similar_tickets(ref, rows, exclude_id=ticket_id, limit=8)
            return ref, similar
    ref = {
        "title": query[:120],
        "description": query,
        "category": "",
        "component": "",
    }
    scored = find_similar_tickets(ref, rows, limit=1)
    anchor = scored[0] if scored else (rows[0] if rows else ref)
    similar = find_similar_tickets(anchor, rows, exclude_id=str(anchor.get("ticket_id") or ""), limit=8)
    if not similar:
        similar = find_similar_tickets(ref, rows, limit=8)
    return anchor, similar


def answer_sap_ticket_chat(
    defn: Dict[str, Any],
    messages: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """Return assistant message dict with structured markdown content."""
    if not is_sap_ticket_app(defn):
        return None
    query = _last_user(messages)
    if not query:
        return None
    rows = _ticket_rows(defn)
    if not rows:
        return {
            "content": (
                "No SAP tickets are loaded yet. Refresh the app or re-import data, "
                "then ask again about an incident or ticket ID."
            )
        }

    count_answer = _answer_count(query, rows)
    if count_answer:
        return {"content": count_answer}

    ticket_id = None
    id_m = _TICKET_ID_RE.search(query)
    if id_m:
        ticket_id = id_m.group(1).upper()

    anchor, similar = _pick_anchor(query, rows, ticket_id)
    suggested = str(anchor.get("suggested_answer") or "")
    if not suggested:
        bag = defn.get("data") or {}
        rag_bag = bag.get("sap_ticket_rag") if isinstance(bag.get("sap_ticket_rag"), dict) else {}
        tid = str(anchor.get("ticket_id") or "")
        rag = rag_bag.get(tid) if tid else {}
        if isinstance(rag, dict):
            suggested = str(rag.get("suggested_answer") or "")
    if not suggested:
        from app_builder.services.sap_community_rag import retrieve_ticket_rag_context

        rag_ctx = retrieve_ticket_rag_context(defn, query)
        if rag_ctx:
            suggested = rag_ctx.split("Suggested answer:")[-1].strip()[:2000]
    content = _format_similarity_answer(query, anchor, similar, suggested_block=suggested)
    return {"content": content}


SAP_TICKET_CHAT_SYSTEM = """You are the Similarity Tickets assistant for "{app_name}".
Answer using ONLY the APPLICATION DATA and SIMILARITY HINTS below.

When the user describes an error, incident, paste text, or asks about a ticket ID:
1. Use this exact section order with markdown bold headers:
   **Problem Description:**
   **SAP Area:**
   **Cause of the problem:**
   **Suggested Solution (if applicable):** (bullet list with •)
   **Referenced Ticket IDs:** (id and match % from similarity hints)
2. For count questions (e.g. how many critical tickets), give a short **Summary:** with exact counts.
3. Do not invent ticket IDs — only use IDs from the data or similarity hints.
4. Prefer SAP ECC area labels (FI, MM, SD) from ticket category.

SIMILARITY HINTS (precomputed for the latest user message):
{similarity_hints}

APPLICATION DATA:
{context}
"""
