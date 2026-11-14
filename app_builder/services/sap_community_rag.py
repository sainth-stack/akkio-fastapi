"""Per-ticket RAG chunks and SAP Community–style suggested answers."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Curated community-style knowledge (matches common search.sap.com / community Q&A patterns).
_KNOWLEDGE_BY_TOPIC: List[Tuple[re.Pattern, Dict[str, str]]] = [
    (
        re.compile(r"account determination|reconciliation|0ucactdetid|fi-ca", re.I),
        {
            "suggested_answer": (
                "**Account Determination ID** identifies the GL account used when posting "
                "contract accounts in FI-CA (Contract Accounts Receivable and Payable).\n\n"
                "**Technical name:** 0UCACTDETID — use Account determination ID in the master data "
                "contract account.\n\n"
                "**Usage:**\n"
                "• 0UCACTDETID: Account determination ID for contract account\n"
                "• 0UCACT: Contract account\n"
                "• 0COMP_CODE: Company code\n\n"
                "If the ID is missing in the master data contract account, invoice posting and "
                "clearing can fail until determination is maintained in customizing (OBYC / "
                "account determination for the application area)."
            ),
            "product": "SAP ERP",
            "source_hint": "community.sap.com — FI-CA account determination",
        },
    ),
    (
        re.compile(r"idoc|status 51|goods receipt", re.I),
        {
            "suggested_answer": (
                "**IDoc status 51** (application document not posted) usually means SAP could not "
                "post the inbound document.\n\n"
                "**Check:** WE02/WE05 for segment errors, partner profile, message type, and "
                "posting status in the application document.\n\n"
                "**Typical fixes:** Reprocess after correcting master data; fix account assignment "
                "in MM; release IDoc in BD87 after clearing error in application log."
            ),
            "product": "SAP ERP",
            "source_hint": "community.sap.com — IDoc MM goods receipt",
        },
    ),
    (
        re.compile(r"bapi|rfc|timeout", re.I),
        {
            "suggested_answer": (
                "**BAPI / RFC timeouts** often trace to SM59 destination configuration, "
                "network latency, or user authorization for the RFC service user.\n\n"
                "Verify destination test in SM59, increase client timeout if appropriate, and "
                "check short dump in ST22 for the wrapper function module."
            ),
            "product": "SAP ERP",
            "source_hint": "community.sap.com — BAPI RFC",
        },
    ),
]


def _search_query_from_ticket(row: Dict[str, Any]) -> str:
    parts = [
        str(row.get("title") or ""),
        str(row.get("description") or ""),
        str(row.get("component") or ""),
    ]
    q = " ".join(p for p in parts if p).strip()
    return q[:200] or "SAP service incident"


def _match_knowledge(query: str) -> Dict[str, str]:
    for pat, payload in _KNOWLEDGE_BY_TOPIC:
        if pat.search(query):
            return dict(payload)
    return {
        "suggested_answer": (
            "Search SAP Community for threads matching this incident. Maintain master data and "
            "customizing for the application area (FI, MM, SD) and review application logs "
            "(SM37, SLG1, ST22) for the failing step."
        ),
        "product": "SAP ERP",
        "source_hint": "community.sap.com",
    }


def build_ticket_rag_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Build RAG record for one ticket (offline curated + query metadata)."""
    q = _search_query_from_ticket(row)
    kb = _match_knowledge(q)
    ticket_id = str(row.get("ticket_id") or "")
    title = str(row.get("title") or "")
    url = str(row.get("community_url") or "").strip()
    if not url:
        slug = re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")[:48] or "sap-question"
        url = f"https://community.sap.com/t5/forums/searchpage/tab/message?advanced=false&allow_punctuation=true&q={slug}"

    body = (
        f"Title: {title}\n"
        f"Description: {row.get('description')}\n"
        f"Product: {row.get('product') or kb.get('product')}\n"
        f"Category: {row.get('category')}\n"
        f"Component: {row.get('component')}\n\n"
        f"Suggested answer:\n{kb.get('suggested_answer')}"
    )
    chunks = _chunk_text(body, max_len=900)
    return {
        "ticket_id": ticket_id,
        "search_query": q,
        "source_url": url,
        "source_title": title,
        "status": "ready",
        "product": kb.get("product") or row.get("product") or "SAP ERP",
        "suggested_answer": kb.get("suggested_answer"),
        "chunks": [{"id": f"c{i}", "text": t} for i, t in enumerate(chunks)],
    }


def _chunk_text(text: str, *, max_len: int = 900) -> List[str]:
    text = text.strip()
    if len(text) <= max_len:
        return [text] if text else []
    parts: List[str] = []
    start = 0
    while start < len(text):
        parts.append(text[start : start + max_len])
        start += max_len
    return parts


def enrich_ticket_row(row: Dict[str, Any], rag: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Add community display fields + link RAG."""
    out = dict(row)
    rag = rag or build_ticket_rag_record(out)
    desc = str(out.get("description") or "")
    if not out.get("snippet"):
        out["snippet"] = desc[:220] + ("…" if len(desc) > 220 else "")
    out.setdefault("document_type", "Question")
    out.setdefault("product", rag.get("product") or "SAP ERP")
    out["community_url"] = out.get("community_url") or rag.get("source_url")
    out["suggested_answer"] = rag.get("suggested_answer") or out.get("suggested_answer")
    out["rag_status"] = rag.get("status") or "ready"
    out["search_query"] = rag.get("search_query")
    return out


def enrich_definition_tickets(defn: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure every ticket has RAG + sap_ticket_rag bag."""
    out = dict(defn)
    bag = dict(out.get("data") or out.get("mock") or {})
    rows = bag.get("source_sap_tickets")
    if not isinstance(rows, list):
        return out
    rag_bag: Dict[str, Any] = dict(bag.get("sap_ticket_rag") or {})
    new_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        tid = str(row.get("ticket_id") or "")
        rag = rag_bag.get(tid) if tid else None
        if not isinstance(rag, dict) or rag.get("status") != "ready":
            rag = build_ticket_rag_record(row)
            if tid:
                rag_bag[tid] = rag
        new_rows.append(enrich_ticket_row(row, rag))
    bag["source_sap_tickets"] = new_rows
    bag["sap_ticket_rag"] = rag_bag
    out["data"] = bag
    out.pop("mock", None)
    return out


def retrieve_ticket_rag_context(defn: Dict[str, Any], query: str, *, top_tickets: int = 3) -> str:
    """Lexical retrieval over per-ticket RAG chunks for chat."""
    from app_builder.services.sap_similar_tickets import find_similar_tickets

    bag = defn.get("data") or defn.get("mock") or {}
    rows = bag.get("source_sap_tickets") if isinstance(bag.get("source_sap_tickets"), list) else []
    rag_bag = bag.get("sap_ticket_rag") if isinstance(bag.get("sap_ticket_rag"), dict) else {}
    ref = {"title": query, "description": query, "category": "", "component": ""}
    picks = find_similar_tickets(ref, rows, limit=top_tickets)
    lines: List[str] = []
    q_tokens = set(re.findall(r"[a-z0-9]{2,}", query.lower()))
    for row in picks:
        tid = str(row.get("ticket_id") or "")
        rag = rag_bag.get(tid) if tid else {}
        if not isinstance(rag, dict):
            rag = build_ticket_rag_record(row)
        lines.append(f"--- Ticket {tid}: {row.get('title')} ---")
        if rag.get("suggested_answer"):
            lines.append(str(rag["suggested_answer"]))
        for ch in rag.get("chunks") or []:
            if not isinstance(ch, dict):
                continue
            text = str(ch.get("text") or "")
            score = sum(1 for t in q_tokens if t in text.lower())
            if score or not q_tokens:
                lines.append(text[:1200])
        lines.append(f"Source: {rag.get('source_url')}")
    return "\n\n".join(lines)[:14000]