"""Per-ticket RAG: SAP Community search (first hit) + stored chunks per ticket."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from app_builder.services.sap_community_search import (
    CommunitySearchHit,
    community_search_link,
    search_sap_community_first,
)

# Legacy topic hints only when search returns nothing useful.
_KNOWLEDGE_FALLBACK = (
    "Start from the ticket title and component (FI, MM, SD). Search SAP Community for the same "
    "error text and compare the first accepted answer with your customizing and master data.\n\n"
    "In the system, use SLG1 / ST22 / SM37 (or WE02 for IDocs) to capture the exact failing step, "
    "then align account determination, authorizations, or RFC destinations before reprocessing."
)

_MIN_RAG_PARAGRAPHS = 2


def _split_paragraphs(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(parts) >= 2:
        return parts
    # Bullet lists still count as one block — peel first block into a paragraph if needed.
    if parts and re.search(r"(?m)^\s*•", parts[0]):
        lines = [ln.strip().lstrip("•").strip() for ln in parts[0].splitlines() if ln.strip()]
        if len(lines) >= 2:
            return [lines[0], " ".join(lines[1:3])]
    if len(parts) == 1 and len(parts[0]) > 120:
        mid = len(parts[0]) // 2
        cut = parts[0].rfind(". ", 0, mid + 80)
        if cut > 40:
            return [parts[0][: cut + 1].strip(), parts[0][cut + 1 :].strip()]
    return parts


def ensure_min_paragraphs(text: str, *, minimum: int = _MIN_RAG_PARAGRAPHS, tail: str = "") -> str:
    """RAG answers and bodies should read as at least two short paragraphs."""
    paras = _split_paragraphs(text)
    component_hint = (tail or "").strip()
    generic_second = (
        "Cross-check similar incidents in your historical ticket list (same category and SAP "
        "component) before changing transports. Document the customizing path and retest in a "
        "non-production client when possible."
    )
    while len(paras) < minimum:
        if component_hint and len(paras) == 1:
            paras.append(
                f"For this {component_hint} area, validate master data and application logs, then "
                "re-run the failing transaction. " + generic_second
            )
        else:
            paras.append(generic_second)
    return "\n\n".join(paras[: max(minimum, len(paras))])


def count_paragraphs(text: str) -> int:
    return len(_split_paragraphs(text)) or (1 if (text or "").strip() else 0)


def _search_query_from_ticket(row: Dict[str, Any]) -> str:
    """Search query is driven by the ticket (title), not dashboard default search."""
    title = str(row.get("title") or "").strip()
    if title:
        comp = str(row.get("component") or "").strip()
        if comp and comp.lower() not in title.lower():
            return f"{title} {comp}"[:220]
        return title[:220]
    desc = str(row.get("description") or "").strip()
    if desc:
        return desc[:220]
    return "SAP service incident"


def _live_search_enabled() -> bool:
    raw = os.getenv("SAP_TICKET_RAG_LIVE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _ticket_narrative(row: Dict[str, Any]) -> str:
    """Two-paragraph incident context stored in RAG (independent of Community hit)."""
    title = str(row.get("title") or "SAP incident")
    desc = str(row.get("description") or "").strip()
    category = str(row.get("category") or "SAP")
    component = str(row.get("component") or "Application")
    p1 = (
        f"Incident: {title}. Reported under {category} / {component}. "
        f"{desc or 'Users see a posting or interface failure in the SAP ERP landscape.'}"
    )
    p2 = (
        f"Similarity hint: compare this ticket with others in group "
        f"'{row.get('similarity_group') or category}' sharing the same component "
        f"'{component}' and error patterns (status codes, dumps, missing master data)."
    )
    return ensure_min_paragraphs(f"{p1}\n\n{p2}", minimum=_MIN_RAG_PARAGRAPHS, tail=component)


def build_ticket_rag_record(row: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
    """Search SAP Community for this ticket's query and build RAG chunks."""
    q = _search_query_from_ticket(row)
    ticket_id = str(row.get("ticket_id") or "")
    title = str(row.get("title") or "")
    component = str(row.get("component") or "")
    hit: CommunitySearchHit = search_sap_community_first(q, live=_live_search_enabled())
    url = hit.url or community_search_link(q)
    narrative = _ticket_narrative(row)
    community_body = ensure_min_paragraphs(
        hit.body or hit.snippet or "",
        minimum=_MIN_RAG_PARAGRAPHS,
        tail=component,
    )
    suggested = ensure_min_paragraphs(
        hit.suggested_answer or _KNOWLEDGE_FALLBACK,
        minimum=_MIN_RAG_PARAGRAPHS,
        tail=component,
    )
    body = "\n\n".join(
        [
            narrative,
            (
                f"SAP Community reference — {hit.title or title}\n"
                f"Query: {q}\nURL: {url}\nSource: {hit.source}\n\n{community_body}"
            ),
            f"Suggested answer:\n{suggested}",
        ]
    )
    chunks = _chunk_text(body, max_len=900)
    return {
        "ticket_id": ticket_id,
        "search_query": q,
        "source_url": url,
        "source_title": hit.title or title,
        "status": "ready",
        "product": hit.product or row.get("product") or "SAP ERP",
        "snippet": ensure_min_paragraphs(
            hit.snippet or str(row.get("description") or "")[:400],
            minimum=1,
            tail=component,
        )[:480],
        "suggested_answer": suggested,
        "rag_source": hit.source,
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
    out = dict(row)
    rag = rag or build_ticket_rag_record(out)
    desc = str(out.get("description") or "")
    if not out.get("snippet"):
        out["snippet"] = rag.get("snippet") or (desc[:220] + ("…" if len(desc) > 220 else ""))
    out.setdefault("document_type", "Question")
    out.setdefault("product", rag.get("product") or "SAP ERP")
    out["community_url"] = rag.get("source_url") or out.get("community_url")
    out["suggested_answer"] = rag.get("suggested_answer") or out.get("suggested_answer")
    out["rag_status"] = rag.get("status") or "ready"
    out["search_query"] = rag.get("search_query")
    out["rag_source"] = rag.get("rag_source")
    return out


def _needs_rag_refresh(row: Dict[str, Any], rag: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(rag, dict):
        return True
    if rag.get("status") not in ("ready", None):
        return True
    expected_q = _search_query_from_ticket(row)
    if str(rag.get("search_query") or "") != expected_q:
        return True
    if not rag.get("chunks"):
        return True
    return False


def _defer_rag_search(defn: Dict[str, Any]) -> bool:
    meta = defn.get("meta") if isinstance(defn.get("meta"), dict) else {}
    if str(meta.get("domain") or "").lower() != "sap":
        return False
    phase = str(meta.get("sapRagPhase") or "dashboard").lower()
    return phase != "rag_complete"


def enrich_definition_tickets(defn: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
    """Run SAP search + RAG build for every ticket (one search per ticket)."""
    out = dict(defn)
    defer = _defer_rag_search(out) and not force
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
        if defer:
            pending = ensure_min_paragraphs(
                "Send step 2 in the app builder chat to search SAP Community and build "
                "per-ticket RAG (at least two paragraphs of context per ticket).",
                minimum=_MIN_RAG_PARAGRAPHS,
                tail=str(row.get("component") or ""),
            )
            rag = {
                "ticket_id": tid,
                "search_query": _search_query_from_ticket(row),
                "status": "pending",
                "suggested_answer": pending,
                "chunks": [],
            }
        elif force or _needs_rag_refresh(row, rag if isinstance(rag, dict) else None):
            rag = build_ticket_rag_record(row, force=force)
            if tid:
                rag_bag[tid] = rag
        new_rows.append(enrich_ticket_row(row, rag if isinstance(rag, dict) else None))
    bag["source_sap_tickets"] = new_rows
    bag["sap_ticket_rag"] = rag_bag
    out["data"] = bag
    out.pop("mock", None)
    return out


def retrieve_ticket_rag_context(defn: Dict[str, Any], query: str, *, top_tickets: int = 3) -> str:
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
        lines.append(f"--- Ticket {tid}: {row.get('title')} (search: {rag.get('search_query')}) ---")
        if rag.get("suggested_answer"):
            lines.append(str(rag["suggested_answer"]))
        for ch in rag.get("chunks") or []:
            if not isinstance(ch, dict):
                continue
            text = str(ch.get("text") or "")
            score = sum(1 for t in q_tokens if t in text.lower())
            if score or not q_tokens:
                lines.append(text[:1200])
        lines.append(f"Source: {rag.get('source_url')} ({rag.get('rag_source')})")
    return "\n\n".join(lines)[:14000]