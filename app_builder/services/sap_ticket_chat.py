"""Similarity Tickets assistant — LLM answers grounded in per-ticket RAG."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app_builder.services.sap_similar_tickets import find_similar_tickets

_TICKET_ID_RE = re.compile(
    r"\b(SAP-\d{4}|RETMFG\d+|SCN-\d+|TKT-[A-Z0-9]+)\b",
    re.I,
)

_REFERENCED_LIMIT = 3
_SIMILAR_OTHERS = 2


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
    if not isinstance(rows, list):
        return []
    return sorted(rows, key=lambda r: int((r or {}).get("ticket_seq") or 0))


def _last_user(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages):
        if (m.get("role") or "").lower() == "user":
            return str(m.get("content") or "").strip()
    return ""


def _resolve_row_by_ticket_token(rows: List[Dict[str, Any]], token: str) -> Optional[Dict[str, Any]]:
    t = (token or "").strip().upper()
    if not t:
        return None
    for row in rows:
        if str(row.get("ticket_id") or "").upper() == t:
            return row
        if str(row.get("external_ref") or "").upper() == t:
            return row
    return None


def _extract_ticket_id(query: str, rows: List[Dict[str, Any]]) -> Optional[str]:
    id_m = _TICKET_ID_RE.search(query or "")
    if not id_m:
        return None
    row = _resolve_row_by_ticket_token(rows, id_m.group(1))
    if row:
        return str(row.get("ticket_id") or "")
    return id_m.group(1).upper()


def _is_placeholder_solution(text: str) -> bool:
    low = (text or "").lower()
    return "step 2" in low or "app builder chat" in low


def _rag_solution(defn: Dict[str, Any], row: Dict[str, Any]) -> str:
    suggested = str(row.get("suggested_answer") or "").strip()
    if suggested and not _is_placeholder_solution(suggested):
        return suggested
    bag = defn.get("data") or {}
    rag_bag = bag.get("sap_ticket_rag") if isinstance(bag.get("sap_ticket_rag"), dict) else {}
    tid = str(row.get("ticket_id") or "")
    ext = str(row.get("external_ref") or "")
    for key in (tid, ext):
        if not key:
            continue
        rag = rag_bag.get(key)
        if isinstance(rag, dict):
            ans = str(rag.get("suggested_answer") or "").strip()
            if ans and not _is_placeholder_solution(ans):
                return ans
    from app_builder.services.sap_community_rag import build_ticket_rag_record

    built = build_ticket_rag_record(row, force=True)
    return str(built.get("suggested_answer") or "").strip()


def _pick_anchor(
    query: str, rows: List[Dict[str, Any]], ticket_id: Optional[str]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if ticket_id:
        ref = _resolve_row_by_ticket_token(rows, ticket_id) or next(
            (r for r in rows if str(r.get("ticket_id") or "").upper() == ticket_id.upper()),
            None,
        )
        if ref:
            similar = find_similar_tickets(
                ref,
                rows,
                exclude_id=str(ref.get("ticket_id") or ""),
                limit=_SIMILAR_OTHERS,
            )
            return ref, similar
    ref = {
        "title": query[:120],
        "description": query,
        "category": "",
        "component": "",
    }
    scored = find_similar_tickets(ref, rows, limit=1)
    anchor = scored[0] if scored else (rows[0] if rows else ref)
    similar = find_similar_tickets(
        anchor,
        rows,
        exclude_id=str(anchor.get("ticket_id") or ""),
        limit=_SIMILAR_OTHERS,
    )
    return anchor, similar


def _format_match_line(row: Dict[str, Any], pct: float) -> str:
    tid = str(row.get("ticket_id") or "")
    ext = str(row.get("external_ref") or "")
    title = str(row.get("title") or "")[:120]
    extra = f" (community ref {ext})" if ext and ext != tid else ""
    return f"- {tid}{extra} — {pct:.2f}% — {title}"


def build_sap_similarity_hints(
    defn: Dict[str, Any],
    messages: List[Dict[str, str]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """Anchor row, up to 2 similar rows, and hint block for the LLM."""
    query = _last_user(messages)
    rows = _ticket_rows(defn)
    ticket_id = _extract_ticket_id(query, rows)
    anchor, similar = _pick_anchor(query, rows, ticket_id)

    lines: List[str] = [
        "Rules for **Referenced Ticket IDs:** list exactly 3 tickets when possible.",
        "• Line 1 MUST be the primary (queried) ticket at 100.00%.",
        "• Lines 2–3 are the next best matches from SIMILAR TICKETS below (do not invent IDs).",
        "",
        f"USER MESSAGE: {query}",
        "",
        "PRIMARY TICKET (100.00% match):",
        _format_match_line(anchor, 100.0),
        f"Category: {anchor.get('category')} | Component: {anchor.get('component')}",
        f"Title: {anchor.get('title')}",
        "",
        "RAG SOLUTION FOR PRIMARY TICKET (quote verbatim in **Suggested Solution**):",
        _rag_solution(defn, anchor) or "(Run Step 2 in builder to load Community solutions.)",
        "",
        "SIMILAR TICKETS (use for lines 2–3 only):",
    ]
    for row in similar:
        pct = min(99.99, round(float(row.get("similarity_score") or 0) * 100, 2))
        lines.append(_format_match_line(row, pct))
        lines.append(f"  RAG excerpt: {_rag_solution(defn, row)[:400]}")
    return anchor, similar, "\n".join(lines)


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
    return f"**Summary:** There are **{n}** matching tickets in the historical store (total **{len(rows)}**)."


SAP_TICKET_CHAT_SYSTEM = """You are the Similarity Tickets assistant for "{app_name}".
Answer using ONLY the RAG solutions and ticket hints below — do not invent SAP steps or ticket IDs.

When the user describes an error, pastes a ticket ID (e.g. SAP-0001 or SCN-58841), or asks about an incident:
1. Use this exact section order with markdown bold headers:
   **Problem Description:**
   **SAP Area:**
   **Cause of the problem:**
   **Suggested Solution (if applicable):** (use bullet list with • when multiple steps)
   **Referenced Ticket IDs:** (exactly 3 entries when data allows: primary at 100.00%, then top 2 similar)
2. **Suggested Solution** must copy the RAG SOLUTION text for the primary ticket (verbatim; no "Step 2" placeholder).
3. **Cause of the problem** must be a short analyst summary — do NOT paste the user's greeting ("Hello, I have a problem…").
4. For count questions, output only **Summary:** with counts.
5. SAP area labels: SAP ECC (FI), SAP ECC (MM), or SAP ECC (SD) from ticket category.

SIMILARITY HINTS:
{similarity_hints}

TICKET RAG CONTEXT:
{context}
"""