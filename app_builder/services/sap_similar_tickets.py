"""SAP service management — create tickets + find similar historical incidents."""
from __future__ import annotations

import re
import re as re_module
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from app_builder.schemas.app_definition import AppDefinition, AppMeta, NavItemDef, PageDef

_TOKEN = re_module.compile(r"[a-z0-9]{2,}")

_SAP_SYNTHETIC_TICKETS: List[Dict[str, Any]] = [
    {
        "ticket_id": "TKT-2001",
        "title": "IDoc failure in MM goods receipt",
        "description": "Inbound IDoc 740 failed with status 51 during GR posting for plant 1000.",
        "category": "Materials Management",
        "component": "IDoc",
        "priority": "High",
        "status": "Open",
        "reported_by": "A. Chen",
        "created_at": "2026-06-01 09:00",
    },
    {
        "ticket_id": "TKT-2002",
        "title": "BAPI timeout on sales order create",
        "description": "SD order create via BAPI hangs after 120s; short dump in ZSD_ORDER_WRAP.",
        "category": "Sales & Distribution",
        "component": "BAPI",
        "priority": "High",
        "status": "In progress",
        "reported_by": "R. Patel",
        "created_at": "2026-06-02 11:30",
    },
    {
        "ticket_id": "TKT-2003",
        "title": "Similar IDoc 51 error on invoice posting",
        "description": "FI invoice IDoc fails validation; same segment E1EDK03 as prior MM issue.",
        "category": "Finance",
        "component": "IDoc",
        "priority": "Medium",
        "status": "Open",
        "reported_by": "S. Okafor",
        "created_at": "2026-06-03 14:00",
    },
    {
        "ticket_id": "TKT-2004",
        "title": "Authorization missing for ZMM_RFC user",
        "description": "RFC call ZMM_STOCK_GET returns auth error for interface user SAP_MM_RFC.",
        "category": "Materials Management",
        "component": "RFC",
        "priority": "Medium",
        "status": "Closed",
        "reported_by": "L. Nguyen",
        "created_at": "2026-06-04 08:45",
    },
    {
        "ticket_id": "TKT-2005",
        "title": "Batch job SM37 failure — pricing update",
        "description": "Background job for SD pricing update ends with CANCELLED; dump in SAPLV60A.",
        "category": "Sales & Distribution",
        "component": "Background job",
        "priority": "Low",
        "status": "Scheduled",
        "reported_by": "J. Martinez",
        "created_at": "2026-06-05 16:20",
    },
    {
        "ticket_id": "TKT-2006",
        "title": "User exit pricing rounding mismatch",
        "description": "Custom user exit on VA01 changes net value by 0.01 vs standard SD pricing.",
        "category": "Sales & Distribution",
        "component": "User exit",
        "priority": "Medium",
        "status": "Open",
        "reported_by": "A. Chen",
        "created_at": "2026-06-06 10:10",
    },
    {
        "ticket_id": "TKT-2007",
        "title": "PO release workflow stuck in SAP inbox",
        "description": "MM purchase order workflow item not visible for approver group MM_MGR.",
        "category": "Materials Management",
        "component": "Workflow",
        "priority": "High",
        "status": "In progress",
        "reported_by": "R. Patel",
        "created_at": "2026-06-07 13:00",
    },
    {
        "ticket_id": "TKT-2008",
        "title": "FI document parking authorization",
        "description": "FBV0 parking fails with insufficient authorization object F_BKPF_BUK.",
        "category": "Finance",
        "component": "Authorization",
        "priority": "Medium",
        "status": "Closed",
        "reported_by": "S. Okafor",
        "created_at": "2026-06-08 09:30",
    },
]


def wants_sap_similar_tickets(prd: str, requirement: str) -> bool:
    blob = f"{prd}\n{requirement}".lower()
    return bool(
        re.search(
            r"\b(sap\b.*\b(similar|ticket|incident)|similar ticket|find similar|"
            r"service ticket|sap service|mm\b.*\bsd\b.*\bticket)",
            blob,
        )
    )


def wants_two_ticket_screens_only(prd: str, requirement: str) -> bool:
    blob = f"{prd}\n{requirement}".lower()
    return bool(re.search(r"\btwo\s+screens?\b", blob)) or bool(
        re.search(r"create ticket.*similar ticket", blob)
    )


def sap_ticket_columns() -> List[Dict[str, str]]:
    return [
        {"field": "ticket_id", "header": "Ticket ID", "type": "string"},
        {"field": "title", "header": "Title", "type": "string"},
        {"field": "description", "header": "Description", "type": "string"},
        {"field": "category", "header": "Category", "type": "string"},
        {"field": "component", "header": "SAP component", "type": "string"},
        {"field": "priority", "header": "Priority", "type": "string"},
        {"field": "status", "header": "Status", "type": "string"},
        {"field": "reported_by", "header": "Reported by", "type": "string"},
        {"field": "created_at", "header": "Created at", "type": "datetime"},
    ]


def _tokens(text: str) -> set:
    return set(_TOKEN.findall((text or "").lower()))


def similarity_score(reference: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    ref_txt = " ".join(
        str(reference.get(k) or "")
        for k in ("title", "description", "category", "component")
    )
    cand_txt = " ".join(
        str(candidate.get(k) or "")
        for k in ("title", "description", "category", "component")
    )
    a, b = _tokens(ref_txt), _tokens(cand_txt)
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    union = len(a | b)
    jaccard = overlap / union if union else 0.0
    if str(reference.get("category") or "") == str(candidate.get("category") or ""):
        jaccard += 0.15
    if str(reference.get("component") or "") == str(candidate.get("component") or ""):
        jaccard += 0.2
    return round(min(1.0, jaccard), 4)


def find_similar_tickets(
    reference: Dict[str, Any],
    pool: List[Dict[str, Any]],
    *,
    limit: int = 12,
    exclude_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    ex = exclude_id or str(reference.get("ticket_id") or "")
    for row in pool:
        tid = str(row.get("ticket_id") or "")
        if ex and tid == ex:
            continue
        score = similarity_score(reference, row)
        if score <= 0:
            continue
        scored.append((score, {**row, "similarity_score": score}))
    scored.sort(key=lambda x: -x[0])
    return [row for _, row in scored[:limit]]


def seed_sap_tickets_if_empty(defn: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defn)
    bag = dict(out.get("data") or out.get("mock") or {})
    key = "source_sap_tickets"
    rows = bag.get(key) if isinstance(bag.get(key), list) else []
    if not rows:
        bag[key] = [dict(t) for t in _SAP_SYNTHETIC_TICKETS]
        out["data"] = bag
        out.pop("mock", None)
    return out


def new_ticket_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    return {
        "ticket_id": f"TKT-{uuid4().hex[:6].upper()}",
        "title": str(payload.get("title") or "").strip(),
        "description": str(payload.get("description") or "").strip(),
        "category": str(payload.get("category") or "General").strip(),
        "component": str(payload.get("component") or "Other").strip(),
        "priority": str(payload.get("priority") or "Medium").strip(),
        "status": "Open",
        "reported_by": str(payload.get("reported_by") or "App user").strip(),
        "created_at": now,
    }


def ensure_sap_similar_ticket_features(
    defn: AppDefinition, *, prd: str = "", requirement: str = ""
) -> AppDefinition:
    if not wants_sap_similar_tickets(prd, requirement):
        return defn

    caps = list(defn.meta.capabilities or [])
    if "sap_similar_tickets" not in caps:
        caps.append("sap_similar_tickets")
    defn.meta = defn.meta.model_copy(
        update={
            "capabilities": caps,
            "domain": "sap",
            "dataModel": "source_compute_v1",
            "description": defn.meta.description
            or "SAP service tickets — create incidents and find similar historical tickets.",
        }
    )
    datasets = list(defn.meta.sourceDatasets or [])
    if not any(str(d.get("dataKey")) == "source_sap_tickets" for d in datasets if isinstance(d, dict)):
        datasets = [
            {
                "id": "sap_tickets",
                "dataKey": "source_sap_tickets",
                "title": "SAP tickets",
                "description": "Historical and newly created service tickets.",
                "columns": sap_ticket_columns(),
            }
        ]
        defn.meta = defn.meta.model_copy(update={"sourceDatasets": datasets})

    create_page = PageDef(
        id="create_ticket",
        title="Create Ticket",
        subtitle="Log a new SAP service incident",
        blocks=[{"type": "text", "content": "", "hidden": True}],
    )
    similar_page = PageDef(
        id="similar_tickets",
        title="Similar Tickets",
        subtitle="Find historical tickets like the selected incident",
        blocks=[{"type": "text", "content": "", "hidden": True}],
    )
    nav_create = NavItemDef(
        label="Create Ticket", path="/create-ticket", icon="AddCircle", pageId="create_ticket"
    )
    nav_similar = NavItemDef(
        label="Similar Tickets", path="/similar-tickets", icon="Search", pageId="similar_tickets"
    )

    if wants_two_ticket_screens_only(prd, requirement):
        defn.pages = [create_page, similar_page]
        defn.navigation = [nav_create, nav_similar]
        return defn

    nav = list(defn.navigation or [])
    pages = list(defn.pages or [])
    if not any(n.pageId == "create_ticket" for n in nav):
        nav.append(nav_create)
        pages.append(create_page)
    if not any(n.pageId == "similar_tickets" for n in nav):
        nav.append(nav_similar)
        pages.append(similar_page)
    defn.navigation = nav
    defn.pages = pages
    return defn
