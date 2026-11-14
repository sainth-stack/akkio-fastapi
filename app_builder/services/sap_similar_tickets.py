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
        "ticket_id": "SCN-58841",
        "title": "Account determination ID missing in master data contract account (FI-CA)",
        "description": (
            "Hello, I have a problem with the account determination ID in FI-CA. "
            "It is missing in the master data contract account and posting fails."
        ),
        "category": "Finance",
        "component": "Account determination",
        "product": "SAP ERP",
        "document_type": "Question",
        "priority": "High",
        "status": "Open",
        "reported_by": "Community user",
        "created_at": "2011-08-09 12:05",
        "posted_at": "Aug 9, 2011, 12:05 PM",
        "community_url": "https://community.sap.com/t5/forums/searchpage/tab/message?q=account+determination+fi-ca",
    },
    {
        "ticket_id": "SCN-91204",
        "title": "FI-AR Account determination",
        "description": (
            "Where should account determination be set for FI-AR postings so customer "
            "line items use the correct reconciliation accounts?"
        ),
        "category": "Finance",
        "component": "Account determination",
        "product": "SAP ERP",
        "document_type": "Question",
        "priority": "Medium",
        "status": "Open",
        "reported_by": "Community user",
        "created_at": "2013-12-13 02:55",
        "posted_at": "Dec 13, 2013, 2:55 AM",
    },
    {
        "ticket_id": "SCN-44102",
        "title": "Account Determination in Debit / Credit Memo",
        "description": (
            "We create debit and credit memos in SD; account determination does not pick "
            "the expected revenue accounts. What customizing should we review?"
        ),
        "category": "Sales & Distribution",
        "component": "Account determination",
        "product": "SAP ERP",
        "document_type": "Question",
        "priority": "Medium",
        "status": "Open",
        "reported_by": "Community user",
        "created_at": "2011-04-06 23:59",
        "posted_at": "Apr 6, 2011, 11:59 PM",
    },
    {
        "ticket_id": "SCN-99231023",
        "title": "account determination id in DOCUBATCH",
        "description": (
            "Is there an inconsistency in account determination when IDOC is processed "
            "via DOCUBATCH with regards to the FI-CA account determination?"
        ),
        "category": "Finance",
        "component": "Account determination",
        "product": "SAP ERP",
        "document_type": "Question",
        "priority": "Medium",
        "status": "Open",
        "reported_by": "Community user",
        "created_at": "2012-11-28 15:56",
        "posted_at": "Nov 28, 2012, 3:56 PM",
    },
    {
        "ticket_id": "RETMFG00869",
        "title": "Invoice posting failed — vendor reconciliation account missing",
        "description": (
            "Account determination error during FI invoice posting; vendor master lacks "
            "reconciliation account in company code 1000."
        ),
        "category": "Finance",
        "component": "Account determination",
        "product": "SAP ERP",
        "document_type": "Question",
        "priority": "High",
        "status": "Open",
        "reported_by": "Plant AP team",
        "created_at": "2025-11-02 08:15",
        "posted_at": "Nov 2, 2025, 8:15 AM",
    },
    {
        "ticket_id": "TKT-2001",
        "title": "IDoc failure in MM goods receipt",
        "description": "Inbound IDoc 740 failed with status 51 during GR posting for plant 1000.",
        "category": "Materials Management",
        "component": "IDoc",
        "product": "SAP ERP",
        "document_type": "Question",
        "priority": "High",
        "status": "Open",
        "reported_by": "A. Chen",
        "created_at": "2026-06-01 09:00",
        "posted_at": "Jun 1, 2026, 9:00 AM",
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


def parse_sap_ticket_count(prd: str, requirement: str) -> Optional[int]:
    """Read requested seed count from prompt, e.g. 'seed 10 SAP community tickets'."""
    blob = f"{prd}\n{requirement}"
    patterns = [
        r"\b(?:seed|create|generate|with|load)\s+(\d{1,3})\s+(?:synthetic\s+)?(?:sap\s+)?(?:community\s+)?tickets?\b",
        r"\b(\d{1,3})\s+(?:synthetic\s+)?(?:sap\s+)?(?:community\s+)?tickets?\b",
        r"\b(\d{1,3})\s+community\s+questions?\b",
    ]
    for pat in patterns:
        m = re.search(pat, blob, re.I)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 200:
                return n
    return None


def synthetic_tickets_for_count(count: int) -> List[Dict[str, Any]]:
    """Build N community-style tickets by cycling curated templates."""
    count = max(1, min(200, int(count)))
    base = _SAP_SYNTHETIC_TICKETS
    out: List[Dict[str, Any]] = []
    for i in range(count):
        t = dict(base[i % len(base)])
        if i >= len(base):
            t["ticket_id"] = f"SCN-{90000 + i}"
            suffix = f" (variant {i + 1})"
            t["title"] = str(t.get("title") or "SAP incident") + suffix
        out.append(t)
    return out


def resolve_sap_ticket_seed_count(defn: Dict[str, Any], *, prd: str = "", requirement: str = "") -> int:
    meta = defn.get("meta") if isinstance(defn.get("meta"), dict) else {}
    parsed = parse_sap_ticket_count(prd, requirement)
    if parsed is not None:
        return parsed
    stored = meta.get("sapTicketSeedCount")
    if stored is not None:
        try:
            return max(1, min(200, int(stored)))
        except (TypeError, ValueError):
            pass
    return len(_SAP_SYNTHETIC_TICKETS)


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


def seed_sap_tickets(
    defn: Dict[str, Any],
    *,
    prd: str = "",
    requirement: str = "",
) -> Dict[str, Any]:
    out = dict(defn)
    count = resolve_sap_ticket_seed_count(out, prd=prd, requirement=requirement)
    meta = dict(out.get("meta") or {})
    meta["sapTicketSeedCount"] = count
    out["meta"] = meta
    bag = dict(out.get("data") or out.get("mock") or {})
    key = "source_sap_tickets"
    rows = bag.get(key) if isinstance(bag.get(key), list) else []
    if not rows:
        bag[key] = synthetic_tickets_for_count(count)
        out["data"] = bag
        out.pop("mock", None)
    from app_builder.services.sap_community_rag import enrich_definition_tickets

    return enrich_definition_tickets(out)


def seed_sap_tickets_if_empty(
    defn: Dict[str, Any],
    *,
    prd: str = "",
    requirement: str = "",
) -> Dict[str, Any]:
    return seed_sap_tickets(defn, prd=prd, requirement=requirement)


def new_ticket_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    created = now.strftime("%Y-%m-%d %H:%M")
    posted = now.strftime("%b %d, %Y, %I:%M %p").replace(" 0", " ")
    desc = str(payload.get("description") or "").strip()
    title = str(payload.get("title") or "").strip()
    return {
        "ticket_id": f"TKT-{uuid4().hex[:6].upper()}",
        "title": title,
        "description": desc,
        "snippet": desc[:220] + ("…" if len(desc) > 220 else ""),
        "category": str(payload.get("category") or "General").strip(),
        "component": str(payload.get("component") or "Other").strip(),
        "product": str(payload.get("product") or "SAP ERP").strip(),
        "document_type": "Question",
        "priority": str(payload.get("priority") or "Medium").strip(),
        "status": "Open",
        "reported_by": str(payload.get("reported_by") or "App user").strip(),
        "created_at": created,
        "posted_at": posted,
        "rag_status": "pending",
    }


def sap_dashboard_blocks() -> List[Dict[str, Any]]:
    return [
        {
            "type": "kpiRow",
            "dataKey": "kpis_dashboard",
            "items": [
                {"label": "Open tickets", "valueKey": "open_tickets", "color": "warning"},
                {"label": "High priority", "valueKey": "high_priority_tickets", "color": "error"},
                {"label": "Total tickets", "valueKey": "total_tickets", "color": "primary"},
                {"label": "Closed", "valueKey": "closed_tickets", "color": "success"},
            ],
        },
        {
            "type": "sapCommunityFeed",
            "title": "Community tickets",
            "dataKey": "source_sap_tickets",
            "defaultQuery": "account determination id",
        },
    ]


def _enforce_sap_single_dashboard(defn: AppDefinition) -> AppDefinition:
    """SAP apps: one screen only — Dashboard with KPIs + Community ticket feed (no extra nav)."""
    dash = PageDef(
        id="dashboard",
        title="Dashboard",
        subtitle="SAP Community tickets — search, KPIs, and similarity assistant",
        blocks=sap_dashboard_blocks(),
    )
    defn.pages = [dash]
    defn.navigation = [
        NavItemDef(label="Dashboard", path="/", icon="Dashboard", pageId="dashboard"),
    ]
    caps = [c for c in list(defn.meta.capabilities or []) if c != "predict"]
    for c in ("sap_similar_tickets", "rag_chat"):
        if c not in caps:
            caps.append(c)
    defn.meta = defn.meta.model_copy(update={"capabilities": caps})
    return defn


def _merge_sap_dashboard(defn: AppDefinition) -> AppDefinition:
    """Default UX: dashboard KPIs + ticket table (not separate create/similar screens)."""
    pages = list(defn.pages or [])
    dash = next((p for p in pages if p.id == "dashboard"), None)
    sap_blocks = sap_dashboard_blocks()
    if dash is None:
        dash = PageDef(
            id="dashboard",
            title="Dashboard",
            subtitle="Ticket volume and recent SAP incidents",
            blocks=sap_blocks,
        )
        pages.insert(0, dash)
    else:
        blocks = list(dash.blocks or [])
        blocks = [
            sap_blocks[1]
            if isinstance(b, dict)
            and b.get("type") == "dataTable"
            and str(b.get("dataKey") or "").startswith("table_dashboard")
            else b
            for b in blocks
        ]
        has_table = any(
            isinstance(b, dict) and b.get("type") in ("dataTable", "sapCommunityFeed")
            for b in blocks
        )
        has_kpi = any(isinstance(b, dict) and b.get("type") == "kpiRow" for b in blocks)
        if not has_kpi:
            blocks = [sap_blocks[0], *blocks]
        if not has_table:
            blocks = [*blocks, sap_blocks[1]]
        dash = dash.model_copy(
            update={
                "blocks": blocks,
                "subtitle": dash.subtitle or "Ticket volume and recent SAP incidents",
            }
        )
        pages = [dash if p.id == "dashboard" else p for p in pages]
    nav = list(defn.navigation or [])
    if not any(n.pageId == "dashboard" for n in nav):
        nav.insert(
            0,
            NavItemDef(label="Dashboard", path="/", icon="Dashboard", pageId="dashboard"),
        )
    defn.pages = pages
    defn.navigation = nav
    return defn


def ensure_sap_similar_ticket_features(
    defn: AppDefinition, *, prd: str = "", requirement: str = ""
) -> AppDefinition:
    meta_domain = str(defn.meta.domain or "").lower()
    if meta_domain != "sap" and not wants_sap_similar_tickets(prd, requirement):
        return defn

    caps = list(defn.meta.capabilities or [])
    if "sap_similar_tickets" not in caps:
        caps.append("sap_similar_tickets")
    if "rag_chat" not in caps:
        caps.append("rag_chat")
    seed_n = parse_sap_ticket_count(prd, requirement) or len(_SAP_SYNTHETIC_TICKETS)
    rag_phase = str(defn.meta.sapRagPhase or "dashboard")
    defn.meta = defn.meta.model_copy(
        update={
            "capabilities": caps,
            "domain": "sap",
            "dataModel": "source_compute_v1",
            "sapTicketSeedCount": seed_n,
            "sapRagPhase": rag_phase,
            "description": defn.meta.description
            or "SAP Similarity Tickets — dashboard and assistant over historical incidents.",
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

    defn = _enforce_sap_single_dashboard(defn)
    return defn
