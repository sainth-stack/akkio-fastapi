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

_SAP_TICKET_BY_ID: Dict[str, Dict[str, Any]] = {
    str(t["ticket_id"]): t for t in _SAP_SYNTHETIC_TICKETS if t.get("ticket_id")
}

# 10-ticket demo: 3 + 3 + 3 + 1 — grouped so similarity search returns same-category neighbors.
_SAP_TICKET_GROUPS: List[Dict[str, Any]] = [
    {
        "similarity_group": "fi_account_determination",
        "category": "Finance",
        "component": "Account determination",
        "ticket_ids": ["SCN-58841", "SCN-91204", "SCN-99231023", "RETMFG00869"],
    },
    {
        "similarity_group": "mm_idoc_rfc",
        "category": "Materials Management",
        "component": "IDoc",
        "ticket_ids": ["TKT-2001", "TKT-2004", "TKT-2007"],
    },
    {
        "similarity_group": "sd_bapi_jobs",
        "category": "Sales & Distribution",
        "component": "BAPI",
        "ticket_ids": ["TKT-2002", "TKT-2005", "TKT-2006"],
    },
]

_GROUP_DESCRIPTION_TAIL: Dict[str, str] = {
    "fi_account_determination": (
        "Account determination and reconciliation settings in FI/SD often cause this symptom "
        "when master data or customizing keys are incomplete."
    ),
    "mm_idoc_rfc": (
        "Materials Management interfaces (IDoc, RFC, workflow) frequently fail when plant data, "
        "authorizations, or segment mappings do not match the application posting."
    ),
    "sd_bapi_jobs": (
        "Sales & Distribution order and pricing flows fail when RFC/BAPI timeouts, user exits, "
        "or background jobs hit inconsistent pricing or incomplete document data."
    ),
}


def _expand_ticket_description(row: Dict[str, Any], *, group_key: str) -> str:
    """Ensure at least two paragraphs for RAG / similarity token overlap."""
    desc = str(row.get("description") or "").strip()
    tail = _GROUP_DESCRIPTION_TAIL.get(group_key, "")
    p1 = desc if desc else str(row.get("title") or "SAP incident")
    if "\n\n" in p1:
        return p1
    p2 = (
        tail
        or f"Operational context: {row.get('category')} / {row.get('component')} — "
        "check SAP Notes and Community threads with the same component label."
    )
    return f"{p1}\n\n{p2}"


def wants_sap_similar_tickets(prd: str, requirement: str) -> bool:
    blob = f"{prd}\n{requirement}".lower()
    return bool(
        re.search(
            r"\b(sap\b.*\b(similar|ticket|incident|similarity)|similarity tickets?|"
            r"similar ticket|find similar|service ticket|sap service|mm\b.*\bsd\b.*\bticket)",
            blob,
        )
    )


def is_sap_similarity_app(defn: Dict[str, Any]) -> bool:
    """Hosted JSON app: SAP Similarity Tickets (not generic manufacturing)."""
    meta = defn.get("meta") if isinstance(defn.get("meta"), dict) else {}
    if str(meta.get("domain") or "").lower() == "sap":
        return True
    caps = meta.get("capabilities") or []
    if isinstance(caps, list) and "sap_similar_tickets" in caps:
        return True
    bag = defn.get("data") if isinstance(defn.get("data"), dict) else {}
    if isinstance(bag.get("source_sap_tickets"), list) and bag.get("source_sap_tickets"):
        return True
    name = str(meta.get("appName") or "").lower()
    return "sap" in name and "similar" in name


def wants_two_ticket_screens_only(prd: str, requirement: str) -> bool:
    blob = f"{prd}\n{requirement}".lower()
    return bool(re.search(r"\btwo\s+screens?\b", blob)) or bool(
        re.search(r"create ticket.*similar ticket", blob)
    )


DEFAULT_SAP_TICKET_SEED = 10


def sap_dashboard_table_columns() -> List[Dict[str, str]]:
    """Dashboard table columns (KPIs + table only — no community feed UI)."""
    return [
        {"field": "ticket_id", "header": "Ticket ID", "type": "string"},
        {"field": "title", "header": "Title", "type": "string"},
        {"field": "posted_at", "header": "Posted", "type": "string"},
        {"field": "snippet", "header": "Description", "type": "string"},
        {"field": "category", "header": "Category", "type": "string"},
        {"field": "priority", "header": "Priority", "type": "string"},
        {"field": "status", "header": "Status", "type": "string"},
    ]


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
    ref_group = str(reference.get("similarity_group") or "")
    cand_group = str(candidate.get("similarity_group") or "")
    if ref_group and ref_group == cand_group:
        jaccard += 0.35
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
    """
    Build N tickets in similarity groups (3 per category for the default 10-ticket demo).
    Layout: 3× Finance, 3× Materials Management, 3× Sales & Distribution, then repeat groups.
    """
    count = max(1, min(200, int(count)))
    per_group = 3
    out: List[Dict[str, Any]] = []
    g = 0
    while len(out) < count:
        group = _SAP_TICKET_GROUPS[g % len(_SAP_TICKET_GROUPS)]
        group_key = str(group["similarity_group"])
        ids = list(group.get("ticket_ids") or [])
        take = min(per_group, count - len(out))
        for off in range(take):
            tid = ids[off % len(ids)] if ids else "TKT-0000"
            template = dict(_SAP_TICKET_BY_ID.get(tid) or _SAP_SYNTHETIC_TICKETS[0])
            row = dict(template)
            row["category"] = group["category"]
            row.setdefault("component", str(group.get("component") or ""))
            row["similarity_group"] = group_key
            row["description"] = _expand_ticket_description(row, group_key=group_key)
            if off >= len(ids):
                row["ticket_id"] = f"{tid}-V{g + 1}"
            out.append(row)
        g += 1
    return out[:count]


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
    return DEFAULT_SAP_TICKET_SEED


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
            "type": "dataTable",
            "title": "SAP tickets",
            "dataKey": "table_sap_tickets",
            "searchable": True,
            "columns": sap_dashboard_table_columns(),
        },
    ]


def _enforce_sap_single_dashboard(defn: AppDefinition) -> AppDefinition:
    """SAP apps: one screen only — Dashboard with KPIs + ticket table (no extra nav)."""
    dash = PageDef(
        id="dashboard",
        title="Dashboard",
        subtitle="Ticket KPIs and ticket list",
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
        dash = dash.model_copy(
            update={
                "blocks": sap_blocks,
                "subtitle": dash.subtitle or "Ticket KPIs and ticket list",
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
    seed_n = parse_sap_ticket_count(prd, requirement) or DEFAULT_SAP_TICKET_SEED
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


def _sap_ticket_kpis(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    open_status = {"open", "in progress", "scheduled", "pending"}
    high_pri = {"high", "critical"}
    closed_status = {"closed", "resolved"}

    def _norm(v: Any) -> str:
        return str(v or "").strip().lower()

    open_n = sum(1 for r in rows if _norm(r.get("status")) in open_status)
    high_n = sum(1 for r in rows if _norm(r.get("priority")) in high_pri)
    closed_n = sum(1 for r in rows if _norm(r.get("status")) in closed_status)
    total = len(rows)
    return {
        "open_tickets": str(open_n),
        "high_priority_tickets": str(high_n),
        "total_tickets": str(total),
        "closed_tickets": str(closed_n),
    }


def _sap_table_row(row: Dict[str, Any]) -> Dict[str, Any]:
    desc = str(row.get("description") or "")
    snippet = str(row.get("snippet") or (desc[:220] + ("…" if len(desc) > 220 else "")))
    posted = str(row.get("posted_at") or row.get("created_at") or "")
    return {
        "ticket_id": str(row.get("ticket_id") or ""),
        "title": str(row.get("title") or ""),
        "posted_at": posted,
        "snippet": snippet,
        "category": str(row.get("category") or ""),
        "priority": str(row.get("priority") or ""),
        "status": str(row.get("status") or ""),
    }


def recompute_sap_dashboard_from_tickets(defn: Dict[str, Any]) -> Dict[str, Any]:
    """KPI row + ticket table only — no inventory/quality charts."""
    out = dict(defn)
    bag = dict(out.get("data") or out.get("mock") or {})
    rows = bag.get("source_sap_tickets")
    if not isinstance(rows, list):
        rows = []
    bag["kpis_dashboard"] = _sap_ticket_kpis(rows)
    fields = [c["field"] for c in sap_dashboard_table_columns()]
    bag["table_sap_tickets"] = [_sap_table_row(r) for r in rows if isinstance(r, dict)]
    for key in list(bag.keys()):
        if key.startswith("chart_dashboard"):
            bag.pop(key, None)
    pages = []
    for page in out.get("pages") or []:
        if not isinstance(page, dict):
            continue
        if str(page.get("id") or "") == "dashboard":
            page = {
                **page,
                "blocks": sap_dashboard_blocks(),
                "subtitle": page.get("subtitle") or "Ticket KPIs and ticket list",
            }
        pages.append(page)
    out["pages"] = pages
    out["data"] = bag
    out.pop("mock", None)
    meta = dict(out.get("meta") or {})
    meta["dataSource"] = "computed_from_source"
    meta["domain"] = meta.get("domain") or "sap"
    out["meta"] = meta
    return out


def ensure_sap_similar_ticket_features_dict(
    defn: Dict[str, Any], *, prd: str = "", requirement: str = ""
) -> Dict[str, Any]:
    """Dict wrapper for enrich/RAG refresh paths."""
    try:
        model = AppDefinition.model_validate(defn)
        out = ensure_sap_similar_ticket_features(model, prd=prd, requirement=requirement)
        return out.model_dump(mode="json", exclude_none=True)
    except Exception:
        return defn
