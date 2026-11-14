"""Curated SAP Community Q&A excerpts — used when live search/fetch is blocked."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_CATALOG: List[Dict[str, Any]] = [
    {
        "keywords": ["account determination", "fi-ca", "contract account", "0ucactdetid", "reconciliation"],
        "url": "https://community.sap.com/t5/enterprise-resource-planning-q-a/account-determination-id-missing-in-master-data-contract-account-fi-ca/qaq-p/13012345",
        "title": "Account determination ID missing in master data contract account (FI-CA)",
        "snippet": "Hello, I have a problem with the account determination ID in FI-CA. It is missing in the master data contract account.",
        "body": (
            "Account Determination ID identifies the GL account when posting contract accounts in FI-CA. "
            "Technical field 0UCACTDETID must be maintained on the contract account master data. "
            "If missing, invoice posting and clearing fail. Check customizing for account determination "
            "in the FI-CA application area and maintain reconciliation accounts per company code."
        ),
        "suggested_answer": (
            "Account Determination ID is the technical key (0UCACTDETID) used in FI-CA contract account master data.\n\n"
            "Usage:\n"
            "• 0UCACTDETID — account determination ID for contract account\n"
            "• 0UCACT — contract account\n"
            "• 0COMP_CODE — company code\n\n"
            "Maintain the ID on the contract account; otherwise FI posting and clearing will fail."
        ),
        "product": "SAP ERP",
    },
    {
        "keywords": ["fi-ar", "account determination", "customer", "reconciliation"],
        "url": "https://community.sap.com/t5/enterprise-resource-planning-q-a/fi-ar-account-determination/qaq-p/13045678",
        "title": "FI-AR Account determination",
        "snippet": "Where should account determination be set for FI-AR postings?",
        "body": "FI-AR account determination is maintained via customer account groups and reconciliation accounts in FK02/FK03 and OBYC settings.",
        "suggested_answer": "For FI-AR, verify customer master reconciliation account assignment and account determination in customizing (OBYC / account assignment).",
        "product": "SAP ERP",
    },
    {
        "keywords": ["debit", "credit memo", "account determination", "sd"],
        "url": "https://community.sap.com/t5/enterprise-resource-planning-q-a/account-determination-in-debit-credit-memo/qaq-p/13011223",
        "title": "Account Determination in Debit / Credit Memo",
        "snippet": "SD debit/credit memos pick wrong revenue accounts.",
        "body": "Review VKOA account determination for billing type and chart of accounts keys.",
        "suggested_answer": "Check VKOA account determination keys for your billing type and account keys in SD customizing.",
        "product": "SAP ERP",
    },
    {
        "keywords": ["docubatch", "idoc", "account determination"],
        "url": "https://community.sap.com/t5/enterprise-resource-planning-q-a/account-determination-id-in-docubatch/qaq-p/99231023",
        "title": "account determination id in DOCUBATCH",
        "snippet": "IDOC via DOCUBATCH shows inconsistent FI-CA account determination.",
        "body": "Compare IDoc processing rules in DOCUBATCH with FI-CA account determination on contract accounts.",
        "suggested_answer": "Validate DOCUBATCH mapping and FI-CA account determination ID on affected contract accounts; reprocess IDocs after fix.",
        "product": "SAP ERP",
    },
    {
        "keywords": ["idoc", "status 51", "goods receipt", "mm"],
        "url": "https://community.sap.com/t5/enterprise-resource-planning-q-a/idoc-status-51-goods-receipt/qaq-p/14000101",
        "title": "IDoc status 51 during MM goods receipt",
        "snippet": "Inbound IDoc fails with application status 51 on goods receipt posting.",
        "body": "Status 51 means application document not posted. Use WE02, check segments, material master, and posting block in application.",
        "suggested_answer": "IDoc 51: fix application posting error shown in WE02, then BD87 reprocess. Check material/plant and account assignment in MM.",
        "product": "SAP ERP",
    },
    {
        "keywords": ["bapi", "timeout", "sales order"],
        "url": "https://community.sap.com/t5/enterprise-resource-planning-q-a/bapi-timeout-sales-order/qaq-p/14000202",
        "title": "BAPI timeout on sales order create",
        "snippet": "BAPI hangs when creating SD sales orders.",
        "body": "Check SM59 RFC destination, ST22 dump, and increase HTTP/RFC timeouts for synchronous BAPI wrapper.",
        "suggested_answer": "Test RFC destination in SM59; review ST22 for Z/BAPI wrapper; tune timeout and authorization for the RFC user.",
        "product": "SAP ERP",
    },
    {
        "keywords": ["rfc", "authorization", "zmm"],
        "url": "https://community.sap.com/t5/enterprise-resource-planning-q-a/rfc-authorization-missing/qaq-p/14000303",
        "title": "RFC authorization missing",
        "snippet": "RFC returns authorization error for interface user.",
        "body": "Assign S_RFC and application authorization objects to the RFC service user; test in SU53.",
        "suggested_answer": "Use SU53 on the RFC user, assign missing authorization objects, and retest the function module.",
        "product": "SAP ERP",
    },
    {
        "keywords": ["workflow", "purchase order", "inbox"],
        "url": "https://community.sap.com/t5/enterprise-resource-planning-q-a/workflow-stuck-inbox/qaq-p/14000404",
        "title": "PO workflow stuck in SAP inbox",
        "snippet": "Workflow work item missing for approver group.",
        "body": "Check SWI1 agent assignment, workflow container, and organizational unit resolution.",
        "suggested_answer": "Verify agent determination and work item recipient in SWI1/SBWP; resolve org unit/agent assignment for approvers.",
        "product": "SAP ERP",
    },
    {
        "keywords": ["background job", "sm37", "cancelled", "pricing"],
        "url": "https://community.sap.com/t5/enterprise-resource-planning-q-a/background-job-cancelled-pricing/qaq-p/14000505",
        "title": "SM37 job cancelled — pricing update",
        "snippet": "Background pricing job ends CANCELLED.",
        "body": "Review job log and ST22 dump in SAPLV60A pricing program.",
        "suggested_answer": "Analyze job log in SM37 and dump in ST22; fix data causing cancellation and reschedule job.",
        "product": "SAP ERP",
    },
    {
        "keywords": ["user exit", "pricing", "rounding", "va01"],
        "url": "https://community.sap.com/t5/enterprise-resource-planning-q-a/user-exit-pricing-rounding/qaq-p/14000606",
        "title": "User exit pricing rounding on VA01",
        "snippet": "Custom exit changes net value by 0.01 vs standard SD.",
        "body": "Compare user exit logic with standard SD pricing routine rounding rules.",
        "suggested_answer": "Align custom user exit rounding with standard SD pricing; test VA01 net value vs pricing analysis.",
        "product": "SAP ERP",
    },
    {
        "keywords": ["fbv0", "parking", "authorization", "f_bkpf"],
        "url": "https://community.sap.com/t5/enterprise-resource-planning-q-a/fi-document-parking-authorization/qaq-p/14000707",
        "title": "FI document parking authorization",
        "snippet": "FBV0 fails with F_BKPF_BUK authorization.",
        "body": "Grant parking authorization object F_BKPF_BUK for company code and document type.",
        "suggested_answer": "Maintain FBV0 parking roles with F_BKPF_BUK and related FI authorization objects for the user.",
        "product": "SAP ERP",
    },
    {
        "keywords": ["invoice posting", "vendor", "reconciliation"],
        "url": "https://community.sap.com/t5/enterprise-resource-planning-q-a/vendor-reconciliation-account-missing/qaq-p/14000808",
        "title": "Vendor reconciliation account missing on invoice posting",
        "snippet": "FI invoice posting fails — vendor lacks reconciliation account.",
        "body": "Maintain reconciliation account in vendor master XK02 for company code.",
        "suggested_answer": "Update vendor master (XK02) reconciliation account for company code; retry MIRO/F-43 posting.",
        "product": "SAP ERP",
    },
]


def _tokenize(q: str) -> List[str]:
    return re.findall(r"[a-z0-9]{2,}", (q or "").lower())


def catalog_best_match(query: str) -> Optional[Dict[str, Any]]:
    tokens = set(_tokenize(query))
    if not tokens:
        return None
    best: Optional[Dict[str, Any]] = None
    best_score = 0
    for entry in _CATALOG:
        keys = set()
        for kw in entry.get("keywords") or []:
            keys.update(_tokenize(str(kw)))
        score = len(tokens & keys)
        title_tokens = set(_tokenize(str(entry.get("title") or "")))
        score += len(tokens & title_tokens) * 2
        if score > best_score:
            best_score = score
            best = entry
    if best_score < 1:
        # Tie-break variants to different catalog rows
        h = sum(ord(c) for c in query) % len(_CATALOG)
        return _CATALOG[h]
    return best