"""Project-level SAP ticket RAG refresh."""
from __future__ import annotations

from typing import Any, Dict

from app_builder.services.app_data_rag import load_project_app_definition
from app_builder.services.data_import_service import save_data_json_to_project
from app_builder.services.sap_community_rag import enrich_definition_tickets
from app_builder.services.sap_similar_tickets import seed_sap_tickets


def refresh_project_ticket_rag(
    project_id: str,
    *,
    force: bool = False,
    prd: str = "",
    requirement: str = "",
) -> Dict[str, Any]:
    defn = load_project_app_definition(project_id)
    if not defn:
        raise ValueError("App definition not found")
    defn = seed_sap_tickets(defn, prd=prd, requirement=requirement)
    defn = enrich_definition_tickets(defn, force=force)
    meta = dict(defn.get("meta") or {})
    meta["sapRagPhase"] = "rag_complete"
    defn["meta"] = meta
    save_data_json_to_project(project_id, defn)
    bag = defn.get("data") or {}
    rows = bag.get("source_sap_tickets") or []
    rag_bag = bag.get("sap_ticket_rag") or {}
    return {
        "ticket_count": len(rows) if isinstance(rows, list) else 0,
        "rag_count": len(rag_bag) if isinstance(rag_bag, dict) else 0,
        "definition": {"data": defn.get("data"), "meta": defn.get("meta")},
    }