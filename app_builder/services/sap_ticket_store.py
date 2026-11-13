"""Persist SAP tickets in project data.json (newest rows first)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app_builder.services.app_data_rag import load_project_app_definition
from app_builder.services.data_import_service import save_data_json_to_project
from app_builder.services.sap_similar_tickets import (
    find_similar_tickets,
    new_ticket_row,
    seed_sap_tickets_if_empty,
)

TICKET_DATA_KEY = "source_sap_tickets"


def _ticket_bag(defn: Dict[str, Any]) -> List[Dict[str, Any]]:
    bag = defn.get("data") or defn.get("mock") or {}
    rows = bag.get(TICKET_DATA_KEY)
    return list(rows) if isinstance(rows, list) else []


def list_project_tickets(project_id: str) -> List[Dict[str, Any]]:
    defn = load_project_app_definition(project_id)
    if not defn:
        return []
    defn = seed_sap_tickets_if_empty(defn)
    return _ticket_bag(defn)


def add_project_ticket(
    project_id: str,
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    defn = load_project_app_definition(project_id)
    if not defn:
        raise ValueError("App definition not found")
    defn = seed_sap_tickets_if_empty(defn)
    row = new_ticket_row(payload)
    rows = _ticket_bag(defn)
    rows.insert(0, row)
    bag = dict(defn.get("data") or {})
    bag[TICKET_DATA_KEY] = rows
    defn["data"] = bag
    defn.pop("mock", None)
    similar = find_similar_tickets(row, rows, exclude_id=row["ticket_id"])
    save_data_json_to_project(project_id, defn)
    return row, similar, defn


def similar_for_ticket_id(
    project_id: str,
    ticket_id: str,
    *,
    limit: int = 12,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list_project_tickets(project_id)
    ref = next((r for r in rows if str(r.get("ticket_id")) == ticket_id), None)
    if not ref:
        return None, []
    return ref, find_similar_tickets(ref, rows, exclude_id=ticket_id, limit=limit)
