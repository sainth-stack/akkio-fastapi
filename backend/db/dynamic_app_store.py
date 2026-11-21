"""MongoDB storage for generated-app CRUD (replaces per-project SQLite)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pymongo import ASCENDING, ReturnDocument

from db.mongodb import get_collection, next_seq, utc_now

_ROWS = "dynamic_app_rows"


def _rows():
    return get_collection(_ROWS)


def ensure_indexes() -> None:
    _rows().create_index(
        [("project_id", ASCENDING), ("table", ASCENDING), ("id", ASCENDING)],
        unique=True,
    )


def _public(doc: dict) -> dict:
    out: dict[str, Any] = {}
    for k, v in doc.items():
        if k in ("_id", "project_id", "table"):
            continue
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def list_items(project_id: str, table: str) -> list[dict]:
    cur = _rows().find({"project_id": project_id, "table": table}).sort("id", ASCENDING)
    return [_public(d) for d in cur]


def get_item(project_id: str, table: str, item_id: int) -> Optional[dict]:
    doc = _rows().find_one({"project_id": project_id, "table": table, "id": item_id})
    return _public(doc) if doc else None


def create_item(project_id: str, table: str, fields: dict[str, Any]) -> dict:
    payload = {k: v for k, v in fields.items() if k != "id"}
    item_id = next_seq(f"dyn_{project_id}_{table}")
    doc = {
        "project_id": project_id,
        "table": table,
        "id": item_id,
        **payload,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    _rows().insert_one(doc)
    return _public(doc)


def update_item(project_id: str, table: str, item_id: int, fields: dict[str, Any]) -> Optional[dict]:
    payload = {k: v for k, v in fields.items() if k != "id"}
    if not payload:
        return get_item(project_id, table, item_id)
    payload["updated_at"] = utc_now()
    res = _rows().find_one_and_update(
        {"project_id": project_id, "table": table, "id": item_id},
        {"$set": payload},
        return_document=ReturnDocument.AFTER,
    )
    return _public(res) if res else None


def delete_item(project_id: str, table: str, item_id: int) -> bool:
    res = _rows().delete_one({"project_id": project_id, "table": table, "id": item_id})
    return res.deleted_count > 0
