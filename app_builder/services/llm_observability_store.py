"""Persist per-app LLM call records for the Observability screen."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app_builder.services.runtime_paths import resolve_project_root

logger = logging.getLogger("app_builder")

_FILENAME = "llm_observability.json"
_MAX_EVENTS = 500


def _store_path(project_id: str) -> Optional[str]:
    root = resolve_project_root(project_id)
    if not root or not os.path.isdir(root):
        return None
    return os.path.join(root, _FILENAME)


def _read_doc(project_id: str) -> Dict[str, Any]:
    path = _store_path(project_id)
    if not path or not os.path.isfile(path):
        return {"version": 1, "events": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("events"), list):
            return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[llm_observability] read failed %s: %s", project_id, exc)
    return {"version": 1, "events": []}


def _write_doc(project_id: str, doc: Dict[str, Any]) -> bool:
    path = _store_path(project_id)
    if not path:
        return False
    tmp = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path) or path, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        os.replace(tmp, path)
        return True
    except OSError as exc:
        logger.warning("[llm_observability] write failed %s: %s", project_id, exc)
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


def build_observability_event(
    *,
    source: str,
    model: str,
    prompt: str,
    output: str,
    llm_response_obj: Any,
) -> Dict[str, Any]:
    from app_builder.services.llm_model_pricing import (
        estimate_cost_usd,
        extract_token_breakdown,
        normalize_model_id,
    )

    tokens = extract_token_breakdown(llm_response_obj)
    model_id = normalize_model_id(model)
    cost = estimate_cost_usd(model_id, tokens["input_tokens"], tokens["output_tokens"])
    return {
        "id": uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": source,
        "model": model_id or model,
        "prompt": (prompt or "")[:4000],
        "output": (output or "")[:8000],
        "input_tokens": tokens["input_tokens"],
        "output_tokens": tokens["output_tokens"],
        "total_tokens": tokens["total_tokens"],
        "cost_usd": cost,
    }


def append_observability_event(project_id: str, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not project_id or not isinstance(event, dict):
        return None
    doc = _read_doc(project_id)
    events: List[Dict[str, Any]] = list(doc.get("events") or [])
    events.append(event)
    if len(events) > _MAX_EVENTS:
        events = events[-_MAX_EVENTS:]
    doc["events"] = events
    doc["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _write_doc(project_id, doc)
    return event


def list_observability_events(project_id: str, *, limit: int = 100) -> List[Dict[str, Any]]:
    doc = _read_doc(project_id)
    events = list(doc.get("events") or [])
    if limit > 0:
        events = events[-limit:]
    return list(reversed(events))


def summarize_observability(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_tokens = sum(int(e.get("total_tokens") or 0) for e in events)
    total_cost = sum(float(e.get("cost_usd") or 0) for e in events)
    by_model: Dict[str, Dict[str, Any]] = {}
    for e in events:
        model = str(e.get("model") or "unknown")
        bucket = by_model.setdefault(
            model,
            {"model": model, "calls": 0, "total_tokens": 0, "cost_usd": 0.0},
        )
        bucket["calls"] += 1
        bucket["total_tokens"] += int(e.get("total_tokens") or 0)
        bucket["cost_usd"] += float(e.get("cost_usd") or 0)
    for bucket in by_model.values():
        bucket["cost_usd"] = round(bucket["cost_usd"], 6)
    return {
        "call_count": len(events),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "by_model": sorted(by_model.values(), key=lambda x: -x["total_tokens"]),
    }
