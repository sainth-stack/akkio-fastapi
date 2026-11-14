"""Step 2 chat update — SAP RAG agent searches Community and refreshes data.json."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from app_builder.services.runtime_paths import resolve_project_root


def _looks_like_sap_definition(data: Dict[str, Any]) -> bool:
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caps = meta.get("capabilities") or []
    if str(meta.get("domain") or "").lower() == "sap":
        return True
    if isinstance(caps, list) and "sap_similar_tickets" in caps:
        return True
    name = str(meta.get("appName") or "").lower()
    if "sap" in name and "similar" in name:
        return True
    bag = data.get("data") if isinstance(data.get("data"), dict) else {}
    if isinstance(bag.get("source_sap_tickets"), list):
        return True
    for page in data.get("pages") or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks") or []:
            if isinstance(block, dict) and block.get("type") == "sapCommunityFeed":
                return True
    return False


def is_sap_json_project(project_root: str) -> bool:
    for rel in (
        "frontend/src/mock/data.json",
        "frontend/src/data/store-context.json",
    ):
        path = os.path.join(project_root, rel)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if _looks_like_sap_definition(data):
                return True
        except (OSError, json.JSONDecodeError):
            continue
    name = os.path.basename(project_root.rstrip(os.sep)).lower()
    if "sap" in name and "similar" in name:
        return True
    return False


def wants_sap_rag_followup(text: str) -> bool:
    low = (text or "").lower()
    if re.search(r"\b(step\s*2|prompt\s*2|second prompt)\b", low):
        return True
    if re.search(r"\bstep\s*2\b.*\b(load|rag|context|community)\b", low):
        return True
    if "load sap ticket" in low or "sap data & rag" in low:
        return True
    if re.search(r"\b(rag|community|per-ticket|ticket context|sap context|search sap)\b", low):
        return True
    if re.search(r"\b(enrich|refresh)\b.*\b(ticket|rag)\b", low):
        return True
    if re.search(r"\b(seed|create|generate)\s+\d+\s+.*ticket", low):
        return True
    return False


async def run_sap_rag_update(
    *,
    project_name: str,
    project_root: str,
    user_request: str,
    websocket: Any = None,
) -> Optional[Dict[str, Any]]:
    if not is_sap_json_project(project_root) or not wants_sap_rag_followup(user_request):
        return None

    from app_builder.services.agent_pipeline_events import (
        AGENT_SAP_DATA_DEFINITION,
        AGENT_SAP_INTEGRATION,
        AGENT_SAP_RAG,
    )

    async def _emit(payload: Dict[str, Any]) -> None:
        if websocket:
            try:
                await websocket.send_text(json.dumps(payload))
            except Exception:
                pass

    await _emit({
        "event": "agent_start",
        "agent": AGENT_SAP_RAG,
        "message": "Step 2 — searching SAP Community for each ticket…",
    })
    await _emit({
        "event": "agent_progress",
        "agent": AGENT_SAP_RAG,
        "message": "Building per-ticket search query from title + component",
    })

    try:
        from app_builder.services.sap_ticket_rag_service import refresh_project_ticket_rag

        result = refresh_project_ticket_rag(
            project_name,
            force=True,
            requirement=user_request,
        )
    except Exception as exc:
        await _emit({
            "event": "agent_error",
            "agent": AGENT_SAP_RAG,
            "message": str(exc),
        })
        return {"status": "error", "message": str(exc)}

    n = int(result.get("ticket_count") or 0)
    rag_n = int(result.get("rag_count") or 0)
    await _emit({
        "event": "agent_progress",
        "agent": AGENT_SAP_RAG,
        "message": f"RAG ready for {rag_n}/{n} tickets (Community search + excerpts)",
    })
    await _emit({
        "event": "agent_complete",
        "agent": AGENT_SAP_RAG,
        "message": f"Per-ticket SAP context stored in data.json ({rag_n} records).",
    })

    await _emit({
        "event": "agent_start",
        "agent": AGENT_SAP_INTEGRATION,
        "message": "Refreshing app data bundle…",
    })
    from app_builder.services.app_data_rag import load_project_app_definition
    from app_builder.services.sap_similar_tickets import ensure_sap_similar_ticket_features_dict

    updated: Dict[str, str] = {}
    data_rel = "frontend/src/mock/data.json"
    full = load_project_app_definition(project_name)
    if full:
        full = ensure_sap_similar_ticket_features_dict(full, requirement=user_request)
        full_meta = full.get("meta") if isinstance(full.get("meta"), dict) else {}
        full_meta["sapRagPhase"] = "rag_complete"
        full["meta"] = full_meta
        updated[data_rel] = json.dumps(full, indent=2, ensure_ascii=False)
        try:
            from app_builder.schemas.files import GeneratedFiles
            from app_builder.services.file_writer import file_writer

            file_writer(project_name, GeneratedFiles(files=updated))
        except Exception as write_exc:
            print(f"[sap_rag] file write warning: {write_exc}")

    await _emit({
        "event": "agent_complete",
        "agent": AGENT_SAP_INTEGRATION,
        "message": "Dashboard and chat now use refreshed ticket RAG.",
    })
    await _emit({
        "event": "agent_complete",
        "agent": AGENT_SAP_DATA_DEFINITION,
        "message": "Ticket definitions aligned with step 2 prompt.",
    })

    return {
        "status": "success",
        "analysis": f"SAP RAG agent enriched {rag_n} tickets from Community search.",
        "updated_files": list(updated.keys()) or [data_rel],
        "updated_code_dict": updated,
        "update_type": "sap_rag",
    }