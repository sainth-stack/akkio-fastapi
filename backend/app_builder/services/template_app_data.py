"""Default `frontend/src/mock/data.json` for base-frontend-vite-mui (standalone template build)."""
from __future__ import annotations

from typing import Any, Dict


def app_data_dict_from_definition(definition: Any) -> Dict[str, Any]:
    """Serialize AppDefinition to data.json shape (`data` bag only — production-style payloads)."""
    from app_builder.services.production_data_polish import app_definition_to_client_json

    if hasattr(definition, "model_dump"):
        return app_definition_to_client_json(definition)
    raw = dict(definition)
    datasets = raw.get("data") or raw.get("mock") or {}
    raw.pop("mock", None)
    raw["data"] = datasets
    meta = dict(raw.get("meta") or {})
    meta.setdefault("dataSource", "static")
    raw["meta"] = meta
    return raw


def default_template_app_data() -> Dict[str, Any]:
    """Demo app from PRD-style requirement (sync path — no LLM required for template checkout)."""
    from app_builder.services.app_definition_generator import resolve_app_definition_sync

    definition = resolve_app_definition_sync(
        requirement="PulsePlant — manufacturing monitoring with dashboard, analytics, and reports",
        plan_json={"blueprint": {"selected_domain": "manufacturing", "domain": "manufacturing"}},
        app_name="PulsePlant",
    )
    raw = app_data_dict_from_definition(definition)
    raw["meta"] = {
        **raw.get("meta", {}),
        "description": "PulsePlant — live-style operational data from data.json",
    }
    return raw
