"""Default `frontend/src/mock/data.json` for base-frontend-vite-mui (standalone template build)."""
from __future__ import annotations

from typing import Any, Dict


def app_data_dict_from_definition(definition: Any) -> Dict[str, Any]:
    """Serialize AppDefinition to mock/data.json shape (includes `data` + legacy `mock`)."""
    raw = definition.to_json_dict() if hasattr(definition, "to_json_dict") else dict(definition)
    datasets = raw.get("data") or raw.get("mock") or {}
    raw["data"] = datasets
    raw["mock"] = datasets
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
        "description": "Default template — all UI is driven from mock/data.json",
    }
    return raw
