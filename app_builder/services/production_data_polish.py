"""Shape AppDefinition datasets like production API payloads (not demo placeholders)."""
from __future__ import annotations

from typing import Any, Dict, Set

from app_builder.schemas.app_definition import AppDefinition

_PLACEHOLDER_SCALARS = frozenset({"—", "1,240", "Sample", "N/A", ""})


def _kpi_value_keys(defn: AppDefinition) -> Set[str]:
    keys: Set[str] = set()
    for page in defn.pages:
        for block in page.blocks:
            if block.get("type") != "kpiRow":
                continue
            for item in block.get("items") or []:
                if isinstance(item, dict) and item.get("valueKey"):
                    keys.add(str(item["valueKey"]))
    return keys


def polish_production_data_bag(defn: AppDefinition) -> Dict[str, Any]:
    """Return a single data bag: page KPI bundles, series, tables — no stray placeholder scalars."""
    bag: Dict[str, Any] = dict(defn.mock or {})
    value_keys = _kpi_value_keys(defn)

    bundled: Set[str] = set()
    for page in defn.pages:
        bundle_key = f"kpis_{page.id}"
        bundle = bag.get(bundle_key)
        if isinstance(bundle, dict):
            bundled.update(str(k) for k in bundle.keys())

    for key in list(bag.keys()):
        if key.startswith("kpis_"):
            continue
        if key in value_keys and key in bundled:
            val = bag[key]
            if val in _PLACEHOLDER_SCALARS or (isinstance(val, str) and val.strip() in _PLACEHOLDER_SCALARS):
                del bag[key]
            elif key in bundled:
                del bag[key]

    return bag


def app_definition_to_client_json(defn: AppDefinition) -> Dict[str, Any]:
    """Serialize for frontend/src/mock/data.json — production `data` only (no duplicate mock bag)."""
    raw = defn.model_dump(mode="json")
    datasets = polish_production_data_bag(defn)
    raw.pop("mock", None)
    raw["data"] = datasets
    meta = dict(raw.get("meta") or {})
    if meta.get("description", "").lower().find("mock") >= 0:
        meta["description"] = (
            meta.get("description", "")
            .replace("mock/data.json", "data.json")
            .replace("from mock/", "from ")
        )
    if not meta.get("description"):
        meta["description"] = "Operational datasets loaded from data.json"
    meta.setdefault("dataSource", "static")
    raw["meta"] = meta
    return raw
