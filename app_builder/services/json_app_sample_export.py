"""Build Excel export from json_app mock/data.json (frontend-only projects)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from app_builder.services.runtime_paths import resolve_project_root

MIN_EXPORT_ROWS = 12


def _load_app_definition_mock(project_name: str) -> Dict[str, Any] | None:
    root = resolve_project_root(project_name)
    for rel in (
        "frontend/src/mock/data.json",
        "frontend/src/config/app-definition.json",
    ):
        path = os.path.join(root, rel)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            bag = data.get("data") or data.get("mock") or {}
            if isinstance(bag, dict) and bag:
                return {
                    "meta": data.get("meta") or {},
                    "mock": bag,
                    "pages": data.get("pages") or [],
                }
    return None


def _flatten_kpis(mock: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, val in mock.items():
        if key.startswith("kpis_") and isinstance(val, dict):
            for k, v in val.items():
                rows.append({"Metric": f"{key}.{k}", "Value": v})
        elif key.startswith("kpis_") and not isinstance(val, (list, dict)):
            rows.append({"Metric": key, "Value": val})
    if not rows:
        for k, v in mock.items():
            if isinstance(v, (str, int, float)) and not k.startswith("series_") and not k.startswith("table_"):
                if "session" not in k and "analytics" not in k and "report" not in k:
                    rows.append({"Metric": k.replace("_", " ").title(), "Value": v})
    return rows


def _table_candidates(mock: Dict[str, Any]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    out: List[Tuple[str, List[Dict[str, Any]]]] = []
    for key, val in mock.items():
        if isinstance(val, list) and val and isinstance(val[0], dict):
            out.append((key, val))
    out.sort(key=lambda x: len(x[1]), reverse=True)
    return out


def _series_as_rows(mock: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key, val in mock.items():
        if key.startswith("series_") and isinstance(val, list) and val:
            return val
        if key.endswith("_trends") and isinstance(val, list) and val:
            return val
    return []


def build_sheets_from_json_app(project_name: str) -> Dict[str, List[Dict[str, Any]]] | None:
    loaded = _load_app_definition_mock(project_name)
    if not loaded:
        return None

    mock = loaded["mock"]
    meta = loaded.get("meta") or {}
    app_name = meta.get("appName") or project_name

    kpis = _flatten_kpis(mock)
    if not kpis and isinstance(mock.get("kpis_dashboard"), dict):
        for k, v in mock["kpis_dashboard"].items():
            kpis.append({"Metric": k.replace("_", " ").title(), "Value": v})

    tables = _table_candidates(mock)
    detail: List[Dict[str, Any]] = tables[0][1] if tables else []
    breakdown: List[Dict[str, Any]] = tables[1][1] if len(tables) > 1 else _series_as_rows(mock)

    if not breakdown and detail:
        breakdown = [{"Note": f"Exported from {app_name}", "Rows": len(detail)}]

    if not kpis:
        kpis = [{"Metric": "Application", "Value": app_name}]

    if not detail:
        detail = [{"id": "—", "message": "No tabular mock keys in data.json"}]

    return {
        "KPIs": [{"Label": r.get("Metric", r.get("label")), "Value": r.get("Value", r.get("value"))} for r in kpis],
        "Detail": detail[:500],
        "Breakdown": breakdown[:200] if breakdown else detail[:50],
    }
