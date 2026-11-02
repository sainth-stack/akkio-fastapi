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


def _load_full_data_json(project_name: str) -> Dict[str, Any] | None:
    root = resolve_project_root(project_name)
    path = os.path.join(root, "frontend", "src", "mock", "data.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else None


def _sheet_title(data_key: str, used: set) -> str:
    base = (data_key or "data")[:31]
    title = base
    n = 2
    while title in used:
        suffix = f"_{n}"
        title = f"{base[: 31 - len(suffix)]}{suffix}"
        n += 1
    used.add(title)
    return title


def build_import_template_sheets(project_name: str) -> Dict[str, List[Dict[str, Any]]] | None:
    """
    Excel template aligned with upload import: one sheet per block dataKey.
    KPI sheets use columns label,value; tables use column field names; charts use x + series keys.
    """
    defn = _load_full_data_json(project_name)
    if not defn or not defn.get("pages"):
        return None

    bag = defn.get("data") or defn.get("mock") or {}
    if not isinstance(bag, dict):
        bag = {}

    sheets: Dict[str, List[Dict[str, Any]]] = {}
    used_titles: set = set()
    seen_keys: set = set()

    for page in defn.get("pages") or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            data_key = str(block.get("dataKey") or "").strip()
            if not data_key or data_key in seen_keys:
                continue
            seen_keys.add(data_key)
            sheet = _sheet_title(data_key, used_titles)
            btype = str(block.get("type") or "")

            if btype == "kpiRow":
                items = block.get("items") or []
                bundle = bag.get(data_key) if isinstance(bag.get(data_key), dict) else {}
                rows: List[Dict[str, Any]] = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("label") or "")
                    vk = str(item.get("valueKey") or "")
                    val = bundle.get(vk, "") if isinstance(bundle, dict) and vk else ""
                    rows.append({"label": label, "value": val})
                if not rows:
                    rows = [{"label": "Example metric", "value": "100"}]
                sheets[sheet] = rows
                continue

            if btype == "dataTable":
                cols = block.get("columns") or []
                fields = [str(c.get("field")) for c in cols if isinstance(c, dict) and c.get("field")]
                existing = bag.get(data_key)
                if isinstance(existing, list) and existing:
                    sheets[sheet] = existing[:500]
                elif fields:
                    example = {f: f"sample_{f}" for f in fields}
                    sheets[sheet] = [example]
                continue

            if btype in ("lineChart", "barChart"):
                x_key = str(block.get("xAxisKey") or "period")
                series_keys = [
                    str(s.get("dataKey"))
                    for s in (block.get("series") or [])
                    if isinstance(s, dict) and s.get("dataKey")
                ]
                existing = bag.get(data_key)
                if isinstance(existing, list) and existing:
                    sheets[sheet] = existing[:200]
                else:
                    row: Dict[str, Any] = {x_key: "Period 1"}
                    for sk in series_keys:
                        row[sk] = 0
                    sheets[sheet] = [row]
                continue

            raw = bag.get(data_key)
            if isinstance(raw, list) and raw and isinstance(raw[0], dict):
                sheets[sheet] = raw[:200]
            elif isinstance(raw, dict):
                sheets[sheet] = [{"label": k, "value": v} for k, v in raw.items()]

    return sheets if sheets else None
