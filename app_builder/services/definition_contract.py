import json
import math
import re
from typing import Any, Dict, List, Optional


BLOCK_TYPES = frozenset({"filterBar", "actionBar", "kpiRow", "lineChart", "barChart", "pieChart", "dataTable", "text"})


def validate_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for block in blocks:
        kind = block.get("type")
        if kind not in BLOCK_TYPES:
            raise ValueError(f"Unsupported block type: {kind}")
        if kind in {"kpiRow", "lineChart", "barChart", "pieChart", "dataTable"} and not block.get("dataKey"):
            raise ValueError(f"{kind} requires dataKey")
        members = {"kpiRow": ("items", "valueKey"), "dataTable": ("columns", "field"), "lineChart": ("series", "dataKey"), "barChart": ("series", "dataKey")}
        if kind in members:
            key, field = members[kind]
            values = block.get(key)
            if not isinstance(values, list) or not values or any(not isinstance(v, dict) or not v.get(field) for v in values):
                raise ValueError(f"{kind} requires nonempty {key} with {field}")
            names = [v[field] for v in values]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate {field} in {kind}")
        if kind == "dataTable" and any(not c.get("header") for c in block["columns"]):
            raise ValueError("Table columns require header")
        if kind == "dataTable":
            import re

            if any(re.match(r"^(col_?\d+|field_?\d+|column_?\d+)$", c["field"], re.I) for c in block["columns"]):
                raise ValueError("Placeholder table columns are not allowed")
        if kind in {"lineChart", "barChart"} and not block.get("xAxisKey"):
            raise ValueError("Charts require xAxisKey")
        if kind == "filterBar":
            for f in block.get("filters", []):
                if f.get("filterType") not in {"search", "select", "dateRange"}:
                    raise ValueError("Unsupported filter type")
                if f.get("filterType") != "search" and not f.get("field"):
                    raise ValueError("Select and date filters require field")
        if kind == "actionBar":
            for action in block.get("actions", []):
                if action.get("actionType") not in {"create", "exportCsv", "exportPdf"}:
                    raise ValueError("Actions must be create, exportCsv, or exportPdf")
                if action.get("actionType") == "create" and not action.get("dataKey"):
                    raise ValueError("Create action requires a table dataKey")
    return blocks


_CONTAINER_KEYS = ("data", "mock", "datasets", "pageData", "datasetsByKey")
_NEST_KEYS = {"datasets", "mock", "data", "pageData", "datasetsByKey", "meta", "payload"}
_SKIP_KEYS = {"description", "notes", "commentary", "page", "pageId", "page_id", "title"}


def _parse_json_object(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _datasets_from_list(items: List[Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = item.get("dataKey") or item.get("key") or item.get("id")
        if not key:
            continue
        if "values" in item:
            payload = item["values"]
        elif "rows" in item:
            payload = item["rows"]
        elif "data" in item:
            payload = item["data"]
        else:
            payload = {k: v for k, v in item.items() if k not in {"dataKey", "key", "id", "type", "title"}}
        if payload is not None:
            out[str(key)] = payload
    return out


def _unwrap_nested(payload: Dict[str, Any]) -> Dict[str, Any]:
    current = dict(payload)
    for _ in range(3):
        nested = [key for key, value in current.items() if key in _NEST_KEYS and isinstance(value, dict)]
        direct = [key for key, value in current.items() if key not in _NEST_KEYS and isinstance(value, (dict, list))]
        if direct or len(nested) != 1:
            return current
        inner = current[nested[0]]
        if not any(isinstance(value, (dict, list)) for value in inner.values()):
            return current
        current = dict(inner)
    return current


def coerce_page_data(value: Any) -> Dict[str, Any]:
    """Collect datasets whether the model nested them under data/mock or emitted them as top-level keys."""
    value = _parse_json_object(value)
    if not isinstance(value, dict):
        return {}
    merged: Dict[str, Any] = {}
    for name in _CONTAINER_KEYS:
        raw = _parse_json_object(value.get(name))
        if isinstance(raw, dict):
            for key, item in _unwrap_nested(raw).items():
                merged.setdefault(key, item)
        elif isinstance(raw, list):
            for key, item in _datasets_from_list(raw).items():
                merged.setdefault(key, item)
    for key, item in value.items():
        if key in _CONTAINER_KEYS or key in _SKIP_KEYS:
            continue
        if isinstance(item, (dict, list)) and key not in merged:
            merged[key] = item
    return _unwrap_nested(merged)


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _key_tokens(key: str) -> frozenset:
    return frozenset(re.findall(r"[a-z0-9]+", key.lower()))


def _binding_kind(block: Dict[str, Any]) -> str:
    return str(block.get("type") or "")


def _shape_ok(kind: str, value: Any) -> bool:
    if kind == "kpiRow":
        return isinstance(value, dict)
    return isinstance(value, list)


def required_bindings(blocks: List[Dict[str, Any]]) -> List[tuple]:
    bindings = []
    for block in blocks:
        kind = _binding_kind(block)
        key = block.get("dataKey")
        if key and kind in {"kpiRow", "lineChart", "barChart", "pieChart", "dataTable"}:
            bindings.append((str(key), kind))
        if kind == "filterBar":
            for flt in block.get("filters") or []:
                if isinstance(flt, dict) and flt.get("optionsKey"):
                    bindings.append((str(flt["optionsKey"]), "options"))
    return bindings


def _best_alias(key: str, kind: str, data: Dict[str, Any], required: set) -> Optional[str]:
    wanted = _norm_key(key)
    wanted_tokens = _key_tokens(key)
    ranked = []
    for name, value in data.items():
        if name in required or str(name).endswith("_meta") or not _shape_ok(kind, value):
            continue
        tokens = _key_tokens(str(name))
        if _norm_key(str(name)) == wanted or tokens == wanted_tokens:
            ranked.append((0, str(name)))
        elif kind == "kpiRow" and "kpi" in tokens and tokens <= wanted_tokens:
            ranked.append((1, str(name)))
    if ranked:
        ranked.sort()
        top = [name for rank, name in ranked if rank == ranked[0][0]]
        return top[0] if len(top) == 1 else None
    loose = [
        str(name)
        for name, value in data.items()
        if name not in required and not str(name).endswith("_meta") and _shape_ok(kind, value) and (_key_tokens(str(name)) & wanted_tokens)
    ]
    return loose[0] if len(loose) == 1 else None


def align_page_datasets(blocks: List[Dict[str, Any]], data: Dict[str, Any]) -> Dict[str, Any]:
    """Bind misnamed LLM datasets onto the dataKeys declared by the page blocks."""
    out = coerce_page_data(data if isinstance(data, dict) else {})
    bindings = required_bindings(blocks)
    required = {key for key, _kind in bindings}
    for key, kind in bindings:
        if key in out and _shape_ok(kind, out[key]):
            continue
        match = _best_alias(key, kind, out, required)
        if match:
            out[key] = out[match]
    return out


def dataset_shape_hint(blocks: List[Dict[str, Any]]) -> str:
    shape: Dict[str, Any] = {}
    for block in blocks:
        kind = _binding_kind(block)
        key = block.get("dataKey")
        if key and kind == "kpiRow":
            shape[str(key)] = {
                str(item.get("valueKey")): "<string or number>"
                for item in block.get("items") or []
                if isinstance(item, dict) and item.get("valueKey")
            }
        elif key and kind in {"lineChart", "barChart"}:
            row: Dict[str, Any] = {str(block.get("xAxisKey") or "period"): "<label>"}
            for series in block.get("series") or []:
                if isinstance(series, dict) and series.get("dataKey"):
                    row[str(series["dataKey"])] = 0
            shape[str(key)] = [row]
        elif key and kind == "pieChart":
            shape[str(key)] = [{"name": "<segment>", "value": 0}]
        elif key and kind == "dataTable":
            shape[str(key)] = [
                {
                    str(col.get("field")): "<value>"
                    for col in block.get("columns") or []
                    if isinstance(col, dict) and col.get("field")
                }
            ]
        if kind == "filterBar":
            for flt in block.get("filters") or []:
                options_key = flt.get("optionsKey") if isinstance(flt, dict) else None
                if options_key:
                    shape[str(options_key)] = [{"value": "<id>", "label": "<label>"}]
    return json.dumps({"data": shape}, indent=2)[:8000]


def validate_page_data(blocks: List[Dict[str, Any]], data: Dict[str, Any]) -> None:
    for block in blocks:
        kind = block["type"]
        key = block.get("dataKey")
        if key:
            if key not in data:
                raise ValueError(f"Missing dataset: {key}")
            rows = data[key]
            if kind == "kpiRow":
                if not isinstance(rows, dict) or any(item["valueKey"] not in rows for item in block["items"]):
                    raise ValueError(f"Incomplete KPI bundle: {key}")
            elif kind in {"dataTable", "lineChart", "barChart", "pieChart"}:
                if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                    raise ValueError(f"{key} must contain object rows")
                fields = [c["field"] for c in block["columns"]] if kind == "dataTable" else ([block["xAxisKey"]] + [s["dataKey"] for s in block["series"]] if kind in {"lineChart", "barChart"} else ["name", "value"])
                if any(any(f not in row for f in fields) for row in rows):
                    raise ValueError(f"Dataset {key} must include fields {fields}")
                numeric = [s["dataKey"] for s in block["series"]] if kind in {"lineChart", "barChart"} else (["value"] if kind == "pieChart" else [])
                if any(any(row[f] is not None and (not isinstance(row[f], (int, float)) or isinstance(row[f], bool) or not math.isfinite(row[f])) for f in numeric) for row in rows):
                    raise ValueError(f"Dataset {key} requires finite numeric series")
        for f in block.get("filters", []):
            if f.get("optionsKey"):
                options = data.get(f["optionsKey"])
                if not isinstance(options, list) or any(not isinstance(o, dict) or "value" not in o or "label" not in o for o in options):
                    raise ValueError(f"Invalid options dataset: {f['optionsKey']}")


def validate_definition(definition: Any) -> None:
    ids = [p.id for p in definition.pages]
    paths = [n.path for n in definition.navigation]
    if not ids or len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise ValueError("Pages and navigation paths must be nonempty and unique")
    if set(ids) != {n.pageId for n in definition.navigation} or not paths or paths[0] != "/":
        raise ValueError("Navigation must reference every page and begin at /")
    datasets = set()
    tables = set()
    for page in definition.pages:
        if not page.blocks:
            raise ValueError(f"Page {page.id} has no blocks")
        validate_blocks(page.blocks)
        validate_page_data(page.blocks, definition.mock)
        for block in page.blocks:
            key = block.get("dataKey")
            if key and key in datasets:
                raise ValueError(f"Duplicate dataset binding: {key}")
            if key:
                datasets.add(key)
            if block["type"] == "dataTable":
                tables.add(key)
    for page in definition.pages:
        for block in page.blocks:
            for action in block.get("actions", []):
                if action.get("actionType") == "create" and action.get("dataKey") not in tables:
                    raise ValueError("Create action must reference an existing table")
    from app_builder.services.definition_metrics import compute_definition

    compute_definition(definition.model_dump(mode="json"))