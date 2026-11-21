import math
from typing import Any, Dict, List


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