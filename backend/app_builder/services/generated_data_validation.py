from __future__ import annotations

import math
import re
from typing import Any, Dict, List


class GeneratedDataValidationError(ValueError):
    pass


def validate_page_data(blocks: List[Dict[str, Any]], data: Dict[str, Any], page_id: str) -> None:
    errors: List[str] = []
    for block in blocks:
        kind = block.get("type")
        key = str(block.get("dataKey") or "")
        if kind == "filterBar":
            for item in block.get("filters") or []:
                options_key = item.get("optionsKey")
                if options_key and not isinstance(data.get(options_key), list):
                    errors.append(f"{options_key}: filter options must be an array")
            continue
        if kind not in ("kpiRow", "dataTable", "lineChart", "barChart", "pieChart"):
            continue
        if not key or key not in data:
            errors.append(f"{key or kind}: missing dataset")
            continue
        value = data[key]
        if kind == "kpiRow":
            items = block.get("items") or []
            if not items or not isinstance(value, dict):
                errors.append(f"{key}: KPI items and an object dataset are required")
                continue
            for item in items:
                field = item.get("valueKey")
                cell = value.get(field)
                if not field or not isinstance(cell, (str, int, float)) or isinstance(cell, bool) or cell == "":
                    errors.append(f"{key}.{field}: missing KPI value")
            continue
        if not isinstance(value, list) or not value:
            errors.append(f"{key}: a non-empty row array is required")
            continue
        numeric_fields: List[str] = []
        if kind == "dataTable":
            columns = block.get("columns") or []
            fields = [str(c.get("field") or "") for c in columns if isinstance(c, dict)]
            if not fields or len(fields) != len(columns) or len(fields) != len(set(fields)) or any(
                not f or re.fullmatch(r"(?:col_?|field_?|column_?)\d+", f, re.I) for f in fields
            ):
                errors.append(f"{key}: invalid or placeholder table columns")
            if any(not c.get("header") for c in columns if isinstance(c, dict)):
                errors.append(f"{key}: table column headers are required")
        elif kind == "pieChart":
            fields = ["name", "value"]
            numeric_fields = ["value"]
        else:
            series = block.get("series") or []
            numeric_fields = [str(s.get("dataKey") or "") for s in series if isinstance(s, dict)]
            x_key = str(block.get("xAxisKey") or "period")
            fields = [x_key, *numeric_fields]
            if not numeric_fields or len(numeric_fields) != len(series) or any(not f for f in numeric_fields):
                errors.append(f"{key}: chart series are required")
        for index, row in enumerate(value):
            if not isinstance(row, dict):
                errors.append(f"{key}[{index}]: row must be an object")
                continue
            for field in fields:
                if field not in row:
                    errors.append(f"{key}[{index}].{field}: missing field")
                elif isinstance(row[field], str) and re.match(r"^(Sample |Field \d+ \d+$|Column \d+)", row[field]):
                    errors.append(f"{key}[{index}].{field}: placeholder value")
            for field in numeric_fields:
                number = row.get(field)
                if not isinstance(number, (int, float)) or isinstance(number, bool) or not math.isfinite(number):
                    errors.append(f"{key}[{index}].{field}: finite chart number required")
    if errors:
        raise GeneratedDataValidationError(f"Generated data for {page_id} is incomplete: " + "; ".join(errors[:20]))


def validate_generated_definition(definition: Any) -> None:
    raw = definition.model_dump(mode="json") if hasattr(definition, "model_dump") else definition
    data = raw.get("data") if isinstance(raw.get("data"), dict) else raw.get("mock") or {}
    pages = raw.get("pages") or []
    ids = [p.get("id") for p in pages]
    if not ids or len(ids) != len(set(ids)):
        raise GeneratedDataValidationError("Generated pages must have unique IDs")
    paths = [n.get("path") for n in raw.get("navigation") or []]
    if not paths or len(paths) != len(set(paths)) or any(n.get("pageId") not in ids for n in raw.get("navigation") or []):
        raise GeneratedDataValidationError("Generated navigation must have unique routes targeting existing pages")
    keys: set[str] = set()
    for page in pages:
        blocks = page.get("blocks") or []
        if not blocks and page.get("id") not in ("predict", "observability", "create_ticket", "similar_tickets"):
            raise GeneratedDataValidationError(f"Generated page {page.get('id')} has no blocks")
        if any(not isinstance(b, dict) for b in blocks):
            raise GeneratedDataValidationError("Generated blocks must be objects")
        for block in blocks:
            key = block.get("dataKey")
            if key:
                if key in keys:
                    raise GeneratedDataValidationError(f"Generated dataset key {key} is reused by multiple blocks")
                keys.add(key)
        validate_page_data(blocks, data, str(page.get("id")))