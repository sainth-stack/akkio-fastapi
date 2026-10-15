"""Normalize compact LLM blocks to full AppDefinition block shape (items, series, columns)."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from app_builder.schemas.app_definition import AppDefinition

_KPI_COLORS = ("primary", "success", "warning", "info", "error")
_X_CANDIDATES = ("label", "name", "week", "shift", "hour", "date", "period")


def _humanize(key: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", key.replace("_", " "))
    return s.title() if s else key


def _is_placeholder_field(field: str) -> bool:
    f = (field or "").strip().lower()
    return bool(re.match(r"^(col_?\d+|field_?\d+|column_?\d+)$", f))


def _field_from_header(header: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (header or "").lower()).strip("_")
    return slug or f"field_{index}"


def repair_data_table_columns(block: Dict[str, Any]) -> Dict[str, Any]:
    b = dict(block)
    cols = b.get("columns")
    if not isinstance(cols, list):
        return b
    fixed: List[Dict[str, Any]] = []
    for i, col in enumerate(cols):
        if not isinstance(col, dict):
            continue
        field = str(col.get("field") or "")
        header = str(col.get("header") or _humanize(field))
        if _is_placeholder_field(field):
            field = _field_from_header(header, i)
        fixed.append({**col, "field": field, "header": header})
    b["columns"] = fixed
    return b


def repair_table_mock_keys(defn: AppDefinition) -> AppDefinition:
    from app_builder.services.generic_dataset_builder import _cell_value, _seed, _table_has_placeholders

    mock = dict(defn.mock or {})
    for page in defn.pages:
        new_blocks: List[Any] = []
        for block in page.blocks:
            if not isinstance(block, dict) or block.get("type") != "dataTable":
                new_blocks.append(block)
                continue
            block = repair_data_table_columns(block)
            new_blocks.append(block)
            dk = str(block.get("dataKey") or "")
            cols = block.get("columns") or []
            fields = [str(c.get("field")) for c in cols if isinstance(c, dict) and c.get("field")]
            headers = {str(c.get("field")): str(c.get("header") or "") for c in cols if isinstance(c, dict)}
            if not dk or not fields:
                continue
            rows = mock.get(dk)
            if not isinstance(rows, list):
                continue
            if rows and not _table_has_placeholders(rows):
                keys = list(rows[0].keys()) if isinstance(rows[0], dict) else []
                if keys == fields:
                    continue
            rng = _seed(dk)
            rebuilt: List[Dict[str, Any]] = []
            for i, row in enumerate(rows if isinstance(rows, list) else []):
                old = row if isinstance(row, dict) else {}
                old_keys = list(old.keys())
                new_row: Dict[str, Any] = {}
                for j, f in enumerate(fields):
                    if f in old and not _is_placeholder_field(str(f)):
                        new_row[f] = old[f]
                    elif j < len(old_keys) and not _is_placeholder_field(str(old_keys[j])):
                        new_row[f] = old[old_keys[j]]
                    else:
                        new_row[f] = _cell_value(f, i, rng, headers.get(f, ""))
                rebuilt.append(new_row)
            if len(rebuilt) < 12:
                start = len(rebuilt)
                for i in range(start, 28):
                    rebuilt.append({f: _cell_value(f, i, rng, headers.get(f, "")) for f in fields})
            mock[dk] = rebuilt
        page.blocks = new_blocks
    defn.mock = mock
    return defn


def _mock_bag(defn: AppDefinition) -> Dict[str, Any]:
    return dict(defn.mock or {})


def normalize_block(block: Dict[str, Any], mock: Dict[str, Any], page_id: str) -> Dict[str, Any]:
    b = dict(block)
    btype = b.get("type")

    if btype == "kpiRow":
        items = b.get("items")
        if isinstance(items, list) and items:
            return b
        data_key = b.get("dataKey") or f"kpis_{page_id}"
        slice_ = mock.get(data_key)
        if not isinstance(slice_, dict) or isinstance(slice_, list):
            return b
        b["items"] = [
            {
                "label": _humanize(str(k)),
                "valueKey": str(k),
                "color": _KPI_COLORS[i % len(_KPI_COLORS)],
            }
            for i, k in enumerate(slice_.keys())
        ]
        return b

    if btype in ("lineChart", "barChart"):
        series = b.get("series")
        data_key = b.get("dataKey")
        rows = mock.get(data_key) if data_key else None
        if isinstance(series, list) and series:
            return b
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            return b
        sample = rows[0]
        x_key = b.get("xAxisKey") or "label"
        if x_key not in sample:
            for cand in _X_CANDIDATES:
                if cand in sample:
                    x_key = cand
                    break
        b["xAxisKey"] = x_key
        b["series"] = [
            {"dataKey": k, "name": _humanize(k)}
            for k in sample.keys()
            if k != x_key
        ]
        return b

    if btype == "dataTable":
        b = repair_data_table_columns(b)
        columns = b.get("columns")
        if isinstance(columns, list) and columns:
            return b
        data_key = b.get("dataKey")
        rows = mock.get(data_key) if data_key else None
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            return b
        fields = list(rows[0].keys())
        b["columns"] = [
            {
                "field": f,
                "header": _humanize(f),
                "align": "right" if isinstance(rows[0].get(f), (int, float)) else None,
            }
            for f in fields
        ]
        b.setdefault("searchable", True)
        return b

    return b


def normalize_app_definition_blocks(defn: AppDefinition) -> AppDefinition:
    mock = _mock_bag(defn)
    for page in defn.pages:
        page.blocks = [
            normalize_block(b, mock, page.id) if isinstance(b, dict) else b
            for b in page.blocks
        ]
    defn = repair_table_mock_keys(defn)
    return defn
