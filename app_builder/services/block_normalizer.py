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
    return defn
