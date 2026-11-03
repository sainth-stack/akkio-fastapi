"""Per-app predict form: numeric source columns + user-chosen target field."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app_builder.services.source_dataset_planner import get_source_datasets


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def _avg_from_rows(rows: List[Dict[str, Any]], field: str) -> float:
    from app_builder.services.operational_metrics_compute import _num

    nums = [_num(r.get(field)) for r in rows]
    vals = [n for n in nums if n is not None]
    return sum(vals) / len(vals) if vals else 0.0


def build_predict_config(defn: Dict[str, Any]) -> Dict[str, Any]:
    """Fields derived from sourceDatasets + dashboard KPI labels (prompt-specific)."""
    rows: List[Dict[str, Any]] = []
    bag = defn.get("data") or defn.get("mock") or {}
    numeric_fields: List[Dict[str, Any]] = []
    seen: set = set()

    for ds in get_source_datasets(defn):
        dk = str(ds.get("dataKey") or "")
        src_rows = bag.get(dk) if isinstance(bag.get(dk), list) else []
        if src_rows:
            rows = src_rows
        for col in ds.get("columns") or []:
            if not isinstance(col, dict):
                continue
            field = str(col.get("field") or "")
            if not field or field in seen:
                continue
            ctype = str(col.get("type") or "")
            if ctype != "number" and ctype != "datetime":
                if not re.search(
                    r"hours|pct|rate|qty|count|units|total|price|output|defect|yield|downtime|mtbf|mttr|oee",
                    field,
                    re.I,
                ):
                    continue
            seen.add(field)
            numeric_fields.append(
                {
                    "key": field,
                    "label": str(col.get("header") or field.replace("_", " ").title()),
                    "unit": _unit_for_field(field),
                }
            )

    # Prefer KPI-linked targets from dashboard labels
    kpi_targets: List[str] = []
    for page in defn.get("pages") or []:
        for block in page.get("blocks") or []:
            if block.get("type") != "kpiRow":
                continue
            for item in block.get("items") or []:
                lab = str(item.get("label") or "").lower()
                vk = str(item.get("valueKey") or "")
                for f in numeric_fields:
                    if _slug(f["key"]) in _slug(vk) or _slug(f["key"]) in _slug(lab):
                        kpi_targets.append(f["key"])
                if "yield" in lab or "fpy" in lab:
                    for f in numeric_fields:
                        if "yield" in f["key"]:
                            kpi_targets.append(f["key"])
                if "defect" in lab:
                    for f in numeric_fields:
                        if "defect" in f["key"]:
                            kpi_targets.append(f["key"])

    default_target = kpi_targets[0] if kpi_targets else (numeric_fields[0]["key"] if numeric_fields else "value")
    baseline = {f["key"]: round(_avg_from_rows(rows, f["key"]), 4) for f in numeric_fields if rows}

    return {
        "version": 1,
        "targetField": default_target,
        "fields": numeric_fields,
        "baseline": baseline,
        "description": "Choose the metric to forecast; fill the other inputs from your operational data averages.",
    }


def attach_predict_config(defn: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defn)
    meta = dict(out.get("meta") or {})
    if not meta.get("predictConfig"):
        meta["predictConfig"] = build_predict_config(out)
    out["meta"] = meta
    return out


def _unit_for_field(field: str) -> str:
    f = field.lower()
    if "pct" in f or "rate" in f or "yield" in f or "oee" in f:
        return "%"
    if "hours" in f or "mtbf" in f or "mttr" in f or "downtime" in f:
        return "h"
    if "price" in f or "total" in f or "revenue" in f:
        return "$"
    return ""
