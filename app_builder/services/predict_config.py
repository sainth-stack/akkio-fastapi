"""Per-app predict form: numeric source columns + user-chosen target field."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from app_builder.services.source_dataset_planner import get_source_datasets

_NON_PREDICT_FIELD = re.compile(
    r"^(status|shift|product|production_line|root_cause|severity|event_type|"
    r"customer|category|machine|equipment|inspection_id|work_order_id|order_id|"
    r"customer_id|product_name|.*_id)$",
    re.I,
)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def _avg_from_rows(rows: List[Dict[str, Any]], field: str) -> float:
    from app_builder.services.operational_metrics_compute import _num

    nums = [_num(r.get(field)) for r in rows]
    vals = [n for n in nums if n is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _source_rows(defn: Dict[str, Any]) -> List[Dict[str, Any]]:
    bag = defn.get("data") or defn.get("mock") or {}
    for ds in get_source_datasets(defn):
        dk = str(ds.get("dataKey") or "")
        src_rows = bag.get(dk) if isinstance(bag.get(dk), list) else []
        if src_rows:
            return src_rows
    for key, val in bag.items():
        if str(key).startswith("source_") and isinstance(val, list) and val:
            return val
    return []


def resolve_predict_baseline(defn: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, float]:
    """Use stored baseline when present; otherwise (or when zero) average from source rows."""
    rows = _source_rows(defn)
    keys = [str(f.get("key") or "") for f in cfg.get("fields") or [] if isinstance(f, dict) and f.get("key")]
    stored = cfg.get("baseline") if isinstance(cfg.get("baseline"), dict) else {}
    out: Dict[str, float] = {}
    for key in keys:
        raw = stored.get(key)
        val = float(raw) if raw not in (None, "") else 0.0
        if rows and (val == 0.0 or key not in stored):
            avg = round(_avg_from_rows(rows, key), 4)
            if avg:
                val = avg
        out[key] = val
    return out


def _field_is_predictable(field: str, header: str, ctype: str, rows: List[Dict[str, Any]]) -> bool:
    from app_builder.services.operational_metrics_compute import _num

    if not field or _NON_PREDICT_FIELD.match(field):
        return False
    if ctype == "string":
        return False
    numeric_hits = 0
    if rows:
        numeric_hits = sum(1 for r in rows[:12] if _num(r.get(field)) is not None)
    if ctype == "datetime":
        return numeric_hits >= max(1, len(rows[:12]) // 3)
    if ctype == "number":
        return True
    if not re.search(
        r"hours|pct|rate|qty|count|units|total|price|output|defect|yield|downtime|mtbf|mttr|oee|scrap|passed|inspected",
        f"{field} {header}",
        re.I,
    ):
        return False
    if rows:
        if numeric_hits < max(1, len(rows[:12]) // 3):
            return False
    return True


def build_predict_config(defn: Dict[str, Any]) -> Dict[str, Any]:
    """Fields derived from sourceDatasets + dashboard KPI labels (prompt-specific)."""
    rows: List[Dict[str, Any]] = _source_rows(defn)
    numeric_fields: List[Dict[str, Any]] = []
    seen: set = set()

    for ds in get_source_datasets(defn):
        for col in ds.get("columns") or []:
            if not isinstance(col, dict):
                continue
            field = str(col.get("field") or "")
            if not field or field in seen:
                continue
            header = str(col.get("header") or field.replace("_", " ").title())
            ctype = str(col.get("type") or "")
            if not _field_is_predictable(field, header, ctype, rows):
                continue
            seen.add(field)
            unit = _unit_for_field(field, header)
            numeric_fields.append({"key": field, "label": header, "unit": unit})

    kpi_targets: List[str] = []
    for page in defn.get("pages") or []:
        for block in page.get("blocks") or []:
            if block.get("type") != "kpiRow":
                continue
            for item in block.get("items") or []:
                lab = str(item.get("label") or "").lower()
                vk = str(item.get("valueKey") or "")
                for f in numeric_fields:
                    fk = f["key"]
                    if _slug(fk) in _slug(vk) or _slug(fk) in _slug(lab):
                        kpi_targets.append(fk)
                if "yield" in lab or "fpy" in lab:
                    for f in numeric_fields:
                        if "yield" in f["key"]:
                            kpi_targets.append(f["key"])
                if "defect" in lab:
                    for f in numeric_fields:
                        if "defect" in f["key"]:
                            kpi_targets.append(f["key"])

    preferred = [k for k in kpi_targets if k in {f["key"] for f in numeric_fields}]
    default_target = preferred[0] if preferred else (numeric_fields[0]["key"] if numeric_fields else "")
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
    meta["predictConfig"] = build_predict_config(out)
    out["meta"] = meta
    return out


def _unit_for_field(field: str, header: str = "") -> str:
    blob = f"{field} {header}".lower()
    if "pct" in blob or "rate" in blob or "yield" in blob or "oee" in blob:
        return "%"
    if "hours" in blob or "mtbf" in blob or "mttr" in blob or "downtime" in blob:
        return "h"
    if "price" in blob or "total" in blob or "revenue" in blob:
        return "$"
    return ""
