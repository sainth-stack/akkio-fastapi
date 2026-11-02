"""Compute dashboard KPIs, tables, and charts from canonical source datasets."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from app_builder.services.source_dataset_planner import SOURCE_PREFIX, get_source_datasets


def _num(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip().replace(",", "")
        m = re.search(r"-?\d+\.?\d*", s)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def _fmt_count(n: float) -> str:
    return f"{int(round(n)):,}"


def _fmt_hours(n: float) -> str:
    return f"{n:.1f} h"


def _fmt_pct(n: float) -> str:
    return f"{n:.1f}%"


def _primary_source_rows(defn: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    bag = defn.get("data") or defn.get("mock") or {}
    for ds in get_source_datasets(defn):
        dk = str(ds.get("dataKey") or "")
        rows = bag.get(dk)
        if isinstance(rows, list) and rows:
            return dk, rows
    for key, val in bag.items():
        if str(key).startswith(SOURCE_PREFIX) and isinstance(val, list) and val:
            return str(key), val
    return "", []


def _compute_kpi_value(label: str, value_key: str, rows: List[Dict[str, Any]]) -> str:
    lab = f"{label} {value_key}".lower()
    if not rows:
        return "—"

    if "work order" in lab or "open" in lab and "order" in lab:
        ids = {str(r.get("work_order_id") or r.get("order_id") or i) for i, r in enumerate(rows)}
        if "open" in lab:
            open_rows = [r for r in rows if str(r.get("status", "")).lower() in ("open", "in progress", "pending")]
            return _fmt_count(len(open_rows) or len(ids))
        return _fmt_count(len(ids))

    if "mtbf" in lab or "between failures" in lab:
        vals = [_num(r.get("mtbf_hours")) for r in rows]
        nums = [v for v in vals if v is not None]
        if nums:
            return _fmt_hours(sum(nums) / len(nums))
        return _fmt_hours(280.0)

    if "mttr" in lab or "time to repair" in lab:
        vals = [_num(r.get("mttr_hours")) for r in rows]
        nums = [v for v in vals if v is not None]
        if nums:
            return _fmt_hours(sum(nums) / len(nums))
        return _fmt_hours(2.1)

    if "downtime" in lab:
        vals = [_num(r.get("downtime_hours")) for r in rows]
        nums = [v for v in vals if v is not None]
        if nums:
            total = sum(nums)
            if total >= 24:
                return f"{int(total // 24)}d {int(total % 24)}h"
            return _fmt_hours(total)
        return _fmt_hours(12.0)

    if "oee" in lab or "equipment effectiveness" in lab:
        vals = [_num(r.get("oee_pct")) or _num(r.get("oee")) for r in rows]
        nums = [v for v in vals if v is not None]
        if nums:
            return _fmt_pct(sum(nums) / len(nums))
        return _fmt_pct(86.5)

    if "yield" in lab or "first pass" in lab:
        vals = [_num(r.get("first_pass_yield_pct")) for r in rows]
        nums = [v for v in vals if v is not None]
        if nums:
            return _fmt_pct(sum(nums) / len(nums))
        passed = sum(_num(r.get("units_passed")) or 0 for r in rows)
        inspected = sum(_num(r.get("units_inspected")) or 0 for r in rows)
        if inspected:
            return _fmt_pct(100.0 * passed / inspected)
        return _fmt_pct(94.0)

    if "defect" in lab and "rate" in lab:
        vals = [_num(r.get("defect_rate_pct")) for r in rows]
        nums = [v for v in vals if v is not None]
        if nums:
            return _fmt_pct(sum(nums) / len(nums))
        return _fmt_pct(2.4)

    if "revenue" in lab or "sales" in lab:
        totals = [_num(r.get("order_total")) or (_num(r.get("unit_price")) or 0) * (_num(r.get("quantity")) or 1) for r in rows]
        return f"${sum(totals):,.0f}"

    if "order" in lab and "total" not in lab:
        return _fmt_count(len(rows))

    vals = []
    for r in rows:
        for k, v in r.items():
            if _slug(k) == _slug(value_key) or k == value_key:
                n = _num(v)
                if n is not None:
                    vals.append(n)
    if vals:
        return _fmt_count(sum(vals) / len(vals))
    return _fmt_count(len(rows))


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def _aggregate_line_rows(
    rows: List[Dict[str, Any]],
    x_field: str,
    series_fields: List[str],
    *,
    max_points: int = 16,
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        x = str(r.get(x_field) or r.get("occurred_at") or r.get("inspected_at") or "")[:10]
        if not x:
            x = f"Period {len(buckets) + 1}"
        buckets[x].append(r)
    keys = sorted(buckets.keys())[-max_points:]
    out: List[Dict[str, Any]] = []
    for x in keys:
        group = buckets[x]
        row: Dict[str, Any] = {x_field if x_field != "period" else "period": x}
        for sf in series_fields:
            nums = [_num(g.get(sf)) for g in group]
            nums = [n for n in nums if n is not None]
            row[sf] = round(sum(nums) / len(nums), 2) if nums else 0
        out.append(row)
    return out or [{"period": "P1", **{sf: 0 for sf in series_fields}}]


def recompute_derived_data(defn: Dict[str, Any]) -> Dict[str, Any]:
    """Fill KPI bundles, analytics tables, and chart series from source_* rows."""
    out = dict(defn)
    bag = dict(out.get("data") or out.get("mock") or {})
    _, rows = _primary_source_rows(out)
    if not rows:
        out["data"] = bag
        out.pop("mock", None)
        return out

    for page in out.get("pages") or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            dk = str(block.get("dataKey") or "")
            if not dk or dk.startswith(SOURCE_PREFIX):
                continue

            if btype == "kpiRow":
                bundle: Dict[str, str] = {}
                for item in block.get("items") or []:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("label") or "")
                    vk = str(item.get("valueKey") or _slug(label))
                    bundle[vk] = _compute_kpi_value(label, vk, rows)
                bag[dk] = bundle
                continue

            if btype == "dataTable":
                cols = block.get("columns") or []
                fields = [str(c.get("field")) for c in cols if isinstance(c, dict) and c.get("field")]
                table_rows = []
                for r in rows[:500]:
                    if fields:
                        table_rows.append({f: r.get(f, "") for f in fields})
                    else:
                        table_rows.append(dict(r))
                bag[dk] = table_rows
                continue

            if btype in ("lineChart", "barChart"):
                x_key = str(block.get("xAxisKey") or "period")
                series_keys = [
                    str(s.get("dataKey"))
                    for s in (block.get("series") or [])
                    if isinstance(s, dict) and s.get("dataKey")
                ]
                if not series_keys:
                    series_keys = ["value"]
                date_field = "occurred_at"
                for cand in ("occurred_at", "inspected_at", "ordered_at", "date"):
                    if rows and cand in rows[0]:
                        date_field = cand
                        break
                if x_key == "period" or x_key == "date":
                    bag[dk] = _aggregate_line_rows(rows, "period", series_keys, max_points=14)
                else:
                    bag[dk] = _aggregate_line_rows(rows, x_key, series_keys, max_points=14)
                continue

    out["data"] = bag
    out.pop("mock", None)
    meta = dict(out.get("meta") or {})
    meta["dataSource"] = "computed_from_source"
    out["meta"] = meta
    return out


def apply_source_data_pipeline(
    defn: Dict[str, Any],
    *,
    requirement: str = "",
    prd: str = "",
    seed_if_empty: bool = True,
) -> Dict[str, Any]:
    from app_builder.services.source_data_seeder import seed_rows_for_dataset
    from app_builder.services.source_dataset_planner import ensure_source_datasets_on_definition

    out = ensure_source_datasets_on_definition(defn, requirement=requirement, prd=prd)
    bag = dict(out.get("data") or out.get("mock") or {})
    for ds in get_source_datasets(out):
        dk = str(ds.get("dataKey") or "")
        if not dk:
            continue
        if seed_if_empty and (not isinstance(bag.get(dk), list) or not bag.get(dk)):
            bag[dk] = seed_rows_for_dataset(ds, row_count=36, human_headers=False)
    out["data"] = bag
    out.pop("mock", None)
    return recompute_derived_data(out)
