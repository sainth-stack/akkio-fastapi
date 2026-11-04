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
        if re.search(r"sku|skus", lab):
            return "128"
        if re.search(r"unit", lab) and re.search(r"hand|stock|on_hand", lab):
            return "3,420"
        if re.search(r"reorder|below", lab) or "low_stock" in lab:
            return "14"
        if "yield" in lab or "fpy" in lab or "first pass" in lab:
            return _fmt_pct(96.2)
        if "defect" in lab:
            return _fmt_pct(1.8)
        return "—"

    if re.search(r"sku|skus", lab) and re.search(r"track|total|count|active", lab):
        skus = {str(r.get("sku") or r.get("item_id") or i) for i, r in enumerate(rows) if r}
        return _fmt_count(len(skus) or len(rows))

    if (re.search(r"unit", lab) and re.search(r"hand|stock", lab)) or value_key in (
        "quantity_on_hand",
        "total_units_on_hand",
        "on_hand",
    ):
        total = sum(
            _num(r.get("quantity_on_hand")) or _num(r.get("on_hand")) or _num(r.get("qty")) or 0
            for r in rows
        )
        if total:
            return _fmt_count(total)

    if re.search(r"reorder|below", lab) or "low_stock" in lab:
        below = 0
        for r in rows:
            on_hand = _num(r.get("quantity_on_hand")) or _num(r.get("on_hand")) or 0
            reorder = _num(r.get("reorder_point")) or _num(r.get("min_stock")) or 0
            if on_hand is not None and reorder is not None and on_hand < reorder:
                below += 1
        return _fmt_count(below)

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
    series_source_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    series_source_map = series_source_map or {}
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    date_field = None
    if rows:
        for cand in ("inspected_at", "occurred_at", "ordered_at", "date"):
            if cand in rows[0]:
                date_field = cand
                break
    for r in rows:
        if x_field in r and x_field not in ("period", "date", "name", "label"):
            x = str(r.get(x_field) or "Unknown")
        elif date_field:
            x = str(r.get(date_field) or "")[:10] or f"Period {len(buckets) + 1}"
        else:
            x = f"Period {len(buckets) + 1}"
        buckets[x].append(r)
    keys = sorted(buckets.keys())[-max_points:]
    out: List[Dict[str, Any]] = []
    x_out = x_field if x_field not in ("name", "label") else "period"
    for x in keys:
        group = buckets[x]
        row: Dict[str, Any] = {x_out: x}
        main_val = 0.0
        for sf in series_fields:
            src = series_source_map.get(sf, sf)
            val = _series_numeric_value(group, src)
            row[sf] = val
            if "target" not in sf.lower() and "benchmark" not in sf.lower():
                main_val = val or main_val
        for sf in series_fields:
            src = series_source_map.get(sf, sf)
            if src == _BENCHMARK_METRIC or "benchmark" in sf.lower():
                if (row.get(sf) or 0) == 0 and main_val:
                    row[sf] = round(main_val * 1.08, 2)
            elif "target" in sf.lower() and (row.get(sf) or 0) == 0 and main_val:
                row[sf] = round(main_val * 1.02, 2)
        out.append(row)
    return out or [{x_out: "P1", **{sf: 0 for sf in series_fields}}]


def _first_row_key(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in row:
            return c
    for k in row:
        for c in candidates:
            if _slug(k) == _slug(c):
                return k
    return None


_COUNT_METRIC = "__count__"
_BENCHMARK_METRIC = "__benchmark__"

_TABLE_FIELD_ALIASES: Dict[str, List[str]] = {
    "id": ["work_order_id", "order_id", "item_id", "sku", "id"],
    "name": ["machine", "product_name", "equipment", "production_line", "name"],
    "updated_at": ["occurred_at", "ordered_at", "inspected_at", "updated_at", "date"],
}


def _table_field_value(row: Dict[str, Any], field: str) -> Any:
    if field in row and row[field] not in (None, ""):
        return row[field]
    for alt in _TABLE_FIELD_ALIASES.get(field, []):
        if alt in row and row[alt] not in (None, ""):
            return row[alt]
    for k, v in row.items():
        if _slug(k) == _slug(field) and v not in (None, ""):
            return v
    return row.get(field, "")


def _metric_from_title(title: str, sample: Dict[str, Any]) -> Optional[str]:
    title_l = (title or "").lower()
    if re.search(r"mttr|time to repair", title_l):
        return _first_row_key(sample, ["mttr_hours", "mttr"])
    if re.search(r"mtbf|between failures", title_l):
        return _first_row_key(sample, ["mtbf_hours", "mtbf"])
    if re.search(r"downtime", title_l):
        return _first_row_key(sample, ["downtime_hours", "downtime"])
    if re.search(r"event|work order|maintenance", title_l):
        return _COUNT_METRIC
    return _first_row_key(sample, ["downtime_hours", "mttr_hours", "mtbf_hours", "quantity_on_hand"])


def _series_numeric_value(group: List[Dict[str, Any]], src: str) -> float:
    if src == _COUNT_METRIC:
        return float(len(group))
    nums = [_num(g.get(src)) for g in group]
    nums = [n for n in nums if n is not None]
    if not nums:
        return 0.0
    return round(sum(nums) / len(nums), 2)


def _pick_breakdown_dimension(sample: Dict[str, Any], title: str, x_key: str) -> Optional[str]:
    title_l = (title or "").lower()
    if "machine" in title_l and "machine" in sample:
        return "machine"
    if x_key in sample and x_key not in ("name", "label", "category"):
        return x_key
    for dim in (
        "machine",
        "production_line",
        "event_type",
        "equipment",
        "warehouse",
        "category",
        "production_line",
    ):
        if dim in sample:
            return dim
    return None


def _series_source_map(block: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, str]:
    title = str(block.get("title") or "").lower()
    out: Dict[str, str] = {}
    for s in block.get("series") or []:
        if not isinstance(s, dict):
            continue
        sk = str(s.get("dataKey") or "")
        if not sk:
            continue
        name = str(s.get("name") or sk).lower()
        blob = f"{sk} {name} {title}"
        src = None
        if re.search(r"fpy|first.pass|\byield\b", blob):
            src = _first_row_key(sample, ["first_pass_yield_pct", "fpy", "yield_pct", "first_pass_yield"])
        elif re.search(r"defect", blob):
            src = _first_row_key(sample, ["defect_rate_pct", "defect_count", "defects", "defect_rate"])
        elif re.search(r"target", blob):
            src = _first_row_key(sample, ["target", "target_yield", "target_fpy", "target_defect_rate"])
        elif sk in (_BENCHMARK_METRIC, "benchmark") or "benchmark" in name:
            src = _BENCHMARK_METRIC
        elif sk in ("value", "actual") or name in ("value", "actual"):
            src = _metric_from_title(title, sample) or _first_row_key(sample, [sk, _slug(sk)])
        elif sk in sample:
            src = sk
        else:
            src = _first_row_key(sample, [sk, _slug(sk)])
        out[sk] = src or sk
    return out


def _chart_x_field(block: Dict[str, Any], sample: Dict[str, Any]) -> str:
    title = str(block.get("title") or "")
    title_l = title.lower()
    x_key = str(block.get("xAxisKey") or "period")
    if re.search(
        r"production line|by line|by category|by machine|pareto|defects by|breakdown|events by",
        title_l,
    ):
        dim = _pick_breakdown_dimension(sample, title, x_key)
        if dim:
            return dim
    if x_key in ("category", "name") and x_key not in sample:
        dim = _pick_breakdown_dimension(sample, title, x_key)
        if dim:
            return dim
    if re.search(r"breakdown", title_l):
        dim = _pick_breakdown_dimension(sample, title, x_key)
        if dim:
            return dim
    if x_key in sample and x_key not in ("name", "label"):
        return x_key
    return "period"


def _dedupe_series_keys(block: Dict[str, Any], sample: Dict[str, Any]) -> List[str]:
    smap = _series_source_map(block, sample)
    seen_src: set = set()
    out: List[str] = []
    for s in block.get("series") or []:
        if not isinstance(s, dict):
            continue
        sk = str(s.get("dataKey") or "")
        if not sk:
            continue
        src = smap.get(sk, sk)
        if src in seen_src:
            continue
        seen_src.add(src)
        out.append(sk)
    return out


def _pie_from_rows(block: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    title = str(block.get("title") or "").lower()
    if not rows:
        return []
    if re.search(r"production line|by line", title):
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[str(r.get("production_line") or "Other")].append(r)
        metric = "defect_count"
        if "yield" in title or "fpy" in title:
            metric = "first_pass_yield_pct"
        slices = []
        for line, grp in sorted(groups.items(), key=lambda x: -len(x[1]))[:6]:
            nums = [_num(g.get(metric)) for g in grp]
            nums = [n for n in nums if n is not None]
            val = round(sum(nums) / len(nums), 1) if nums else 0.0
            slices.append({"name": line, "value": val})
        return slices
    if re.search(r"yield|fpy|target", title):
        avg_fpy = _num(_compute_kpi_value("First pass yield", "first_pass_yield_pct", rows))
        if avg_fpy is None:
            vals = [_num(r.get("first_pass_yield_pct")) for r in rows]
            vals = [v for v in vals if v is not None]
            avg_fpy = sum(vals) / len(vals) if vals else 94.0
        target = min(100.0, avg_fpy * 1.02)
        gap = max(0.0, target - avg_fpy)
        return [
            {"name": "Actual FPY", "value": round(avg_fpy, 1)},
            {"name": "Gap to target", "value": round(gap, 1)},
            {"name": "Headroom", "value": round(max(0.0, 100 - target), 1)},
        ]
    return []


def _rows_match_schema(row: Dict[str, Any], columns: List[Dict[str, Any]]) -> bool:
    if not row:
        return False
    fields = {str(c.get("field") or "") for c in columns if isinstance(c, dict)}
    overlap = fields.intersection(row.keys())
    return len(overlap) >= max(2, len(fields) // 3)


def _aggregate_by_dimension(
    rows: List[Dict[str, Any]],
    dim: str,
    series_fields: List[str],
    series_source_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = str(r.get(dim) or "Other")
        groups[key].append(r)
    out: List[Dict[str, Any]] = []
    for dim_val, grp in sorted(groups.items(), key=lambda x: -len(x[1]))[:12]:
        row: Dict[str, Any] = {dim: dim_val}
        main_val = 0.0
        for sf in series_fields:
            src = series_source_map.get(sf, sf)
            val = _series_numeric_value(grp, src)
            row[sf] = val
            if "target" not in sf.lower() and "benchmark" not in sf.lower():
                main_val = val or main_val
        for sf in series_fields:
            src = series_source_map.get(sf, sf)
            if src == _BENCHMARK_METRIC or "benchmark" in sf.lower():
                if (row.get(sf) or 0) == 0 and main_val:
                    row[sf] = round(main_val * 1.08, 2)
            elif "target" in sf.lower() and (row.get(sf) or 0) == 0 and main_val:
                row[sf] = round(main_val * 1.02, 2)
        out.append(row)
    return out


def _ensure_minimum_dashboard_charts(out: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """If dashboard only has KPIs, add default charts so the page is not empty."""
    from app_builder.services.page_layout_intent import resolve_dashboard_page_id

    if not rows:
        return out
    pages = list(out.get("pages") or [])
    nav = out.get("navigation") or []
    dash_id = resolve_dashboard_page_id(
        [(str(p.get("id") or ""), str(p.get("title") or "")) for p in pages if isinstance(p, dict)],
        [str(n.get("pageId") or "") for n in nav if isinstance(n, dict)],
    )
    if not dash_id:
        return out
    sample = rows[0]
    new_pages = []
    for page in pages:
        if not isinstance(page, dict) or str(page.get("id") or "") != dash_id:
            new_pages.append(page)
            continue
        blocks = list(page.get("blocks") or [])
        has_chart = any(
            isinstance(b, dict) and b.get("type") in ("lineChart", "barChart", "pieChart") for b in blocks
        )
        if has_chart:
            new_pages.append(page)
            continue
        bag = dict(out.get("data") or {})
        if "warehouse" in sample:
            bar_key = "chart_dashboard_stock_by_warehouse"
            blocks.append(
                {
                    "type": "barChart",
                    "title": "Stock by warehouse",
                    "dataKey": bar_key,
                    "xAxisKey": "warehouse",
                    "series": [{"dataKey": "quantity_on_hand", "name": "Units on hand"}],
                }
            )
            bag[bar_key] = _aggregate_by_dimension(
                rows,
                "warehouse",
                ["quantity_on_hand"],
                {"quantity_on_hand": "quantity_on_hand"},
            )
        line_key = "chart_dashboard_units_trend"
        blocks.append(
            {
                "type": "lineChart",
                "title": "Units on hand trend",
                "dataKey": line_key,
                "xAxisKey": "period",
                "series": [{"dataKey": "quantity_on_hand", "name": "Units on hand"}],
            }
        )
        bag[line_key] = _aggregate_line_rows(
            rows,
            "period",
            ["quantity_on_hand"],
            max_points=12,
            series_source_map={"quantity_on_hand": "quantity_on_hand"},
        )
        out["data"] = bag
        new_pages.append({**page, "blocks": blocks})
    out["pages"] = new_pages
    return out


def _seed_source_rows_in_bag(defn: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure every declared source dataset has rows before KPI/chart compute."""
    from app_builder.services.source_data_constants import DEFAULT_SOURCE_SEED_ROWS
    from app_builder.services.source_data_seeder import seed_rows_for_dataset

    out = dict(defn)
    bag = dict(out.get("data") or out.get("mock") or {})
    meta = dict(out.get("meta") or {})
    user_uploaded = str(meta.get("dataSource") or "") == "upload"
    changed = False
    for ds in get_source_datasets(out):
        if not isinstance(ds, dict):
            continue
        dk = str(ds.get("dataKey") or "")
        cols = ds.get("columns") or []
        if not dk or not cols:
            continue
        existing = bag.get(dk) if isinstance(bag.get(dk), list) else None
        if existing and cols and not _rows_match_schema(existing[0], cols):
            bag.pop(dk, None)
            existing = None
            changed = True
        if user_uploaded and isinstance(existing, list) and existing:
            continue
        if not isinstance(bag.get(dk), list) or not bag.get(dk):
            bag[dk] = seed_rows_for_dataset(
                ds,
                row_count=DEFAULT_SOURCE_SEED_ROWS,
                human_headers=False,
            )
            changed = True
    if changed:
        out["data"] = bag
        out.pop("mock", None)
    return out


def recompute_derived_data(defn: Dict[str, Any]) -> Dict[str, Any]:
    """Fill KPI bundles, analytics tables, and chart series from source_* rows."""
    out = _seed_source_rows_in_bag(defn)
    bag = dict(out.get("data") or out.get("mock") or {})
    _, rows = _primary_source_rows(out)

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
                items = list(block.get("items") or [])
                colors = ("primary", "success", "warning", "error", "info", "default")
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("label") or "")
                    vk = str(item.get("valueKey") or _slug(label))
                    bundle[vk] = _compute_kpi_value(label, vk, rows)
                    item.setdefault("valueKey", vk)
                if not items and bundle:
                    pass
                if not items:
                    for i, vk in enumerate(list(bundle.keys()) if bundle else []):
                        items.append(
                            {
                                "label": vk.replace("_", " ").title(),
                                "valueKey": vk,
                                "color": colors[i % len(colors)],
                            }
                        )
                if not bundle and items:
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        label = str(item.get("label") or "")
                        vk = str(item.get("valueKey") or _slug(label))
                        bundle[vk] = _compute_kpi_value(label, vk, rows)
                if not items:
                    # Fallback KPIs from common quality / maintenance metrics
                    fallbacks = [
                        ("First Pass Yield (FPY)", "first_pass_yield_pct"),
                        ("Defect Rate", "defect_rate_pct"),
                        ("Units Inspected", "units_inspected"),
                    ]
                    for i, (lab, vk) in enumerate(fallbacks):
                        if rows and vk in rows[0]:
                            items.append({"label": lab, "valueKey": vk, "color": colors[i % len(colors)]})
                            bundle[vk] = _compute_kpi_value(lab, vk, rows)
                block["items"] = items
                bag[dk] = bundle
                continue

            if btype == "dataTable":
                cols = block.get("columns") or []
                fields = [str(c.get("field")) for c in cols if isinstance(c, dict) and c.get("field")]
                table_rows = []
                for r in rows[:24]:
                    if fields:
                        table_rows.append({f: _table_field_value(r, f) for f in fields})
                    else:
                        table_rows.append(dict(r))
                bag[dk] = table_rows
                continue

            if btype in ("lineChart", "barChart"):
                sample = rows[0] if rows else {}
                series_keys = _dedupe_series_keys(block, sample)
                if not series_keys:
                    series_keys = ["value"]
                smap = _series_source_map(block, sample)
                x_field = _chart_x_field(block, sample)
                if x_field in sample and x_field not in ("period", "date"):
                    bag[dk] = _aggregate_by_dimension(rows, x_field, series_keys, smap)
                    block["xAxisKey"] = x_field
                else:
                    bag[dk] = _aggregate_line_rows(
                        rows,
                        "period",
                        series_keys,
                        max_points=14,
                        series_source_map=smap,
                    )
                    block["xAxisKey"] = str(block.get("xAxisKey") or "period")
                    if block["xAxisKey"] in ("name", "label"):
                        block["xAxisKey"] = "period"
                continue

            if btype == "pieChart":
                pie = _pie_from_rows(block, rows)
                if pie:
                    bag[dk] = pie
                continue

    out["data"] = bag
    out.pop("mock", None)
    out = _ensure_minimum_dashboard_charts(out, rows)
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
    from app_builder.services.source_data_constants import (
        DEFAULT_SOURCE_SEED_ROWS,
        MAX_AUTO_SOURCE_ROWS,
    )
    from app_builder.services.source_data_seeder import seed_rows_for_dataset
    from app_builder.services.source_dataset_planner import ensure_source_datasets_on_definition

    out = ensure_source_datasets_on_definition(defn, requirement=requirement, prd=prd)
    bag = dict(out.get("data") or out.get("mock") or {})
    meta = dict(out.get("meta") or {})
    user_uploaded = str(meta.get("dataSource") or "") == "upload"
    active_keys = {str(ds.get("dataKey") or "") for ds in get_source_datasets(out)}
    for key in list(bag.keys()):
        if str(key).startswith(SOURCE_PREFIX) and key not in active_keys:
            bag.pop(key, None)
    for ds in get_source_datasets(out):
        dk = str(ds.get("dataKey") or "")
        cols = ds.get("columns") or []
        if not dk:
            continue
        existing = bag.get(dk) if isinstance(bag.get(dk), list) else None
        if existing and cols and not _rows_match_schema(existing[0], cols):
            existing = None
            bag.pop(dk, None)
        if (
            not user_uploaded
            and isinstance(bag.get(dk), list)
            and len(bag[dk]) > MAX_AUTO_SOURCE_ROWS
        ):
            bag[dk] = bag[dk][:MAX_AUTO_SOURCE_ROWS]
        if seed_if_empty and (not isinstance(bag.get(dk), list) or not bag.get(dk)):
            bag[dk] = seed_rows_for_dataset(
                ds,
                row_count=DEFAULT_SOURCE_SEED_ROWS,
                human_headers=False,
            )
    out["data"] = bag
    out.pop("mock", None)
    out = _repair_definition_pages(out, prd=prd, requirement=requirement)
    from app_builder.services.predict_config import attach_predict_config

    out = attach_predict_config(out)
    out = recompute_derived_data(out)
    from app_builder.services.navigation_labels import normalize_definition_navigation

    return normalize_definition_navigation(out)


def _repair_definition_pages(defn: Dict[str, Any], *, prd: str = "", requirement: str = "") -> Dict[str, Any]:
    """Rebuild empty page blocks from PRD (hosted apps often ship with dashboard-only blocks)."""
    try:
        from app_builder.schemas.app_definition import AppDefinition
        from app_builder.services.page_layout_intent import (
            enforce_page_role_blocks,
            repair_pages_with_no_blocks,
        )

        model = AppDefinition.model_validate(defn)
        model = repair_pages_with_no_blocks(model, prd=prd, requirement=requirement)
        model = enforce_page_role_blocks(model, prd=prd)
        from app_builder.services.dashboard_layout import apply_dashboard_analytics_product_rules

        model = apply_dashboard_analytics_product_rules(model, prd=prd)
        dumped = model.model_dump(mode="json")
        from app_builder.services.app_definition_validator import validate_all_screens

        return validate_all_screens(dumped, prd=prd, requirement=requirement)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("[pipeline] page repair skipped: %s", exc)
        return defn
