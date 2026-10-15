"""Post-process datasets for production density: KPI meta, row counts, status realism."""
from __future__ import annotations

import copy
import random
import re
from typing import Any, Dict, List, Optional

from app_builder.schemas.app_definition import AppDefinition

MIN_TABLE_ROWS = 28
MIN_CHART_POINTS = 16


def _seed(key: str) -> random.Random:
    import hashlib

    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:48]


def enhance_production_datasets(defn: AppDefinition) -> AppDefinition:
    bag: Dict[str, Any] = dict(defn.mock or {})

    for page in defn.pages:
        for block in page.blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            page_id = page.id

            if btype == "kpiRow":
                data_key = str(block.get("dataKey") or f"kpis_{page_id}")
                bundle = bag.get(data_key)
                if not isinstance(bundle, dict):
                    bundle = {}
                meta_key = f"{data_key}_meta"
                meta = bag.get(meta_key) if isinstance(bag.get(meta_key), dict) else {}
                rng = _seed(data_key)
                items = block.get("items") or []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    vk = str(item.get("valueKey") or _slug(str(item.get("label") or "")))
                    if vk and vk not in bundle:
                        bundle[vk] = _format_kpi_value(str(item.get("label") or vk), vk, rng)
                    item.setdefault("valueKey", vk)
                    item.setdefault("trendKey", f"{vk}_trend_pct")
                    item.setdefault("targetKey", f"{vk}_target")
                    if f"{vk}_trend_pct" not in meta:
                        meta[f"{vk}_trend_pct"] = round(rng.uniform(-4.5, 6.2), 1)
                    if f"{vk}_target" not in meta:
                        meta[f"{vk}_target"] = _target_for_label(str(item.get("label") or vk), bundle.get(vk), rng)
                bag[data_key] = bundle
                bag[meta_key] = meta
                bag[f"kpis_{page_id}_meta"] = meta

            elif btype in ("lineChart", "barChart"):
                dk = str(block.get("dataKey") or "")
                if not dk:
                    continue
                rows = bag.get(dk)
                series_keys = [
                    str(s.get("dataKey")) for s in (block.get("series") or []) if isinstance(s, dict) and s.get("dataKey")
                ]
                x_key = str(block.get("xAxisKey") or "period")
                if isinstance(rows, list) and series_keys:
                    bag[dk] = _extend_chart(rows, dk, series_keys, x_key, MIN_CHART_POINTS)

            elif btype == "pieChart":
                dk = str(block.get("dataKey") or "")
                if dk and (not isinstance(bag.get(dk), list) or len(bag.get(dk) or []) < 4):
                    rng = _seed(dk)
                    bag[dk] = [
                        {"name": f"Segment {i + 1}", "value": rng.randint(8, 32)}
                        for i in range(5)
                    ]

            elif btype == "dataTable":
                dk = str(block.get("dataKey") or "")
                cols = block.get("columns") or []
                fields = [str(c.get("field")) for c in cols if isinstance(c, dict) and c.get("field")]
                headers = {
                    str(c.get("field")): str(c.get("header") or "")
                    for c in cols
                    if isinstance(c, dict) and c.get("field")
                }
                if not dk or not fields:
                    continue
                rows = bag.get(dk)
                if isinstance(rows, list):
                    bag[dk] = _extend_table(rows, fields, dk, MIN_TABLE_ROWS, headers=headers)
                    bag[dk] = _polish_status_fields(bag[dk], fields)

    defn.mock = bag
    return defn


def _format_kpi_value(label: str, vk: str, rng: random.Random) -> str:
    low = f"{label} {vk}".lower()
    if "mttr" in low or "time to repair" in low:
        return f"{rng.uniform(0.6, 4.2):.1f} h"
    if "mtbf" in low or "between failures" in low:
        return f"{rng.uniform(140, 520):.0f} h"
    if "work order" in low:
        return f"{rng.randint(48, 412):,}"
    if "copq" in low or "$" in label or "cost" in low:
        return f"${rng.randint(18, 220)}k"
    if "rate" in low or "yield" in low or "oee" in low or "pass" in low:
        return f"{rng.uniform(88.5, 99.4):.1f}%"
    if "units" in low or "total" in low:
        return f"{rng.randint(8400, 128000):,}"
    if "issues" in low or "open" in low:
        return str(rng.randint(2, 19))
    return f"{rng.randint(120, 9999):,}"


def _target_for_label(label: str, current: Any, rng: random.Random) -> str:
    low = label.lower()
    if "rate" in low or "yield" in low or "%" in str(current):
        return f"{rng.uniform(90, 97):.1f}%"
    if "$" in str(current) or "copq" in low:
        return f"${rng.randint(10, 80)}k"
    return str(rng.randint(100, 5000))


def _extend_chart(
    rows: List[Dict[str, Any]],
    data_key: str,
    series_keys: List[str],
    x_key: str,
    target_len: int,
) -> List[Dict[str, Any]]:
    if len(rows) >= target_len:
        return rows
    rng = _seed(data_key)
    out = [dict(r) for r in rows if isinstance(r, dict)]
    i = len(out)
    while len(out) < target_len:
        base = out[i % len(out)] if out else {x_key: "2024-06-01"}
        row = copy.deepcopy(base)
        row[x_key] = f"2024-06-{(i % 28) + 1:02d}"
        for sk in series_keys:
            if isinstance(row.get(sk), (int, float)):
                row[sk] = round(float(row[sk]) + rng.uniform(-2, 2.5), 2)
        out.append(row)
        i += 1
    return out


def _extend_table(
    rows: List[Dict[str, Any]],
    fields: List[str],
    data_key: str,
    target_len: int,
    *,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    from app_builder.services.generic_dataset_builder import _cell_value, _table_has_placeholders

    headers = headers or {}
    if isinstance(rows, list) and _table_has_placeholders(rows):
        rows = []
    rng = _seed(data_key)
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows if isinstance(rows, list) else []):
        if not isinstance(r, dict):
            continue
        out.append({f: r.get(f, _cell_value(f, i, rng, headers.get(f, ""))) for f in fields})
    i = len(out)
    while len(out) < target_len:
        row = {f: _cell_value(f, i, rng, headers.get(f, "")) for f in fields}
        out.append(row)
        i += 1
    return out


def _polish_status_fields(rows: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    status_fields = [f for f in fields if "status" in f.lower() or f.lower() == "severity"]
    if not status_fields:
        return rows
    good = ("Passed", "Complete", "Normal", "Published", "Running", "Resolved", "Operational")
    warn = ("Warning", "Watch", "Draft", "Review", "Idle", "Scheduled", "In progress")
    bad = ("Critical", "Failed", "Rejected", "Elevated", "Breakdown")
    rng = random.Random(42)
    for i, row in enumerate(rows):
        for sf in status_fields:
            pick = rng.choice(good * 3 + warn * 2 + bad)
            row[sf] = pick
    return rows
