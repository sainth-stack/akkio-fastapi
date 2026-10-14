"""Fill missing datasets from block metadata only — no domain-specific catalogs."""
from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Dict, List, Set

from app_builder.schemas.app_definition import AppDefinition

MIN_TABLE_ROWS = 24
MIN_CHART_POINTS = 14


def _seed(key: str) -> random.Random:
    h = hashlib.sha256(key.encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _human_header(field: str) -> str:
    return re.sub(r"\b\w", lambda m: m.group(0).upper(), field.replace("_", " "))


def _cell_value(field: str, row_i: int, rng: random.Random, col_header: str = "") -> Any:
    f = field.lower()
    h = (col_header or field).lower()
    if f == "session_id" or f.endswith("session_id"):
        return f"SES-{4100 + row_i}"
    if f == "machine_id" or (f.endswith("_id") and "machine" in f):
        return f"MCH-{row_i % 24 + 1:02d}"
    if "batch_number" in f or f == "batch_number":
        return f"BTH-2024-{880 + row_i}"
    if "product_name" in f or f == "product_name":
        products = (
            "PCB Control Module",
            "Hydraulic Valve Body",
            "Sensor Housing v2",
            "Power Distribution Unit",
            "Motor Controller Board",
        )
        return products[row_i % len(products)]
    if "production_line" in f or ("line" in f and "production" in h):
        lines = ("Line 1 - SMT", "Line 2 - Assembly", "Line 3 - Test", "Line 4 - Packaging")
        return lines[row_i % len(lines)]
    if "defect_type" in f:
        return rng.choice(("Solder bridge", "Misalignment", "Scratch", "Missing component", "Contamination"))
    if "report_data" in f:
        return rng.choice(("Weekly FPY summary", "Batch release certificate", "Pareto defect study", "Line comparison"))
    if "date_range" in f:
        return f"Jun {(row_i % 4) + 1} – Jun {(row_i % 4) + 7}, 2024"
    if f == "id" or f.endswith("_id"):
        prefix = "".join(x[0].upper() for x in re.findall(r"[a-z]+", field)[:3]) or "ID"
        return f"{prefix}-{1000 + row_i}"
    if "date" in f or "time" in f or f in ("when", "occurred_at", "started_at", "updated_at"):
        day = (row_i % 28) + 1
        return f"2024-06-{day:02d}T{8 + row_i % 10:02d}:{(row_i * 7) % 60:02d}:00Z"
    if f == "status" or "status" in h:
        return rng.choice(["Active", "Complete", "Warning", "Critical", "Draft", "In Review"])
    if "severity" in f:
        return rng.choice(["Low", "Medium", "High", "Critical"])
    if "defect_rate" in f or (f == "defect_rate"):
        return f"{rng.uniform(0.4, 3.8):.1f}%"
    if "rate" in f or "percent" in f or f.endswith("_pct"):
        if "defect" in f or "scrap" in f or "rework" in f:
            return f"{rng.uniform(0.3, 4.2):.1f}%"
        return f"{rng.uniform(88.0, 99.2):.1f}%"
    if any(k in f for k in ("qty", "quantity", "units", "count", "passed", "failed", "inspected")):
        return rng.randint(50, 5000)
    if f in ("value", "reading", "score", "oee", "fpy"):
        return round(rng.uniform(70, 99.5), 1)
    if f in ("name", "title", "product", "report", "metric", "parameter"):
        return f"{_human_header(field)} {row_i + 1}"
    if "operator" in f or "owner" in f:
        return rng.choice(["A. Chen", "J. Martinez", "R. Patel", "S. Okafor", "L. Nguyen"])
    if "line" in f or "machine" in f or "batch" in f:
        return f"{_human_header(field.split('_')[0])}-{row_i % 12 + 1}"
    if f == "message" or f == "description":
        return f"Recorded event #{row_i + 1} for {_human_header(field)}"
    if f == "type":
        return rng.choice(["Info", "Warning", "Alarm", "Quality", "Maintenance"])
    if f == "trend":
        return rng.choice(["Up", "Down", "Flat"])
    if f == "duration":
        return f"{1 + row_i % 4}h {(row_i * 11) % 60}m"
    if f == "period":
        return rng.choice(["Jun W1", "Jun W2", "May", "Q2", "Daily"])
    if isinstance(field, str) and field:
        return f"{_human_header(field)} {row_i + 1}"
    return row_i + 1


def _kpi_display(label: str, value_key: str, rng: random.Random) -> str:
    vk = value_key.lower()
    if "rate" in vk or "yield" in vk or "oee" in vk or "pct" in vk:
        return f"{rng.uniform(82, 99.2):.1f}%"
    if "cost" in vk or "copq" in vk or "$" in label:
        return f"${rng.randint(12, 180)}k"
    if "units" in vk or "total" in vk or "count" in vk:
        return f"{rng.randint(1200, 48000):,}"
    if "time" in vk or "duration" in vk or "mtbf" in vk:
        return f"{rng.randint(1, 48)}h {rng.randint(0, 59)}m"
    if "open" in vk or "issues" in vk or "alerts" in vk:
        return str(rng.randint(0, 24))
    return f"{rng.randint(100, 9999):,}"


def _chart_rows(data_key: str, series_keys: List[str], x_key: str = "period") -> List[Dict[str, Any]]:
    rng = _seed(data_key)
    rows: List[Dict[str, Any]] = []
    for i in range(MIN_CHART_POINTS):
        row: Dict[str, Any] = {x_key: f"2024-06-{(i % 28) + 1:02d}"}
        for sk in series_keys:
            skl = sk.lower()
            if "target" in skl or "capacity" in skl or "benchmark" in skl:
                row[sk] = round(rng.uniform(85, 92), 1)
            elif "defect" in skl or "failed" in skl or "draft" in skl:
                row[sk] = max(1, int(30 - i + rng.randint(0, 3)))
            else:
                row[sk] = round(rng.uniform(40, 98) + i * 0.4, 1)
        rows.append(row)
    return rows


def _pie_rows(data_key: str) -> List[Dict[str, Any]]:
    rng = _seed(data_key)
    labels = ("Passed", "Reworked", "Scrapped", "Hold", "Pending review")
    return [{"name": labels[i % len(labels)], "value": rng.randint(8, 35)} for i in range(5)]


def _table_has_placeholders(rows: List[Any]) -> bool:
    for row in rows[:5]:
        if not isinstance(row, dict):
            continue
        for val in row.values():
            if isinstance(val, str) and (val.startswith("Sample ") or val.startswith("Item ")):
                return True
    return False


def _production_line_options(data_key: str) -> List[Dict[str, str]]:
    rng = _seed(data_key)
    lines = ("Line 1 - SMT", "Line 2 - Assembly", "Line 3 - Test", "Line 4 - Packaging", "All lines")
    return [{"value": f"line_{i}", "label": lines[i % len(lines)]} for i in range(len(lines))]


def _status_filter_options(data_key: str) -> List[Dict[str, str]]:
    rng = _seed(data_key)
    statuses = ("Passed", "Failed", "Rework", "In review", "Published", "Draft")
    return [{"value": f"st_{i}", "label": statuses[i % len(statuses)]} for i in range(len(statuses))]


def collect_data_keys(defn: AppDefinition) -> Set[str]:
    keys: Set[str] = set()
    for page in defn.pages:
        for block in page.blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            dk = block.get("dataKey")
            if dk:
                keys.add(str(dk))
            if btype == "filterBar":
                for flt in block.get("filters") or []:
                    if isinstance(flt, dict) and flt.get("optionsKey"):
                        keys.add(str(flt["optionsKey"]))
    return keys


def ensure_datasets_from_blocks(defn: AppDefinition, *, min_table_rows: int = MIN_TABLE_ROWS) -> AppDefinition:
    """Top up missing or undersized datasets using block definitions only."""
    bag: Dict[str, Any] = dict(defn.mock or {})
    llm_filled = getattr(defn.meta, "generatedBy", "") == "llm_pipeline"

    for page in defn.pages:
        for block in page.blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            dk = str(block.get("dataKey") or "")

            if btype == "kpiRow":
                data_key = dk or f"kpis_{page.id}"
                bundle = bag.get(data_key)
                items = block.get("items") or []
                if not isinstance(bundle, dict):
                    bundle = {}
                rng = _seed(data_key)
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    vk = str(item.get("valueKey") or "")
                    if vk and vk not in bundle:
                        bundle[vk] = _kpi_display(str(item.get("label") or vk), vk, rng)
                bag[data_key] = bundle
                bag[f"kpis_{page.id}"] = bundle

            elif btype in ("lineChart", "barChart") and dk:
                series = block.get("series") or []
                series_keys = [str(s.get("dataKey")) for s in series if isinstance(s, dict) and s.get("dataKey")]
                x_key = str(block.get("xAxisKey") or "period")
                existing = bag.get(dk)
                if not isinstance(existing, list) or len(existing) < MIN_CHART_POINTS // 2 or _needs_chart_refresh(existing, series_keys, x_key):
                    if series_keys:
                        bag[dk] = _chart_rows(dk, series_keys, x_key)

            elif btype == "pieChart" and dk:
                existing = bag.get(dk)
                if not isinstance(existing, list) or len(existing) < 3:
                    bag[dk] = _pie_rows(dk)

            elif btype == "dataTable" and dk:
                cols = block.get("columns") or []
                fields = [str(c.get("field")) for c in cols if isinstance(c, dict) and c.get("field")]
                headers = {str(c.get("field")): str(c.get("header") or "") for c in cols if isinstance(c, dict)}
                existing = bag.get(dk)
                if not fields:
                    continue
                placeholders = isinstance(existing, list) and _table_has_placeholders(existing)
                if llm_filled and isinstance(existing, list) and len(existing) >= 12 and not placeholders:
                    continue
                if placeholders or not isinstance(existing, list) or len(existing) < min_table_rows:
                    rng = _seed(dk)
                    bag[dk] = [
                        {
                            f: _cell_value(f, i, rng, headers.get(f, ""))
                            for f in fields
                        }
                        for i in range(min_table_rows)
                    ]

            elif btype == "filterBar":
                for flt in block.get("filters") or []:
                    if not isinstance(flt, dict):
                        continue
                    ok = flt.get("optionsKey")
                    if ok and ok not in bag:
                        ok_s = str(ok)
                        if "line" in ok_s.lower():
                            bag[ok_s] = _production_line_options(ok_s)
                        elif "status" in ok_s.lower():
                            bag[ok_s] = _status_filter_options(ok_s)
                        else:
                            rng = _seed(ok_s)
                            bag[ok_s] = [
                                {"value": f"v{i}", "label": f"Option {i + 1}"}
                                for i in range(6)
                            ]

    defn.mock = bag
    return defn


def _needs_chart_refresh(rows: List[Any], series_keys: List[str], x_key: str) -> bool:
    if not rows or not isinstance(rows[0], dict):
        return True
    if series_keys and not all(k in rows[0] for k in series_keys):
        return True
    if x_key not in rows[0]:
        return True
    return False
