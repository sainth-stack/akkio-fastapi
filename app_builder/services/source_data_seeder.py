"""Realistic sample rows for source dataset schemas."""
from __future__ import annotations

import random
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List

_OPERATORS = ("J. Martinez", "A. Chen", "R. Patel", "S. Okafor", "L. Nguyen")
_MACHINES = ("Press-12", "Welder-3", "CNC-7", "Assembly-2", "Test-Station-4")
_LINES = ("Line A", "Line B", "Line C", "Pack-out")
_STATUSES = ("Open", "In progress", "Closed", "Scheduled")
_EVENT_TYPES = ("Breakdown", "Preventive", "Inspection", "Calibration")


def _seed(key: str) -> random.Random:
    import hashlib

    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def _cell(field: str, header: str, i: int, rng: random.Random) -> Any:
    f = field.lower()
    h = header.lower()
    if "work_order" in f or f == "id" and "order" in h:
        return f"WO-{1000 + i}"
    if "inspection" in f and "id" in f:
        return f"INS-{2000 + i}"
    if "order_id" in f:
        return f"ORD-{3000 + i}"
    if "customer" in f:
        return f"CUST-{100 + i % 40}"
    if "machine" in f or "equipment" in f:
        return rng.choice(_MACHINES)
    if "production_line" in f or f == "line":
        return rng.choice(_LINES)
    if "product" in f and "line" not in f:
        return rng.choice(("Assembly A", "Module B", "Unit C", "SKU-440"))
    if "event_type" in f:
        return rng.choice(_EVENT_TYPES)
    if "root_cause" in f:
        return rng.choice(("Wear", "Misalignment", "Operator error", "Material defect", "Scheduled PM"))
    if "severity" in f:
        return rng.choice(("Low", "Medium", "High"))
    if "status" in f:
        return rng.choice(_STATUSES)
    if "shift" in f:
        return rng.choice(("Day", "Evening", "Night"))
    if "downtime" in f and "hour" in f:
        return round(rng.uniform(0.2, 8.5), 1)
    if "mtbf" in f:
        return int(rng.uniform(120, 520))
    if "mttr" in f:
        return round(rng.uniform(0.5, 4.5), 1)
    if "oee" in f or "yield" in f or "rate" in f or f.endswith("_pct"):
        return round(rng.uniform(82, 99.2), 1)
    if "units_inspected" in f or "units_passed" in f:
        return int(rng.uniform(200, 1200))
    if "defect" in f and "rate" not in f:
        return int(rng.uniform(0, 45))
    if "scrap" in f:
        return int(rng.uniform(0, 20))
    if "quantity" in f:
        return int(rng.uniform(1, 12))
    if "unit_price" in f or "order_total" in f or "price" in f:
        return round(rng.uniform(15, 450), 2)
    if "occurred" in f or "inspected_at" in f or "ordered_at" in f or "date" in f:
        base = datetime(2026, 6, 1) + timedelta(days=i % 28, hours=i % 12)
        return base.strftime("%Y-%m-%d %H:%M")
    if "operator" in f or "owner" in f:
        return rng.choice(_OPERATORS)
    return f"{header} {i + 1}"


def seed_rows_for_dataset(
    dataset: Dict[str, Any],
    *,
    row_count: int = 32,
    human_headers: bool = False,
) -> List[Dict[str, Any]]:
    cols = dataset.get("columns") or []
    dk = str(dataset.get("dataKey") or dataset.get("id") or "source")
    rng = _seed(dk)
    rows: List[Dict[str, Any]] = []
    for i in range(row_count):
        row: Dict[str, Any] = {}
        for c in cols:
            field = str(c.get("field") or "")
            header = str(c.get("header") or field)
            val = _cell(field, header, i, rng)
            key = header if human_headers else field
            row[key] = val
        rows.append(row)
    return rows
