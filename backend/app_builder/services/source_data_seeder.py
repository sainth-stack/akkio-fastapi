"""Realistic sample rows for source dataset schemas (moderate row counts, domain-aware)."""
from __future__ import annotations

import random
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List

from app_builder.services.source_data_constants import DEFAULT_SOURCE_SEED_ROWS

_OPERATORS = ("J. Martinez", "A. Chen", "R. Patel", "S. Okafor", "L. Nguyen")
_MACHINES = ("Press-12", "Welder-3", "CNC-7", "Assembly-2", "Test-Station-4")
_LINES = ("Line A", "Line B", "Line C", "Pack-out")
_STATUSES = ("Open", "In progress", "Closed", "Scheduled")
_EVENT_TYPES = ("Breakdown", "Preventive", "Inspection", "Calibration")
_SKUS = ("SKU-1001", "SKU-2044", "SKU-3300", "SKU-4412", "SKU-5590")


def _seed(key: str) -> random.Random:
    import hashlib

    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def ensure_definition_seed_salt(defn: Dict[str, Any], *, project_id: str = "") -> str:
    """Stable per-app salt so new apps do not all share identical demo rows."""
    meta = defn.get("meta") if isinstance(defn.get("meta"), dict) else {}
    existing = str(meta.get("seedSalt") or "").strip()
    if existing:
        return existing
    parts = [
        str(defn.get("id") or ""),
        project_id.strip(),
        str(meta.get("appName") or ""),
        str(meta.get("description") or ""),
    ]
    if any(p.strip() for p in parts):
        import hashlib

        salt = hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]
    else:
        salt = uuid.uuid4().hex[:24]
    meta = dict(meta)
    meta["seedSalt"] = salt
    defn["meta"] = meta
    return salt


def seed_rng_for_dataset(
    dataset: Dict[str, Any],
    *,
    seed_salt: str = "",
) -> random.Random:
    dk = str(dataset.get("dataKey") or dataset.get("id") or "source")
    key = f"{seed_salt}:{dk}" if seed_salt else dk
    return _seed(key)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def _cell(field: str, header: str, i: int, rng: random.Random) -> Any:
    f = field.lower()
    h = header.lower()
    if "ticket_id" in f:
        year = 2025 + (i % 24) // 12
        return f"TKT-{year}-{i + 1:04d}"
    if "title" in f and "ticket" in h:
        return rng.choice(
            (
                "IDoc status 51 on goods receipt",
                "BAPI timeout on order create",
                "Workflow stuck in SAP inbox",
                "Authorization error on RFC",
            )
        )
    if "description" in f:
        return "SAP incident details for support triage and similar-ticket search."
    if f == "category":
        return rng.choice(("Materials Management", "Sales & Distribution", "Finance", "PP"))
    if f == "component":
        return rng.choice(("IDoc", "BAPI", "RFC", "User exit", "Workflow", "Authorization"))
    if f == "priority":
        return rng.choice(("Low", "Medium", "High"))
    if "reported_by" in f or "reported" in f:
        return rng.choice(_OPERATORS)
    if "work_order" in f or f == "id" and "order" in h:
        return f"WO-{1000 + i}"
    if "inspection" in f and "id" in f:
        year = 2025 + (i % 24) // 12
        return f"INS-{year}-{i + 1:04d}"
    if "order_id" in f and "work" not in f:
        return f"ORD-{3000 + i}"
    if "customer" in f:
        return f"CUST-{100 + i % 40}"
    if "sku" in f or f == "item_id":
        return rng.choice(_SKUS)
    if "warehouse" in f or "location" in f:
        return rng.choice(("WH-A", "WH-B", "Plant-1", "Staging"))
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
        return rng.choice(_STATUSES if "order" not in f else ("Pending", "Shipped", "Delivered", "Cancelled"))
    if "shift" in f:
        return rng.choice(("Day", "Evening", "Night"))
    if "downtime" in f and "hour" in f:
        return round(rng.uniform(0.2, 6.5), 1)
    if "mtbf" in f:
        return int(rng.uniform(180, 480))
    if "mttr" in f:
        return round(rng.uniform(0.5, 3.5), 1)
    if "defect_rate" in f:
        return round(rng.uniform(0.6, 3.8), 2)
    if "first_pass" in f or f == "first_pass_yield_pct":
        return round(rng.uniform(94.0, 99.2), 1)
    if "oee" in f:
        return round(rng.uniform(78, 92), 1)
    if f.endswith("_pct") and "defect" not in f and "yield" not in f:
        return round(rng.uniform(85, 98), 1)
    if "units_inspected" in f:
        return int(rng.uniform(350, 950))
    if "units_passed" in f:
        return 0  # filled in finalize
    if "defect" in f and "rate" not in f:
        return 0  # filled in finalize
    if "scrap" in f:
        return int(rng.uniform(0, 12))
    if "on_hand" in f or "quantity_on_hand" in f or f == "qty":
        return int(rng.uniform(40, 620))
    if "reorder" in f or "min_stock" in f or "safety_stock" in f:
        return int(rng.uniform(15, 120))
    if "quantity" in f:
        return int(rng.uniform(1, 12))
    if "unit_price" in f or "order_total" in f or "price" in f:
        return round(rng.uniform(15, 450), 2)
    if "occurred" in f or "inspected_at" in f or "ordered_at" in f or "date" in f or f.endswith("_at"):
        # Spread rows across Jan 2025–Dec 2026. Never June-only and never year 2000.
        month_index = i % 24
        year = 2025 + month_index // 12
        month = (month_index % 12) + 1
        day = (i % 27) + 1
        base = datetime(year, month, day, 8 + (i % 10), (i * 7) % 60)
        return base.strftime("%Y-%m-%d %H:%M")
    if "operator" in f or "owner" in f:
        return rng.choice(_OPERATORS)
    return f"{header} {i + 1}"


def _finalize_quality_row(row: Dict[str, Any], rng: random.Random) -> None:
    inspected = int(row.get("units_inspected") or rng.randint(400, 900))
    scrap = int(row.get("scrap_units") or 0)
    defect = int(row.get("defect_count") or 0)
    if defect <= 0:
        defect = max(1, int(inspected * rng.uniform(0.006, 0.032)))
    passed = max(0, inspected - defect - scrap)
    row["units_inspected"] = inspected
    row["defect_count"] = defect
    row["units_passed"] = passed
    row["scrap_units"] = scrap
    row["first_pass_yield_pct"] = round(100.0 * passed / inspected, 2) if inspected else 0.0
    row["defect_rate_pct"] = round(100.0 * defect / inspected, 2) if inspected else 0.0


def _finalize_maintenance_row(row: Dict[str, Any], rng: random.Random) -> None:
    if not row.get("downtime_hours"):
        row["downtime_hours"] = round(rng.uniform(0.3, 5.0), 1)
    if not row.get("mtbf_hours"):
        row["mtbf_hours"] = int(rng.uniform(200, 450))
    if not row.get("mttr_hours"):
        row["mttr_hours"] = round(rng.uniform(0.8, 2.8), 1)


def _finalize_procurement_row(row: Dict[str, Any], rng: random.Random) -> None:
    if not row.get("order_value"):
        row["order_value"] = round(rng.uniform(4200, 185000), 2)
    if not row.get("lines"):
        row["lines"] = rng.randint(1, 24)
    if not row.get("lead_time_days"):
        row["lead_time_days"] = rng.randint(3, 28)
    if not row.get("status"):
        row["status"] = rng.choice(["Draft", "Approved", "In Transit", "Received", "Closed"])
    if not row.get("vendor"):
        row["vendor"] = rng.choice(
            ["Apex Metals", "Nova Components", "Grid Industrial", "Precision Forge Co."]
        )
    if not row.get("po_number"):
        row["po_number"] = f"PO-{rng.randint(8800, 9999)}"


def _finalize_inventory_row(row: Dict[str, Any], rng: random.Random) -> None:
    on_hand = int(row.get("quantity_on_hand") or row.get("on_hand") or row.get("qty") or rng.randint(50, 500))
    reorder = int(row.get("reorder_point") or row.get("min_stock") or max(10, on_hand // 4))
    row["quantity_on_hand"] = on_hand
    row["on_hand"] = on_hand
    if "reorder_point" in row or "min_stock" in row:
        row["reorder_point"] = reorder
        row["min_stock"] = reorder


def _finalize_retail_row(row: Dict[str, Any], rng: random.Random) -> None:
    qty = int(row.get("quantity") or 1)
    price = float(row.get("unit_price") or rng.uniform(20, 180))
    row["quantity"] = qty
    row["unit_price"] = round(price, 2)
    row["order_total"] = round(qty * price, 2)


def _finalize_row(row: Dict[str, Any], dataset: Dict[str, Any], rng: random.Random) -> None:
    sid = f"{dataset.get('id') or ''} {dataset.get('dataKey') or ''}".lower()
    fields = {str(c.get("field") or "").lower() for c in (dataset.get("columns") or [])}
    if "quality" in sid or "first_pass_yield_pct" in fields:
        _finalize_quality_row(row, rng)
    elif "maintenance" in sid or "work_order_id" in fields:
        _finalize_maintenance_row(row, rng)
    elif "purchase" in sid or "procurement" in sid or "po_number" in fields:
        _finalize_procurement_row(row, rng)
    elif "inventory" in sid or "quantity_on_hand" in fields or "on_hand" in fields:
        _finalize_inventory_row(row, rng)
    elif "sales" in sid or "order_id" in fields:
        _finalize_retail_row(row, rng)


def seed_rows_for_dataset(
    dataset: Dict[str, Any],
    *,
    row_count: int = DEFAULT_SOURCE_SEED_ROWS,
    human_headers: bool = False,
    seed_salt: str = "",
) -> List[Dict[str, Any]]:
    cols = dataset.get("columns") or []
    rng = seed_rng_for_dataset(dataset, seed_salt=seed_salt)
    rows: List[Dict[str, Any]] = []
    for i in range(max(1, min(row_count, 48))):
        row: Dict[str, Any] = {}
        for c in cols:
            field = str(c.get("field") or "")
            header = str(c.get("header") or field)
            val = _cell(field, header, i, rng)
            key = header if human_headers else field
            row[key] = val
        if not human_headers:
            _finalize_row(row, dataset, rng)
        rows.append(row)
    return rows
