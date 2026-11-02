"""Plan canonical source datasets per app (prompt-driven, not hardcoded KPI sheets)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

SOURCE_PREFIX = "source_"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def _ctx(defn: Dict[str, Any], requirement: str = "", prd: str = "") -> str:
    meta = defn.get("meta") or {}
    parts = [
        requirement,
        prd,
        str(meta.get("domain") or ""),
        str(meta.get("description") or ""),
        str(meta.get("appName") or ""),
    ]
    for page in defn.get("pages") or []:
        parts.append(str(page.get("title") or ""))
        parts.append(str(page.get("subtitle") or ""))
    return " ".join(parts).lower()


def _column_type(field: str, header: str) -> str:
    f = f"{field} {header}".lower()
    if any(k in f for k in ("date", "time", "at", "timestamp", "occurred")):
        return "datetime"
    if any(k in f for k in ("hours", "mtbf", "mttr", "downtime", "qty", "count", "output", "load", "oee", "pct", "rate")):
        return "number"
    return "string"


def _cols_from_table_block(block: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for col in block.get("columns") or []:
        if not isinstance(col, dict):
            continue
        field = str(col.get("field") or "").strip()
        if not field or field == "actions":
            continue
        header = str(col.get("header") or field.replace("_", " ").title())
        out.append({"field": field, "header": header, "type": _column_type(field, header)})
    return out


def _find_primary_table(defn: Dict[str, Any]) -> Tuple[Optional[str], List[Dict[str, str]]]:
    best: Tuple[Optional[str], List[Dict[str, str]]] = (None, [])
    for page in defn.get("pages") or []:
        if not isinstance(page, dict):
            continue
        pid = str(page.get("id") or "")
        for block in page.get("blocks") or []:
            if not isinstance(block, dict) or block.get("type") != "dataTable":
                continue
            cols = _cols_from_table_block(block)
            if len(cols) >= len(best[1]):
                best = (pid, cols)
    return best


def _maintenance_event_columns() -> List[Dict[str, str]]:
    from app_builder.services.prd_driven_layout import _maintenance_analytics_columns

    cols = _maintenance_analytics_columns()
    return [
        {
            "field": c["field"],
            "header": c.get("header") or c["field"],
            "type": _column_type(c["field"], str(c.get("header") or "")),
        }
        for c in cols
    ]


def _quality_event_columns() -> List[Dict[str, str]]:
    return [
        {"field": "inspection_id", "header": "Inspection ID", "type": "string"},
        {"field": "production_line", "header": "Production line", "type": "string"},
        {"field": "product", "header": "Product", "type": "string"},
        {"field": "units_inspected", "header": "Units inspected", "type": "number"},
        {"field": "units_passed", "header": "Units passed", "type": "number"},
        {"field": "defect_count", "header": "Defect count", "type": "number"},
        {"field": "scrap_units", "header": "Scrap units", "type": "number"},
        {"field": "first_pass_yield_pct", "header": "First pass yield (%)", "type": "number"},
        {"field": "defect_rate_pct", "header": "Defect rate (%)", "type": "number"},
        {"field": "shift", "header": "Shift", "type": "string"},
        {"field": "inspected_at", "header": "Inspected at", "type": "datetime"},
    ]


def _retail_order_columns() -> List[Dict[str, str]]:
    return [
        {"field": "order_id", "header": "Order ID", "type": "string"},
        {"field": "customer_id", "header": "Customer ID", "type": "string"},
        {"field": "product_name", "header": "Product", "type": "string"},
        {"field": "quantity", "header": "Quantity", "type": "number"},
        {"field": "unit_price", "header": "Unit price", "type": "number"},
        {"field": "order_total", "header": "Order total", "type": "number"},
        {"field": "status", "header": "Status", "type": "string"},
        {"field": "ordered_at", "header": "Ordered at", "type": "datetime"},
    ]


def infer_source_datasets(
    defn: Dict[str, Any],
    *,
    requirement: str = "",
    prd: str = "",
) -> List[Dict[str, Any]]:
    """
    Derive 1–2 source tables from PRD + page blocks (agent can replace this later).
    """
    existing = (defn.get("meta") or {}).get("sourceDatasets")
    if isinstance(existing, list) and existing:
        return existing

    ctx = _ctx(defn, requirement, prd)
    page_id, table_cols = _find_primary_table(defn)

    if table_cols:
        sid = "operational_records"
        if "maintenance" in ctx or "mtbf" in ctx or "work order" in ctx:
            sid = "maintenance_events"
        elif "quality" in ctx or "defect" in ctx or "yield" in ctx:
            sid = "quality_inspections"
        elif "retail" in ctx or "ecommerce" in ctx or "order" in ctx:
            sid = "sales_orders"
        return [
            {
                "id": sid,
                "dataKey": f"{SOURCE_PREFIX}{sid}",
                "title": "Operational records",
                "description": "Enter one row per event. Dashboard KPIs, analytics tables, and charts are computed from this data.",
                "columns": table_cols,
            }
        ]

    if re.search(r"maintenance|mtbf|mttr|work order|downtime|equipment", ctx):
        cols = _maintenance_event_columns()
        return [
            {
                "id": "maintenance_events",
                "dataKey": f"{SOURCE_PREFIX}maintenance_events",
                "title": "Maintenance & downtime events",
                "description": "One row per work order or downtime event. KPIs are calculated automatically.",
                "columns": cols,
            }
        ]

    if re.search(r"quality|defect|scrap|yield|inspection|manufacturing", ctx):
        return [
            {
                "id": "quality_inspections",
                "dataKey": f"{SOURCE_PREFIX}quality_inspections",
                "title": "Quality inspection records",
                "description": "One row per inspection lot. Yield and defect KPIs are calculated from this sheet.",
                "columns": _quality_event_columns(),
            }
        ]

    if re.search(r"retail|ecommerce|shop|store|cart", ctx):
        return [
            {
                "id": "sales_orders",
                "dataKey": f"{SOURCE_PREFIX}sales_orders",
                "title": "Sales order lines",
                "description": "One row per order line. Revenue and order KPIs are calculated from this sheet.",
                "columns": _retail_order_columns(),
            }
        ]

    cols = _maintenance_event_columns() if "plant" in ctx or "operations" in ctx else _quality_event_columns()
    return [
        {
            "id": "operational_events",
            "dataKey": f"{SOURCE_PREFIX}operational_events",
            "title": "Operational events",
            "description": "Source records for this app. Metrics on dashboards are computed from these rows.",
            "columns": cols,
        }
    ]


def get_source_datasets(defn: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = defn.get("meta") or {}
    raw = meta.get("sourceDatasets")
    return list(raw) if isinstance(raw, list) else []


def ensure_source_datasets_on_definition(
    defn: Dict[str, Any],
    *,
    requirement: str = "",
    prd: str = "",
) -> Dict[str, Any]:
    out = dict(defn)
    meta = dict(out.get("meta") or {})
    datasets = infer_source_datasets(out, requirement=requirement, prd=prd)
    meta["sourceDatasets"] = datasets
    meta.setdefault("dataModel", "source_compute_v1")
    out["meta"] = meta
    return out


def list_source_import_targets(defn: Dict[str, Any]) -> List[Dict[str, str]]:
    targets: List[Dict[str, str]] = []
    for ds in get_source_datasets(defn):
        if not isinstance(ds, dict):
            continue
        dk = str(ds.get("dataKey") or "")
        if not dk:
            continue
        targets.append(
            {
                "dataKey": dk,
                "blockType": "sourceTable",
                "pageId": "",
                "label": str(ds.get("title") or ds.get("id") or dk),
            }
        )
    return targets


def build_template_sheets(defn: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Excel sheets: human-readable headers + sample rows (not KPI label/value)."""
    from app_builder.services.source_data_seeder import seed_rows_for_dataset

    bag = defn.get("data") or defn.get("mock") or {}
    sheets: Dict[str, List[Dict[str, Any]]] = {}
    for ds in get_source_datasets(defn):
        if not isinstance(ds, dict):
            continue
        dk = str(ds.get("dataKey") or "")
        cols = ds.get("columns") or []
        if not dk or not cols:
            continue
        sheet_name = dk.replace(SOURCE_PREFIX, "")[:31]
        existing = bag.get(dk) if isinstance(bag.get(dk), list) else None
        if existing:
            rows = []
            for row in existing[:500]:
                if not isinstance(row, dict):
                    continue
                out_row = {}
                for c in cols:
                    field = str(c.get("field") or "")
                    header = str(c.get("header") or field)
                    out_row[header] = row.get(field, "")
                rows.append(out_row)
            sheets[sheet_name] = rows
        else:
            sheets[sheet_name] = seed_rows_for_dataset(ds, row_count=24, human_headers=True)
    return sheets


def map_upload_row_to_fields(row: Dict[str, Any], columns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map spreadsheet headers (human labels) back to schema field keys."""
    out: Dict[str, Any] = {}
    header_to_field = {}
    for c in columns:
        field = str(c.get("field") or "")
        header = str(c.get("header") or field)
        header_to_field[header.lower().strip()] = field
        header_to_field[_slug(header)] = field
        header_to_field[field] = field
    for key, val in row.items():
        k = str(key).strip()
        field = header_to_field.get(k.lower()) or header_to_field.get(_slug(k))
        if field:
            out[field] = val
    return out
