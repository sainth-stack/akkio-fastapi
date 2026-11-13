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


def _column_fields(cols: List[Dict[str, Any]]) -> set:
    return {str(c.get("field") or "") for c in cols if isinstance(c, dict)}


def _columns_match_quality(cols: List[Dict[str, Any]]) -> bool:
    f = _column_fields(cols)
    return bool(f & {"first_pass_yield_pct", "defect_rate_pct", "units_inspected", "inspection_id"})


def _columns_match_maintenance(cols: List[Dict[str, Any]]) -> bool:
    f = _column_fields(cols)
    return bool(f & {"work_order_id", "mtbf_hours", "mttr_hours", "downtime_hours"})


def _domain_flags(ctx: str) -> Tuple[bool, bool, bool]:
    is_quality = bool(re.search(r"quality|defect|scrap|\byield\b|fpy|inspection|first pass", ctx))
    is_maint = bool(re.search(r"maintenance|mtbf|mttr|work order|downtime|equipment", ctx))
    is_retail = bool(re.search(r"retail|ecommerce|shop|store|cart|order line", ctx))
    return is_quality, is_maint, is_retail


def _pick_columns_for_context(
    table_cols: List[Dict[str, str]],
    ctx: str,
) -> List[Dict[str, str]]:
    is_quality, is_maint, is_retail = _domain_flags(ctx)
    is_inventory = bool(re.search(r"inventory|stock|sku|warehouse|reorder", ctx))
    if is_inventory and not is_retail and not is_quality:
        if "quantity_on_hand" not in _column_fields(table_cols) and "sku" not in _column_fields(table_cols):
            return _inventory_columns()
    if is_quality and not is_maint:
        if not _columns_match_quality(table_cols):
            return _quality_event_columns()
    if is_maint and not is_quality:
        if not _columns_match_maintenance(table_cols):
            return _maintenance_event_columns()
    if is_retail and not is_maint:
        if "order_id" not in _column_fields(table_cols):
            return _retail_order_columns()
    return table_cols


def _existing_datasets_mismatch(defn: Dict[str, Any], ctx: str) -> bool:
    existing = get_source_datasets(defn)
    if not existing or not isinstance(existing[0], dict):
        return False
    cols = existing[0].get("columns") or []
    is_quality, is_maint, _ = _domain_flags(ctx)
    is_inventory = bool(re.search(r"inventory|stock|sku|warehouse|reorder", ctx))
    if is_inventory and not is_maint and not is_quality:
        inv_fields = _column_fields(_inventory_columns())
        have = _column_fields(cols)
        if have and not (have & inv_fields):
            return True
    if is_quality and not is_maint and _columns_match_maintenance(cols) and not _columns_match_quality(cols):
        return True
    if is_maint and not is_quality and _columns_match_quality(cols) and not _columns_match_maintenance(cols):
        return True
    return False


def infer_source_datasets(
    defn: Dict[str, Any],
    *,
    requirement: str = "",
    prd: str = "",
) -> List[Dict[str, Any]]:
    """
    Derive 1–2 source tables from PRD + page blocks (agent can replace this later).
    """
    ctx = _ctx(defn, requirement, prd)
    existing = (defn.get("meta") or {}).get("sourceDatasets")
    if isinstance(existing, list) and existing and not _existing_datasets_mismatch(defn, ctx):
        return existing

    page_id, table_cols = _find_primary_table(defn)

    if table_cols:
        table_cols = _pick_columns_for_context(table_cols, ctx)
        sid = "operational_records"
        is_quality, is_maint, is_retail = _domain_flags(ctx)
        if is_maint and not is_quality:
            sid = "maintenance_events"
        elif is_quality and not is_maint:
            sid = "quality_inspections"
        elif is_retail:
            sid = "sales_orders"
        elif _columns_match_quality(table_cols):
            sid = "quality_inspections"
        elif _columns_match_maintenance(table_cols):
            sid = "maintenance_events"
        return [
            {
                "id": sid,
                "dataKey": f"{SOURCE_PREFIX}{sid}",
                "title": "Operational records",
                "description": "Enter one row per event. Dashboard KPIs, analytics tables, and charts are computed from this data.",
                "columns": table_cols,
            }
        ]

    if re.search(r"\bsap\b|similar ticket|service ticket", ctx):
        from app_builder.services.sap_similar_tickets import sap_ticket_columns

        cols = sap_ticket_columns()
        return [
            {
                "id": "sap_tickets",
                "dataKey": f"{SOURCE_PREFIX}sap_tickets",
                "title": "SAP service tickets",
                "description": "One row per incident. New tickets appear at the top; use Similar Tickets to match history.",
                "columns": cols,
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

    if re.search(r"inventory|stock|sku|warehouse", ctx):
        return [
            {
                "id": "inventory_items",
                "dataKey": f"{SOURCE_PREFIX}inventory_items",
                "title": "Inventory levels",
                "description": "One row per SKU/location. Stock KPIs are calculated from this sheet.",
                "columns": _inventory_columns(),
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


def _inventory_columns() -> List[Dict[str, str]]:
    return [
        {"field": "sku", "header": "SKU", "type": "string"},
        {"field": "product_name", "header": "Product", "type": "string"},
        {"field": "warehouse", "header": "Warehouse", "type": "string"},
        {"field": "quantity_on_hand", "header": "Qty on hand", "type": "number"},
        {"field": "reorder_point", "header": "Reorder point", "type": "number"},
        {"field": "unit_cost", "header": "Unit cost", "type": "number"},
        {"field": "status", "header": "Status", "type": "string"},
        {"field": "last_counted_at", "header": "Last counted", "type": "datetime"},
    ]


def build_template_sheets(
    defn: Dict[str, Any],
    *,
    row_count: int | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Excel sheets: human headers + moderate sample rows only (not live uploaded data)."""
    from app_builder.services.source_data_constants import TEMPLATE_SAMPLE_ROWS
    from app_builder.services.source_data_seeder import seed_rows_for_dataset

    sample_rows = row_count if row_count is not None else TEMPLATE_SAMPLE_ROWS
    sheets: Dict[str, List[Dict[str, Any]]] = {}
    for ds in get_source_datasets(defn):
        if not isinstance(ds, dict):
            continue
        dk = str(ds.get("dataKey") or "")
        cols = ds.get("columns") or []
        if not dk or not cols:
            continue
        sheet_name = dk.replace(SOURCE_PREFIX, "")[:31]
        sheets[sheet_name] = seed_rows_for_dataset(ds, row_count=sample_rows, human_headers=True)
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
