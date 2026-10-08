"""Export backend/seed_data.json to Excel for builder download."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

from app_builder.services.runtime_paths import resolve_project_root
from app_builder.services.supply_chain_demo_data import supply_chain_seed_data


def _flatten_sheet_rows(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return [dict(x) for x in data["items"] if isinstance(x, dict)]
    if isinstance(data, list):
        return [dict(x) for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def build_sample_data_xlsx(project_name: str) -> str:
    """Write a multi-sheet xlsx to a temp file; returns path."""
    import pandas as pd

    project_root = resolve_project_root(project_name)
    seed_path = os.path.join(project_root, "backend", "seed_data.json")
    seed: Dict[str, Any] = {}
    if os.path.isfile(seed_path):
        with open(seed_path, "r", encoding="utf-8") as fh:
            seed = json.load(fh)
    if not seed:
        seed = supply_chain_seed_data(project_name)

    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        dashboard = seed.get("dashboard") or seed.get("kpis")
        if isinstance(dashboard, dict):
            pd.DataFrame([dashboard]).to_excel(writer, sheet_name="Dashboard", index=False)

        for sheet_name, key in (
            ("Materials", "materials"),
            ("Suppliers", "suppliers"),
            ("PurchaseOrders", "purchase_orders"),
            ("Alerts", "alerts"),
            ("InventoryAnomalies", "anomalies"),
            ("Thresholds", "thresholds"),
        ):
            block = seed.get(key)
            rows = _flatten_sheet_rows(block)
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name[:31], index=False)

        reports = seed.get("reports/summary")
        if isinstance(reports, dict):
            aging = reports.get("inventory_aging")
            if isinstance(aging, list) and aging:
                pd.DataFrame(aging).to_excel(writer, sheet_name="InventoryAging", index=False)
            turnover = reports.get("turnover_by_category")
            if isinstance(turnover, list) and turnover:
                pd.DataFrame(turnover).to_excel(writer, sheet_name="Turnover", index=False)

    return path
