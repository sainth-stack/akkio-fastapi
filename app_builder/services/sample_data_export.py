"""Export synthetic seed data to Excel — rows/columns aligned with generic sample templates."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

from app_builder.services.runtime_paths import resolve_project_root
from app_builder.services.sample_report_templates import build_synthetic_export_sheets
from app_builder.services.supply_chain_demo_data import supply_chain_seed_data
from app_builder.services.demo_api_data import build_default_seed


def _load_seed_and_meta(project_name: str) -> tuple[Dict[str, Any], str, str]:
    project_root = resolve_project_root(project_name)
    seed_path = os.path.join(project_root, "backend", "seed_data.json")
    seed: Dict[str, Any] = {}
    requirement = ""
    prd = ""
    if os.path.isfile(seed_path):
        with open(seed_path, "r", encoding="utf-8") as fh:
            seed = json.load(fh)
    meta_path = os.path.join(project_root, "builder_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        requirement = str(meta.get("requirement") or meta.get("prompt") or "")
        prd = str(meta.get("prd") or "")
    if not seed:
        seed = build_default_seed({}, requirement=requirement, prd=prd, project_id=project_name)
    return seed, requirement, prd


def build_sample_data_xlsx(project_name: str) -> str:
    """Write KPIs + Detail + Breakdown sheets; returns temp file path."""
    import pandas as pd

    seed, requirement, prd = _load_seed_and_meta(project_name)
    sheets = seed.get("sample_export")
    if not isinstance(sheets, dict) or not sheets:
        sheets = build_synthetic_export_sheets(seed, project_name, requirement, prd)

    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        wrote = False
        for sheet_name, rows in sheets.items():
            if not isinstance(rows, list) or not rows:
                continue
            if not all(isinstance(r, dict) for r in rows):
                continue
            pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name[:31], index=False)
            wrote = True
        if not wrote:
            pd.DataFrame([{"Label": "Sample", "Value": "No data"}]).to_excel(
                writer, sheet_name="KPIs", index=False
            )

    return path
