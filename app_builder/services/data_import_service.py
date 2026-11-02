"""Merge user-uploaded source data and recompute KPIs / tables / charts."""
from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

_DATA_JSON_REL = "frontend/src/mock/data.json"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def _data_bag(defn: Dict[str, Any]) -> Dict[str, Any]:
    bag = defn.get("data") or defn.get("mock") or {}
    return dict(bag) if isinstance(bag, dict) else {}


def _set_data_bag(defn: Dict[str, Any], bag: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defn)
    out["data"] = bag
    out.pop("mock", None)
    meta = dict(out.get("meta") or {})
    meta["dataSource"] = "upload"
    out["meta"] = meta
    return out


def list_import_targets(defn: Dict[str, Any]) -> List[Dict[str, str]]:
    """Upload targets = source datasets only (not derived KPI/chart keys)."""
    from app_builder.services.source_dataset_planner import list_source_import_targets

    targets = list_source_import_targets(defn)
    if targets:
        return targets
    return _legacy_block_targets(defn)


def _legacy_block_targets(defn: Dict[str, Any]) -> List[Dict[str, str]]:
    targets: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for page in defn.get("pages") or []:
        if not isinstance(page, dict):
            continue
        page_id = str(page.get("id") or "")
        for block in page.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            dk = str(block.get("dataKey") or "")
            if not dk or dk in seen:
                continue
            seen.add(dk)
            targets.append(
                {
                    "dataKey": dk,
                    "blockType": str(block.get("type") or ""),
                    "pageId": page_id,
                    "label": str(block.get("title") or block.get("type") or dk),
                }
            )
    return targets


def _source_dataset_by_key(defn: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    from app_builder.services.source_dataset_planner import get_source_datasets

    key_slug = _slug(key)
    for ds in get_source_datasets(defn):
        dk = str(ds.get("dataKey") or "")
        sid = str(ds.get("id") or "")
        if key == dk or _slug(dk) == key_slug or _slug(sid) == key_slug or key_slug in _slug(dk):
            return ds
    return None


def _match_source_key(defn: Dict[str, Any], hint: str) -> Optional[str]:
    ds = _source_dataset_by_key(defn, hint)
    if ds:
        return str(ds.get("dataKey"))
    from app_builder.services.source_dataset_planner import get_source_datasets

    for ds in get_source_datasets(defn):
        dk = str(ds.get("dataKey") or "")
        sheet = dk.replace("source_", "")
        if _slug(hint) == _slug(sheet) or _slug(hint) == _slug(dk):
            return dk
    return None


def _parse_csv_text(text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text), dialect)
    rows = list(reader)
    if not rows:
        return [], []
    headers = [h.strip() for h in rows[0]]
    out: List[Dict[str, Any]] = []
    for row in rows[1:]:
        if not any(str(c).strip() for c in row):
            continue
        item: Dict[str, Any] = {}
        for i, h in enumerate(headers):
            if i >= len(row):
                break
            item[h] = _coerce_cell(row[i].strip())
        if item:
            out.append(item)
    return headers, out


def _coerce_cell(raw: str) -> Any:
    if raw == "":
        return ""
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if re.match(r"^-?\d+$", raw.replace(",", "")):
            return int(raw.replace(",", ""))
        if re.match(r"^-?\d+\.\d+$", raw.replace(",", "")):
            return float(raw.replace(",", ""))
    except ValueError:
        pass
    if raw.endswith("%"):
        try:
            return float(raw[:-1].replace(",", ""))
        except ValueError:
            return raw
    return raw


def _rows_for_source(defn: Dict[str, Any], dataset: Dict[str, Any], raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from app_builder.services.source_dataset_planner import map_upload_row_to_fields

    cols = dataset.get("columns") or []
    return [map_upload_row_to_fields(r, cols) for r in raw_rows if isinstance(r, dict)]


def merge_upload_bytes(
    defn: Dict[str, Any],
    *,
    filename: str,
    raw: bytes,
    target_key: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from app_builder.services.operational_metrics_compute import apply_source_data_pipeline, recompute_derived_data
    from app_builder.services.source_dataset_planner import get_source_datasets

    if not isinstance(defn, dict) or not defn.get("pages"):
        raise ValueError("App definition is missing pages — regenerate the app first.")

    defn = apply_source_data_pipeline(defn, seed_if_empty=False)
    if not get_source_datasets(defn):
        raise ValueError("No source data model on this app — regenerate from the latest builder.")

    bag = _data_bag(defn)
    name = (filename or "upload").lower()
    stem = _slug(name.rsplit(".", 1)[0])
    summary: Dict[str, Any] = {"filename": filename, "updatedKeys": [], "computed": True}

    def _finish() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        defn2 = dict(defn)
        defn2["data"] = bag
        defn2 = recompute_derived_data(defn2)
        summary["derivedKeys"] = [
            k for k in (defn2.get("data") or {}) if not str(k).startswith("source_")
        ]
        return defn2, summary

    if name.endswith(".json"):
        try:
            payload = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid JSON: {e}") from e
        if not isinstance(payload, dict):
            raise ValueError("JSON must be an object keyed by source dataset id.")
        for key, val in payload.items():
            if not str(key).startswith("source_") and not _match_source_key(defn, str(key)):
                continue
            ds = _source_dataset_by_key(defn, str(key))
            if not ds or not isinstance(val, list):
                continue
            bag[str(ds["dataKey"])] = _rows_for_source(defn, ds, val)
            summary["updatedKeys"].append(str(ds["dataKey"]))
        if not summary["updatedKeys"]:
            raise ValueError("Upload JSON must contain source dataset arrays (e.g. source_maintenance_events).")
        return _finish()

    if name.endswith((".xlsx", ".xls")):
        import pandas as pd

        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        if not sheets:
            raise ValueError("Excel file has no sheets.")
        for sheet_name, frame in sheets.items():
            if frame is None or frame.empty:
                continue
            records = frame.fillna("").to_dict(orient="records")
            raw_rows = [{str(k): _coerce_cell(str(v)) for k, v in r.items()} for r in records]
            hint = target_key or str(sheet_name)
            ds = _source_dataset_by_key(defn, hint)
            if not ds:
                dk = _match_source_key(defn, hint)
                ds = _source_dataset_by_key(defn, dk or hint)
            if not ds:
                raise ValueError(
                    f"Sheet '{sheet_name}' does not match a source dataset. "
                    f"Use the downloaded template sheet names."
                )
            bag[str(ds["dataKey"])] = _rows_for_source(defn, ds, raw_rows)
            summary["updatedKeys"].append(str(ds["dataKey"]))
        if not summary["updatedKeys"]:
            raise ValueError("Excel upload produced no rows.")
        return _finish()

    if name.endswith(".csv") or "," in raw.decode("utf-8-sig", errors="replace")[:500]:
        text = raw.decode("utf-8-sig", errors="replace")
        _, rows = _parse_csv_text(text)
        if not rows:
            raise ValueError("CSV has no data rows.")
        hint = target_key or stem
        ds = _source_dataset_by_key(defn, hint)
        if not ds:
            dk = _match_source_key(defn, hint)
            ds = _source_dataset_by_key(defn, dk or hint)
        if not ds:
            raise ValueError(
                "CSV must target a source dataset — use the downloaded template or name the file like maintenance_events.csv."
            )
        bag[str(ds["dataKey"])] = _rows_for_source(defn, ds, rows)
        summary["updatedKeys"] = [str(ds["dataKey"])]
        summary["rowCount"] = len(bag[str(ds["dataKey"])])
        return _finish()

    raise ValueError("Unsupported file type. Use .csv, .json, or .xlsx from the template.")


def load_data_json_from_project(project_root: str) -> Optional[Dict[str, Any]]:
    import os

    path = os.path.join(project_root, _DATA_JSON_REL)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else None


def save_data_json_to_project(project_id: str, defn: Dict[str, Any]) -> None:
    from app_builder.services.file_writer import write_project_file

    payload = json.dumps(defn, indent=2, ensure_ascii=False)
    write_project_file(project_id, _DATA_JSON_REL, payload)
