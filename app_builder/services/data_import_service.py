"""Merge user-uploaded CSV / JSON / Excel into json_app data.json datasets."""
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
    """Human-readable targets derived from page blocks."""
    targets: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for page in defn.get("pages") or []:
        if not isinstance(page, dict):
            continue
        page_id = str(page.get("id") or "")
        for block in page.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type") or "")
            dk = str(block.get("dataKey") or "")
            if not dk or dk in seen:
                continue
            seen.add(dk)
            label = str(block.get("title") or block.get("type") or dk)
            targets.append(
                {
                    "dataKey": dk,
                    "blockType": btype,
                    "pageId": page_id,
                    "label": label,
                }
            )
    return targets


def _match_data_key(defn: Dict[str, Any], hint: str) -> Optional[str]:
    hint_slug = _slug(hint)
    if not hint_slug:
        return None
    keys = {t["dataKey"] for t in list_import_targets(defn)}
    if hint in keys:
        return hint
    for k in keys:
        if _slug(k) == hint_slug or hint_slug in _slug(k) or _slug(k) in hint_slug:
            return k
    return None


def _default_table_key(defn: Dict[str, Any]) -> Optional[str]:
    for t in list_import_targets(defn):
        if t["blockType"] == "dataTable":
            return t["dataKey"]
    for t in list_import_targets(defn):
        if t["blockType"] == "kpiRow":
            return t["dataKey"]
    targets = list_import_targets(defn)
    return targets[0]["dataKey"] if targets else None


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
    headers = [_slug(h) or f"col_{i}" for i, h in enumerate(rows[0])]
    out: List[Dict[str, Any]] = []
    for row in rows[1:]:
        if not any(str(c).strip() for c in row):
            continue
        item: Dict[str, Any] = {}
        for i, h in enumerate(headers):
            if i >= len(row):
                break
            raw = row[i].strip()
            item[h] = _coerce_cell(raw)
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


def _looks_like_kpi_csv(headers: List[str], rows: List[Dict[str, Any]]) -> bool:
    if len(headers) < 2 or not rows:
        return False
    h = " ".join(headers)
    if re.search(r"label|metric|kpi|name", h) and re.search(r"value|amount|count|reading", h):
        return True
    if len(headers) == 2 and all(len(r) >= 2 for r in rows[:3]):
        return True
    return False


def _kpi_bundle_from_rows(headers: List[str], rows: List[Dict[str, Any]]) -> Dict[str, str]:
    label_idx = 0
    value_idx = 1
    for i, h in enumerate(headers):
        if re.search(r"value|amount|count|reading|metric_value", h):
            value_idx = i
        if re.search(r"label|metric|kpi|name", h):
            label_idx = i
    bundle: Dict[str, str] = {}
    for row in rows:
        keys = list(row.keys())
        if len(keys) < 2:
            continue
        label_key = keys[label_idx] if label_idx < len(keys) else keys[0]
        value_key = keys[value_idx] if value_idx < len(keys) else keys[1]
        label = str(row.get(label_key) or "").strip()
        val = row.get(value_key)
        if not label:
            continue
        vk = _slug(label)[:40] or f"metric_{len(bundle)}"
        bundle[vk] = str(val) if val is not None else ""
    return bundle


def _align_table_rows(defn: Dict[str, Any], data_key: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fields: List[str] = []
    for page in defn.get("pages") or []:
        for block in page.get("blocks") or []:
            if isinstance(block, dict) and block.get("dataKey") == data_key:
                for col in block.get("columns") or []:
                    if isinstance(col, dict) and col.get("field"):
                        fields.append(str(col["field"]))
    if not fields:
        return rows
    aligned: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        out: Dict[str, Any] = {}
        for f in fields:
            if f in row:
                out[f] = row[f]
            else:
                slug_f = _slug(f)
                match = next((row[k] for k in row if _slug(k) == slug_f), None)
                out[f] = match if match is not None else (i + 1 if f == "id" else "")
        aligned.append(out)
    return aligned


def merge_upload_bytes(
    defn: Dict[str, Any],
    *,
    filename: str,
    raw: bytes,
    target_key: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Return (updated_definition, summary).
    Raises ValueError on unsupported or empty input.
    """
    if not isinstance(defn, dict) or not defn.get("pages"):
        raise ValueError("App definition is missing pages — regenerate the app first.")

    bag = _data_bag(defn)
    name = (filename or "upload").lower()
    stem = _slug(name.rsplit(".", 1)[0])
    summary: Dict[str, Any] = {"filename": filename, "updatedKeys": []}

    if name.endswith(".json"):
        try:
            payload = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid JSON: {e}") from e
        if isinstance(payload, dict) and payload.get("pages") and payload.get("navigation"):
            payload = _data_bag(payload)
        if not isinstance(payload, dict):
            raise ValueError("JSON must be an object of data keys or a full app definition.")
        for key, val in payload.items():
            if key in ("meta", "theme", "pages", "navigation", "version"):
                continue
            bag[key] = val
            summary["updatedKeys"].append(key)
        if not summary["updatedKeys"]:
            raise ValueError("No dataset keys found in JSON upload.")
        return _set_data_bag(defn, bag), summary

    if name.endswith((".xlsx", ".xls")):
        import pandas as pd

        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        if not sheets:
            raise ValueError("Excel file has no sheets.")
        for sheet_name, frame in sheets.items():
            if frame is None or frame.empty:
                continue
            records = frame.fillna("").to_dict(orient="records")
            rows = [{_slug(str(k)): _coerce_cell(str(v)) for k, v in r.items()} for r in records]
            key = target_key or _match_data_key(defn, str(sheet_name)) or _match_data_key(defn, stem)
            if not key:
                key = _default_table_key(defn)
            if not key:
                raise ValueError("Could not map Excel sheet to a dashboard block dataKey.")
            block_type = next(
                (t["blockType"] for t in list_import_targets(defn) if t["dataKey"] == key),
                "dataTable",
            )
            if block_type == "kpiRow" or _looks_like_kpi_csv(list(rows[0].keys()) if rows else [], rows):
                bag[key] = _kpi_bundle_from_rows(list(rows[0].keys()) if rows else [], rows)
            elif isinstance(bag.get(key), list) or block_type == "dataTable":
                bag[key] = _align_table_rows(defn, key, rows)
            else:
                bag[key] = rows
            summary["updatedKeys"].append(key)
        if not summary["updatedKeys"]:
            raise ValueError("Excel upload produced no rows.")
        return _set_data_bag(defn, bag), summary

    if name.endswith(".csv") or "," in raw.decode("utf-8-sig", errors="replace")[:500]:
        text = raw.decode("utf-8-sig", errors="replace")
        headers, rows = _parse_csv_text(text)
        if not rows:
            raise ValueError("CSV has no data rows.")
        key = target_key or _match_data_key(defn, stem) or _default_table_key(defn)
        if not key:
            raise ValueError("Could not map CSV to a block dataKey — name the file like table_dashboard.csv.")
        block_type = next(
            (t["blockType"] for t in list_import_targets(defn) if t["dataKey"] == key),
            "dataTable",
        )
        if block_type == "kpiRow" or _looks_like_kpi_csv(headers, rows):
            bag[key] = _kpi_bundle_from_rows(headers, rows)
        elif block_type in ("lineChart", "barChart"):
            bag[key] = rows
        else:
            bag[key] = _align_table_rows(defn, key, rows)
        summary["updatedKeys"] = [key]
        summary["rowCount"] = len(rows) if isinstance(bag.get(key), list) else len(bag.get(key) or {})
        return _set_data_bag(defn, bag), summary

    raise ValueError("Unsupported file type. Use .csv, .json, or .xlsx.")


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
