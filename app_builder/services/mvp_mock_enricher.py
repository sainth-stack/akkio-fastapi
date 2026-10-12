"""Realistic MVP-style mock data for AppDefinition blocks (charts, KPIs, tables)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app_builder.schemas.app_definition import AppDefinition


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_") or "item"


def _kpi_demo_value(label: str, key: str, domain: str, index: int) -> str | int:
    low = f"{label} {key}".lower()
    if "oee" in low:
        return f"{84 + index * 1.2:.1f}%"
    if "throughput" in low or "output" in low or "volume" in low:
        return f"{1100 + index * 80:,} u/h"
    if "downtime" in low:
        return f"{35 + index * 4} min"
    if "alert" in low or "open" in low:
        return 3 + index
    if "scrap" in low or "defect" in low:
        return f"{0.6 + index * 0.1:.1f}%"
    if "energy" in low:
        return f"{11.5 + index * 0.3:.1f} kWh"
    if "mtbf" in low:
        return f"{120 + index * 8} h"
    if "mttr" in low:
        return f"{14 + index} min"
    if "yield" in low or "pass" in low:
        return f"{95.5 + index * 0.3:.1f}%"
    if "conversion" in low:
        return f"{3.2 + index * 0.2:.1f}%"
    if "variance" in low:
        return f"±{1.1 + index * 0.2}%"
    if "report" in low and "count" in low:
        return 18 + index * 2
    if "rate" in low or "delivery" in low or "on-time" in low or "otd" in low:
        return f"{93 + index * 0.5:.1f}%"
    if "cost" in low or "value" in low or "revenue" in low:
        return f"${17 + index:.2f}"
    if "inventory" in low or "sku" in low:
        return f"{900 + index * 50:,}"
    if domain == "manufacturing":
        defaults = ["87.4%", "1,240 u/h", "42 min", "5", "0.8%", "12.4 kWh"]
        return defaults[index % len(defaults)]
    if "count" in low or key.endswith("_d"):
        return 8 + index
    return ["1,248", "$4.2M", "94.2%", "17", "6.4×", "11.2d"][index % 6]


def _line_series_rows(series: List[Dict[str, Any]], x_key: str) -> List[Dict[str, Any]]:
    keys = [s.get("dataKey") for s in series if isinstance(s, dict) and s.get("dataKey")]
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    rows: List[Dict[str, Any]] = []
    for i, lab in enumerate(labels):
        row: Dict[str, Any] = {x_key: lab}
        for j, dk in enumerate(keys):
            base = 72 + i * 2 + j * 3
            if "target" in str(dk).lower() or "benchmark" in str(dk).lower():
                row[str(dk)] = 85
            else:
                row[str(dk)] = base
        rows.append(row)
    return rows


def _bar_series_rows(series: List[Dict[str, Any]], x_key: str) -> List[Dict[str, Any]]:
    weeks = [f"W{i + 1}" for i in range(6)]
    keys = [s.get("dataKey") for s in series if isinstance(s, dict) and s.get("dataKey")]
    rows: List[Dict[str, Any]] = []
    for i, w in enumerate(weeks):
        row: Dict[str, Any] = {x_key: w}
        for j, dk in enumerate(keys):
            dk_s = str(dk)
            if "defect" in dk_s.lower():
                row[dk_s] = max(12, 55 - i * 3)
            elif "benchmark" in dk_s.lower():
                row[dk_s] = 50 + i * 2
            else:
                row[dk_s] = 7800 + i * 180 + j * 40
        rows.append(row)
    return rows


def _table_rows(columns: List[Dict[str, Any]], table_key: str, domain: str) -> List[Dict[str, Any]]:
    fields = [c.get("field") for c in columns if isinstance(c, dict) and c.get("field")]
    if not fields:
        fields = ["id", "name", "status"]

    presets: Dict[str, List[Dict[str, Any]]] = {
        "table_analytics": [
            {"id": "M-01", "name": "CNC Cell A", "value": 91, "status": "Running"},
            {"id": "M-02", "name": "Press Line 2", "value": 78, "status": "Idle"},
            {"id": "M-03", "name": "Assembly Robot 4", "value": 89, "status": "Running"},
            {"id": "M-04", "name": "Paint Booth", "value": 0, "status": "Maintenance"},
        ],
        "table_reports": [
            {"id": "R-2406", "name": "Weekly OEE summary", "period": "Jun W2", "owner": "Operations", "status": "Published"},
            {"id": "R-2405", "name": "Scrap pareto analysis", "period": "Jun W1", "owner": "Quality", "status": "Published"},
            {"id": "R-2404", "name": "Energy intensity", "period": "May", "owner": "Sustainability", "status": "Draft"},
        ],
    }
    if table_key in presets:
        return presets[table_key]

    rows: List[Dict[str, Any]] = []
    for i in range(5):
        row: Dict[str, Any] = {}
        for f in fields:
            fs = str(f)
            if fs == "id":
                row[fs] = f"ROW-{1000 + i}"
            elif fs in ("name", "material", "report", "subject", "line", "machine"):
                row[fs] = f"Line {i + 1}" if "line" in fs else f"Asset {i + 1}"
            elif fs in ("status", "severity"):
                row[fs] = ["Running", "Idle", "Warning", "Critical", "Complete"][i % 5]
            elif fs in ("value", "lines", "alerts", "onHand", "turns"):
                row[fs] = (i + 1) * 420
            elif fs in ("period", "when", "updated"):
                row[fs] = f"{i + 1}d ago" if fs == "when" else f"2024-Q{(i % 4) + 1}"
            elif fs == "owner":
                row[fs] = ["Ops", "Quality", "Finance", "Plant Manager", "Engineering"][i % 5]
            elif fs == "oee":
                row[fs] = f"{88 - i:.1f}%"
            else:
                row[fs] = f"{fs.title()} {i + 1}"
        rows.append(row)
    return rows


def _pick_x_key(block: Dict[str, Any], sample: Optional[Dict[str, Any]]) -> str:
    preferred = str(block.get("xAxisKey") or "name")
    if sample:
        if preferred in sample:
            return preferred
        for k in ("label", "name", "week", "shift", "hour", "date", "period"):
            if k in sample:
                return k
    return preferred if preferred else "name"


def enrich_mvp_mock(defn: AppDefinition) -> AppDefinition:
    """Replace placeholders with product-quality demo data aligned to blocks."""
    mock: Dict[str, Any] = dict(defn.mock or {})
    domain = (defn.meta.domain or "generic").lower()

    for page in defn.pages:
        page_id = page.id
        bundle_key = f"kpis_{page_id}"
        bundle: Dict[str, Any] = {}
        if isinstance(mock.get(bundle_key), dict):
            bundle = dict(mock[bundle_key])

        for block in page.blocks:
            btype = block.get("type")
            if btype == "kpiRow":
                items = block.get("items") or []
                for i, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    vk = str(item.get("valueKey") or f"k{i}")
                    label = str(item.get("label") or vk)
                    cur = bundle.get(vk) if vk in bundle else mock.get(vk)
                    if cur in (None, "", "—", "1,240", "Sample"):
                        val = _kpi_demo_value(label, vk, domain, i)
                        bundle[vk] = val
                        mock[vk] = val
                if bundle:
                    mock[bundle_key] = bundle

            elif btype in ("lineChart", "barChart"):
                dk = block.get("dataKey")
                if not dk:
                    continue
                dk_s = str(dk)
                series = block.get("series") or []
                series_keys = [str(s.get("dataKey")) for s in series if isinstance(s, dict) and s.get("dataKey")]
                existing = mock.get(dk_s)
                needs = (
                    not existing
                    or not isinstance(existing, list)
                    or len(existing) == 0
                    or _looks_placeholder_series(existing)
                    or _series_keys_mismatch(existing, series_keys)
                )
                if needs:
                    x_key = str(block.get("xAxisKey") or "name")
                    if btype == "lineChart":
                        rows = _line_series_rows(series, x_key if x_key != "name" else "label")
                        if x_key == "name" and rows and "label" in rows[0]:
                            block["xAxisKey"] = "label"
                    else:
                        rows = _bar_series_rows(series, x_key if x_key != "name" else "label")
                        if x_key == "name" and rows and "label" in rows[0]:
                            block["xAxisKey"] = "label"
                    mock[dk_s] = rows
                else:
                    x_key = _pick_x_key(block, existing[0] if existing else None)
                    block["xAxisKey"] = x_key

            elif btype == "dataTable":
                dk = block.get("dataKey")
                if not dk:
                    continue
                dk_s = str(dk)
                cols = block.get("columns") or []
                existing = mock.get(dk_s)
                if not existing or not isinstance(existing, list) or _looks_placeholder_table(existing):
                    mock[dk_s] = _table_rows(cols, dk_s, domain)
                fields = [c.get("field") for c in cols if isinstance(c, dict) and c.get("field")]
                for idx, row in enumerate(mock.get(dk_s) or []):
                    if isinstance(row, dict) and "id" not in row:
                        seed = row.get("name") or row.get(fields[0] if fields else "row") or idx
                        row["id"] = f"{_slug(dk_s)}-{idx + 1}-{_slug(str(seed))}"

    defn.mock = mock
    return defn


def _series_keys_mismatch(rows: List[Any], series_keys: List[str]) -> bool:
    if not rows or not isinstance(rows[0], dict) or not series_keys:
        return False
    row = rows[0]
    return not all(k in row for k in series_keys)


def _looks_placeholder_series(rows: List[Any]) -> bool:
    if not rows or not isinstance(rows[0], dict):
        return True
    r0 = rows[0]
    vals = [v for v in r0.values() if isinstance(v, (int, float))]
    return len(vals) > 0 and all(v in (70, 73, 76) for v in vals[:3])


def _looks_placeholder_table(rows: List[Any]) -> bool:
    if not rows:
        return True
    text = str(rows[0])
    return "Sample name" in text or "Sample period" in text
