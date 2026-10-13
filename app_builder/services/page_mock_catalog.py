"""Page-aware synthetic datasets — names and values match each screen's purpose."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

MIN_TABLE_ROWS = 18
CHART_POINTS = 12

_OPERATORS = ("J. Martinez", "A. Chen", "R. Patel", "S. Okafor", "L. Nguyen", "M. Dubois", "K. Weber", "T. Sato")
_OWNERS = ("Operations", "Quality", "Maintenance", "Engineering", "Sustainability", "Plant Manager")
_UNITS = (
    "Transformer T-12", "Transformer T-08", "HV Bay 3", "Substation Alpha", "Unit B-14",
    "Line Reactor 2", "Tap Changer Bank 1", "Cooling Skid 4", "Bushing Set C", "Core Press 7",
    "GIS Module 5", "Feeder F-22", "Capacitor Bank 2", "Relay Panel R-9", "Oil Pump Skid 1",
    "Insulation Test Cell", "Winding Station 3", "Final Test Bay", "Shipping Dock Monitor",
)
_METRICS = (
    "Load (MVA)", "Oil temperature", "Winding hot-spot", "Tap position", "Bushing PD",
    "Cooling fan speed", "Dissolved gas (ppm)", "Efficiency", "Harmonic distortion", "Vibration RMS",
)
_REPORTS = (
    ("Weekly load profile", "Jun W2", "Operations", "Published"),
    ("Oil DGA trend analysis", "Jun W1", "Maintenance", "Published"),
    ("Bushing partial discharge", "May", "Engineering", "Published"),
    ("Cooling system performance", "May", "Sustainability", "Draft"),
    ("Tap changer operations log", "Apr", "Maintenance", "Published"),
    ("Efficiency vs ambient temp", "Q2", "Operations", "Published"),
    ("Alarm summary (30d)", "Jun", "Quality", "Published"),
    ("Session compliance audit", "Jun W2", "Quality", "Draft"),
)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_") or "item"


def infer_page_profile(
    page_id: str,
    page_title: str,
    subtitle: str = "",
    meta_description: str = "",
    domain: str = "",
) -> str:
    pid = (page_id or "").lower()
    pt = (page_title or "").lower().strip()
    meta_l = (meta_description or "").lower()
    domain_l = (domain or "").lower()

    if pid in ("sessions", "session") or pt in ("sessions", "session"):
        return "sessions"
    if pid == "reports" or pt == "reports":
        return "reports"
    if pid == "analytics" or pt == "analytics":
        return "analytics"
    if pid == "dashboard" or pt == "dashboard":
        if any(k in meta_l + domain_l for k in ("transformer", "substation", "gis ", "tap changer")):
            return "transformer_monitoring"
        if "manufacturing" in domain_l or "manufacturing" in meta_l or "plant" in meta_l:
            return "manufacturing"
        return "dashboard"

    ctx = f"{page_id} {page_title} {subtitle}".lower()
    if "session" in ctx and len(pt) < 24:
        return "sessions"
    if "report" in ctx and len(pt) < 24:
        return "reports"
    if ("analytic" in ctx or "insight" in ctx) and len(pt) < 24:
        return "analytics"
    if "dashboard" in ctx or pid == "dashboard":
        return "dashboard"
    if any(k in ctx for k in ("transformer", "substation", "hv ", "monitoring", "product monitor")):
        return "transformer_monitoring"
    if any(k in ctx for k in ("inventory", "material", "sku", "warehouse", "supplier", "purchase")):
        return "inventory"
    if "manufacturing" in ctx or "plant" in ctx or "oee" in ctx:
        return "manufacturing"
    return "operations"


def default_kpi_bundle(profile: str, page_id: str) -> Dict[str, str]:
    if profile == "transformer_monitoring" or profile == "dashboard":
        if "transformer" in page_id or profile == "transformer_monitoring":
            return {
                "load_mva": "42.8 MVA",
                "oil_temp": "62°C",
                "efficiency": "98.2%",
                "active_alarms": "3",
                "tap_position": "14",
                "pd_level": "12 pC",
            }
        return {
            "oee": "87.4%",
            "throughput": "1,240 u/h",
            "downtime": "42 min",
            "active_alerts": "5",
            "scrap_rate": "0.8%",
            "energy_per_unit": "12.4 kWh",
        }
    if profile == "manufacturing":
        return {
            "oee": "87.4%",
            "throughput": "1,240 u/h",
            "downtime": "42 min",
            "active_alerts": "5",
            "scrap_rate": "0.8%",
            "energy_per_unit": "12.4 kWh",
        }
    if profile == "sessions":
        return {
            "active_sessions": "12",
            "avg_duration": "2h 18m",
            "units_online": "9",
            "exceptions": "2",
        }
    if profile == "analytics":
        return {
            "mtbf": "128 h",
            "first_pass_yield": "96.2%",
            "scrap_rate": "0.7%",
            "variance": "±1.1%",
        }
    if profile == "reports":
        return {
            "reports_published": "24",
            "on_time_delivery": "94.8%",
            "open_drafts": "3",
        }
    if profile == "inventory":
        return {
            "total_skus": "1,248",
            "inventory_value": "$4.2M",
            "low_stock_alerts": "17",
            "supplier_otd": "94.2%",
        }
    return {
        "primary_kpi": "1,240",
        "secondary_kpi": "94.2%",
        "risk_indicator": "5",
        "open_items": "12",
    }


_GENERIC_KPI_KEYS = frozenset({"primary_kpi", "secondary_kpi", "risk_indicator", "open_items"})


def enrich_kpi_object(
    existing: Optional[Dict[str, Any]],
    profile: str,
    page_id: str,
) -> Dict[str, str]:
    base = default_kpi_bundle(profile, page_id)
    out: Dict[str, str] = {}
    if isinstance(existing, dict):
        for k, v in existing.items():
            if v not in (None, "", "—"):
                out[str(k)] = str(v) if not isinstance(v, (int, float)) else (
                    f"{v:,}" if isinstance(v, int) and v > 999 else str(v)
                )
    if not out or set(out.keys()).issubset(_GENERIC_KPI_KEYS):
        out = {}
    for k, v in base.items():
        if k not in out or out[k] in ("—", "1,240", "Sample", "94.2%"):
            out[k] = v
    if set(out.keys()).issubset(_GENERIC_KPI_KEYS):
        return dict(base)
    return out


def _column_value(
    field: str,
    row_index: int,
    profile: str,
    table_key: str,
    page_title: str,
) -> Any:
    f = field.lower()
    i = row_index

    if f == "id":
        if profile == "sessions":
            return f"SES-{2400 + i}"
        if profile == "reports":
            return f"RPT-{2400 + i}"
        if "session" in table_key:
            return f"SES-{2400 + i}"
        return f"ROW-{1000 + i}"

    if f in ("name", "unit", "unit_id", "machine", "asset"):
        return _UNITS[i % len(_UNITS)]

    if f in ("subject", "report", "title"):
        return _REPORTS[i % len(_REPORTS)][0]

    if f in ("metric", "parameter"):
        return _METRICS[i % len(_METRICS)]

    if f == "operator":
        return _OPERATORS[i % len(_OPERATORS)]

    if f in ("line", "zone", "area"):
        return f"Zone {(i % 6) + 1}"

    if f == "session" or "session_id" in f:
        return f"SES-{2400 + i}"

    if f in ("status", "state"):
        return ("Running", "Idle", "Warning", "Maintenance", "Complete", "Critical")[i % 6]

    if f == "severity":
        return ("Low", "Medium", "High", "Critical")[i % 4]

    if f in ("when", "updated", "last_seen", "updated_at", "last_updated"):
        day = (i % 28) + 1
        hour = 8 + (i % 10)
        minute = (i * 7) % 60
        if "when" in f or "seen" in f:
            return f"2024-06-{day:02d}T{hour:02d}:{minute:02d}:00Z"
        return f"2024-06-{day:02d}T{hour:02d}:{minute:02d}:00Z"

    if f in ("started", "start_time", "started_at"):
        return f"2024-06-{(i % 28) + 1:02d}T08:{(i * 3) % 60:02d}:00Z"

    if f in ("duration", "runtime"):
        return f"{1 + (i % 4)}h {(10 + i * 7) % 60}m"

    if f in ("period",):
        return _REPORTS[i % len(_REPORTS)][1]

    if f == "owner":
        return _REPORTS[i % len(_REPORTS)][2]

    if f in ("value", "avg", "avg_value", "reading"):
        if profile == "transformer_monitoring":
            return round(38.0 + i * 0.8, 1)
        return 75 + (i % 20)

    if f in ("min", "minimum"):
        return round(30 + i * 0.3, 1)

    if f in ("max", "maximum"):
        return round(55 + i * 0.5, 1)

    if f == "oee":
        return f"{82 + (i % 15):.1f}%"

    if f in ("onhand", "on_hand", "quantity", "lines"):
        return (i + 1) * 120

    if f == "trend":
        return ("Up", "Down", "Flat")[i % 3]

    if f == "type":
        return ("Alarm", "Warning", "Info", "Maintenance")[i % 4]

    return f"{field.replace('_', ' ').title()} {i + 1}"


def generate_table_rows(
    columns: List[Dict[str, Any]],
    *,
    profile: str,
    table_key: str,
    page_title: str,
    row_count: int = MIN_TABLE_ROWS,
) -> List[Dict[str, Any]]:
    fields = [str(c.get("field")) for c in columns if isinstance(c, dict) and c.get("field")]
    if not fields:
        fields = ["id", "name", "status"]

    # Curated full tables for known keys
    if "session" in table_key.lower() or profile == "sessions":
        full_rows = [
            {
                "id": f"SES-{2400 + i}",
                "name": _UNITS[i % len(_UNITS)],
                "operator": _OPERATORS[i % len(_OPERATORS)],
                "duration": f"{1 + (i % 4)}h {(10 + i * 7) % 60}m",
                "status": ("Running", "Idle", "Warning", "Complete")[i % 4],
            }
            for i in range(row_count)
        ]
        return [
            {
                f: r.get(f)
                if r.get(f) is not None
                else _column_value(f, i, profile, table_key, page_title)
                for f in fields
            }
            for i, r in enumerate(full_rows)
        ]

    if profile == "reports" or "report" in table_key.lower():
        rows = []
        for i in range(min(row_count, len(_REPORTS) + 10)):
            name, period, owner, status = _REPORTS[i % len(_REPORTS)]
            rows.append({
                "id": f"RPT-{2400 + i}",
                "name": name,
                "period": period,
                "owner": owner,
                "status": status if i < len(_REPORTS) else "Draft",
            })
        while len(rows) < row_count:
            i = len(rows)
            rows.append({
                "id": f"RPT-{2400 + i}",
                "name": f"Operational summary #{i + 1}",
                "period": f"Jun W{(i % 4) + 1}",
                "owner": _OWNERS[i % len(_OWNERS)],
                "status": "Draft",
            })
        return [{f: r.get(f) for f in fields} for r in rows]

    if profile == "analytics" or "analytics" in table_key.lower():
        full_rows = [
            {
                "id": f"MET-{100 + i}",
                "name": _METRICS[i % len(_METRICS)],
                "value": round(40 + i * 1.2, 1),
                "status": ("Normal", "Watch", "Normal", "Elevated")[i % 4],
            }
            for i in range(row_count)
        ]
        return [
            {
                f: r.get(f)
                if r.get(f) is not None
                else _column_value(f, i, profile, table_key, page_title)
                for f in fields
            }
            for i, r in enumerate(full_rows)
        ]

    rows: List[Dict[str, Any]] = []
    for i in range(row_count):
        row: Dict[str, Any] = {}
        for field in fields:
            row[field] = _column_value(field, i, profile, table_key, page_title)
        rows.append(row)
    return rows


def generate_chart_rows(
    data_key: str,
    series_keys: List[str],
    profile: str,
    chart_type: str,
) -> Tuple[List[Dict[str, Any]], str]:
    x_key = "period"
    labels = [f"2024-06-{(i % 28) + 1:02d}" for i in range(CHART_POINTS)]
    rows: List[Dict[str, Any]] = []

    is_transformer = profile == "transformer_monitoring" or any(
        k in data_key.lower() for k in ("performance", "load", "transformer", "product_metric")
    )

    for i, lab in enumerate(labels):
        row: Dict[str, Any] = {x_key: lab}
        for dk in series_keys:
            dk_l = dk.lower()
            if "target" in dk_l or "benchmark" in dk_l:
                row[dk] = 85 if not is_transformer else 42.0
            elif "defect" in dk_l or "alarm" in dk_l:
                row[dk] = max(3, 28 - i * 2)
            elif "load" in dk_l or "mva" in dk_l:
                row[dk] = round(38 + i * 0.6 + (i % 3) * 0.2, 1)
            elif "oil" in dk_l or "temp" in dk_l:
                row[dk] = round(58 + i * 0.4, 1)
            elif "efficien" in dk_l:
                row[dk] = round(96.5 + i * 0.15, 1)
            elif "oee" in dk_l:
                row[dk] = round(82 + i * 0.9, 1)
            elif "output" in dk_l or "actual" in dk_l or "value" in dk_l:
                row[dk] = 7200 + i * 220
            else:
                row[dk] = 70 + i * 2
        rows.append(row)
    return rows, x_key


def should_replace_table(rows: List[Any]) -> bool:
    if not rows or not isinstance(rows, list):
        return True
    if len(rows) < MIN_TABLE_ROWS:
        return True
    text = str(rows[:4])
    bad = ("Asset 1", "Asset 2", "Sample name", "ROW-1000", "Record 1", "Line 1")
    return any(b in text for b in bad)
