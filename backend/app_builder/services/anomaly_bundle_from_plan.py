"""Derive anomaly appConfig + mockData from PRD, UI/UX, and architecture JSON."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# page keys supported by the static anomaly React shell
PAGE_KEYS = ("dashboard", "anomalies", "alerts", "reports")

PAGE_DEFAULTS: Dict[str, Dict[str, str]] = {
    "dashboard": {"path": "/dashboard", "label": "Dashboard", "icon": "Dashboard"},
    "anomalies": {"path": "/anomalies", "label": "Anomaly Feed", "icon": "BugReport"},
    "alerts": {"path": "/alerts", "label": "Alert Config", "icon": "NotificationsActive"},
    "reports": {"path": "/reports", "label": "Reports", "icon": "Assessment"},
}

_SCREEN_ALIASES: Dict[str, str] = {
    "dashboard": "dashboard",
    "home": "dashboard",
    "overview": "dashboard",
    "anomalyfeed": "anomalies",
    "anomaly feed": "anomalies",
    "anomalies": "anomalies",
    "feed": "anomalies",
    "alertconfig": "alerts",
    "alert config": "alerts",
    "alerts": "alerts",
    "thresholds": "alerts",
    "configuration": "alerts",
    "reports": "reports",
    "analytics": "reports",
}


def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()


def screen_name_to_page_key(name: str) -> Optional[str]:
    k = _norm_key(name)
    if k in _SCREEN_ALIASES:
        return _SCREEN_ALIASES[k]
    for alias, page in _SCREEN_ALIASES.items():
        if alias in k or k in alias:
            return page
    return None


def _extract_screens(
    architecture: Optional[Dict[str, Any]],
    uiux: str,
    prd: str,
) -> List[Dict[str, str]]:
    screens: List[Dict[str, str]] = []

    if isinstance(architecture, dict):
        for s in architecture.get("screens") or []:
            if isinstance(s, dict) and s.get("name"):
                screens.append({
                    "name": str(s["name"]).strip(),
                    "description": str(s.get("description") or s.get("purpose") or "").strip(),
                })
        fs = architecture.get("frontend_structure")
        if isinstance(fs, dict):
            for p in fs.get("pages") or []:
                if isinstance(p, dict):
                    name = p.get("name") or p.get("component") or p.get("path")
                    if name:
                        screens.append({
                            "name": str(name).strip(),
                            "description": str(p.get("description") or "").strip(),
                        })
                elif isinstance(p, str):
                    screens.append({"name": p.strip(), "description": ""})

    for blob in (uiux, prd):
        if not blob:
            continue
        parsed = _try_parse_json_screens(blob)
        screens.extend(parsed)
        for m in re.finditer(
            r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:\d+\.\s*)?"
            r"(?:Screen|Page|Module)\s*[:\-]\s*([^\n]+)",
            blob,
            re.I,
        ):
            screens.append({"name": m.group(1).strip(), "description": ""})
        for m in re.finditer(
            r"(?:^|\n)\s*[-*]\s*\*\*([^*]+)\*\*\s*(?:[-–:]\s*([^\n]+))?",
            blob,
        ):
            title = m.group(1).strip()
            if screen_name_to_page_key(title):
                screens.append({
                    "name": title,
                    "description": (m.group(2) or "").strip(),
                })

    # de-dupe by normalized name
    seen: set = set()
    out: List[Dict[str, str]] = []
    for s in screens:
        nk = _norm_key(s["name"])
        if not nk or nk in seen:
            continue
        seen.add(nk)
        out.append(s)
    return out


def _try_parse_json_screens(text: str) -> List[Dict[str, str]]:
    text = text.strip()
    if not text.startswith("{"):
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    raw = data.get("screens") or data.get("pages")
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict) and item.get("name"):
            out.append({
                "name": str(item["name"]),
                "description": str(item.get("description") or ""),
            })
    return out


def _pick_metric_profile(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("cnc", "lathe", "spindle", "manufacturing", "machine tool", "factory floor")):
        return "manufacturing"
    if any(k in t for k in ("iot", "sensor", "telemetry", "edge device", "gateway")):
        return "iot"
    if any(k in t for k in ("pharma", "batch", "cold chain", "temperature excursion")):
        return "pharma"
    return "infrastructure"


def _metric_catalog(profile: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    if profile == "manufacturing":
        metrics = [
            {"metric_id": "spindle_vibration", "metric_name": "Spindle Vibration", "unit": "mm/s", "threshold_min": 0, "threshold_max": 4.5, "zscore_threshold": 3.0, "enabled": True},
            {"metric_id": "chuck_temperature", "metric_name": "Chuck Temperature", "unit": "°C", "threshold_min": 15, "threshold_max": 65, "zscore_threshold": 2.8, "enabled": True},
            {"metric_id": "feed_rate", "metric_name": "Feed Rate", "unit": "mm/min", "threshold_min": 50, "threshold_max": 420, "zscore_threshold": 3.0, "enabled": True},
            {"metric_id": "tool_wear", "metric_name": "Tool Wear Index", "unit": "%", "threshold_min": 0, "threshold_max": 78, "zscore_threshold": 2.5, "enabled": True},
            {"metric_id": "coolant_flow", "metric_name": "Coolant Flow", "unit": "L/min", "threshold_min": 2, "threshold_max": 18, "zscore_threshold": 3.0, "enabled": True},
            {"metric_id": "axis_load", "metric_name": "Axis Load", "unit": "%", "threshold_min": 0, "threshold_max": 92, "zscore_threshold": 3.0, "enabled": False},
        ]
        ts = {
            "spindle_vibration": [1.8, 0.4, 6.2],
            "chuck_temperature": [42, 3, 72],
            "feed_rate": [220, 25, 510],
            "tool_wear": [38, 6, 88],
            "coolant_flow": [9, 1.5, 22],
            "axis_load": [55, 8, 98],
        }
        return metrics, ts
    if profile == "iot":
        metrics = [
            {"metric_id": "ambient_temp", "metric_name": "Ambient Temperature", "unit": "°C", "threshold_min": 10, "threshold_max": 35, "zscore_threshold": 2.8, "enabled": True},
            {"metric_id": "humidity", "metric_name": "Humidity", "unit": "%", "threshold_min": 20, "threshold_max": 70, "zscore_threshold": 3.0, "enabled": True},
            {"metric_id": "pressure", "metric_name": "Pressure", "unit": "kPa", "threshold_min": 95, "threshold_max": 110, "zscore_threshold": 3.0, "enabled": True},
            {"metric_id": "battery_level", "metric_name": "Battery Level", "unit": "%", "threshold_min": 15, "threshold_max": 100, "zscore_threshold": 2.5, "enabled": True},
            {"metric_id": "signal_rssi", "metric_name": "Signal RSSI", "unit": "dBm", "threshold_min": -95, "threshold_max": -55, "zscore_threshold": 3.0, "enabled": True},
        ]
        ts = {
            "ambient_temp": [24, 2, 41],
            "humidity": [48, 6, 82],
            "pressure": [101, 1.2, 118],
            "battery_level": [76, 5, 12],
            "signal_rssi": [-68, 4, -42],
        }
        return metrics, ts
    if profile == "pharma":
        metrics = [
            {"metric_id": "chamber_temp", "metric_name": "Chamber Temperature", "unit": "°C", "threshold_min": 2, "threshold_max": 8, "zscore_threshold": 2.5, "enabled": True},
            {"metric_id": "humidity", "metric_name": "Humidity", "unit": "%", "threshold_min": 35, "threshold_max": 65, "zscore_threshold": 3.0, "enabled": True},
            {"metric_id": "particle_count", "metric_name": "Particle Count", "unit": "pts/m³", "threshold_min": 0, "threshold_max": 3500, "zscore_threshold": 3.0, "enabled": True},
            {"metric_id": "door_open_duration", "metric_name": "Door Open Duration", "unit": "sec", "threshold_min": 0, "threshold_max": 45, "zscore_threshold": 2.8, "enabled": True},
        ]
        ts = {
            "chamber_temp": [5.5, 0.6, 11.2],
            "humidity": [52, 4, 78],
            "particle_count": [1200, 200, 5200],
            "door_open_duration": [12, 3, 95],
        }
        return metrics, ts
    # infrastructure default
    metrics = [
        {"metric_id": "cpu_usage", "metric_name": "CPU Usage", "unit": "%", "threshold_min": 0, "threshold_max": 85, "zscore_threshold": 3.0, "enabled": True},
        {"metric_id": "memory_usage", "metric_name": "Memory Usage", "unit": "%", "threshold_min": 0, "threshold_max": 90, "zscore_threshold": 3.0, "enabled": True},
        {"metric_id": "request_latency", "metric_name": "Request Latency", "unit": "ms", "threshold_min": 0, "threshold_max": 2000, "zscore_threshold": 2.5, "enabled": True},
        {"metric_id": "error_rate", "metric_name": "Error Rate", "unit": "%", "threshold_min": 0, "threshold_max": 5, "zscore_threshold": 2.5, "enabled": True},
        {"metric_id": "disk_io", "metric_name": "Disk I/O", "unit": "MB/s", "threshold_min": 0, "threshold_max": 500, "zscore_threshold": 3.0, "enabled": True},
        {"metric_id": "network_in", "metric_name": "Network In", "unit": "Mbps", "threshold_min": 0, "threshold_max": 1000, "zscore_threshold": 3.0, "enabled": False},
    ]
    ts = {
        "cpu_usage": [45, 10, 93],
        "memory_usage": [65, 8, 94],
        "request_latency": [250, 50, 3400],
        "error_rate": [0.5, 0.3, 15],
        "disk_io": [120, 30, 640],
        "network_in": [350, 80, 1350],
    }
    return metrics, ts


def _sample_anomalies_for_metrics(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not metrics:
        return []
    sevs = ["Critical", "High", "Medium", "Low"]
    statuses = ["Open", "Acknowledged", "Resolved"]
    rows: List[Dict[str, Any]] = []
    for i, m in enumerate(metrics[:5]):
        mid = m["metric_id"]
        name = m["metric_name"]
        hi = float(m["threshold_max"])
        val = round(hi * (1.12 + i * 0.03), 2)
        rows.append({
            "id": f"a{i + 1}",
            "timestamp": f"2026-10-0{5 - (i % 3)}T{10 + i}:22:00Z",
            "metric_name": name,
            "metric_id": mid,
            "value": val,
            "expected_min": float(m.get("threshold_min", 0)),
            "expected_max": hi,
            "severity": sevs[i % len(sevs)],
            "status": statuses[i % len(statuses)],
            "anomaly_score": round(3.2 + i * 0.45, 2),
        })
    return rows


def _sample_alerts_from_anomalies(anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    alerts = []
    for i, a in enumerate(anomalies[:4]):
        alerts.append({
            "id": f"al{i + 1}",
            "metric_name": a["metric_name"],
            "threshold": a["expected_max"],
            "current_value": a["value"],
            "severity": a["severity"],
            "triggered_at": a["timestamp"],
            "status": "Active" if a["status"] == "Open" else "Acknowledged",
        })
    return alerts


def _top_metrics_from_anomalies(anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    agg: Dict[str, Dict[str, Any]] = {}
    order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    for a in anomalies:
        mn = a["metric_name"]
        if mn not in agg:
            agg[mn] = {"metric_name": mn, "count": 0, "severity": "Low"}
        agg[mn]["count"] += 1
        if order.get(a["severity"], 0) > order.get(agg[mn]["severity"], 0):
            agg[mn]["severity"] = a["severity"]
    return sorted(agg.values(), key=lambda x: x["count"], reverse=True)[:5]


def _first_paragraph(text: str, max_len: int = 160) -> str:
    for block in re.split(r"\n\s*\n", text or ""):
        line = re.sub(r"^#+\s*", "", block.strip())
        line = re.sub(r"\*+", "", line).strip()
        if len(line) > 20:
            return line[:max_len]
    return ""


def apply_plan_to_bundle(
    bundle: Dict[str, Any],
    *,
    requirement: str = "",
    prd: str = "",
    uiux: str = "",
    architecture: Optional[Dict[str, Any]] = None,
    title: str = "",
) -> Dict[str, Any]:
    """Merge PRD / UI/UX / architecture into an existing default bundle."""
    combined = "\n".join([requirement or "", prd or "", uiux or ""])
    profile = _pick_metric_profile(combined)
    metrics, ts_params = _metric_catalog(profile)
    bundle["metricsConfig"] = metrics
    bundle["timeSeriesParams"] = ts_params
    anomalies = _sample_anomalies_for_metrics(metrics)
    bundle["anomalies"] = anomalies
    bundle["alerts"] = _sample_alerts_from_anomalies(anomalies)
    bundle["reportsSummary"]["top_metrics"] = _top_metrics_from_anomalies(anomalies)

    screens = _extract_screens(architecture, uiux, prd)
    label_by_page: Dict[str, str] = {}
    desc_by_page: Dict[str, str] = {}
    for s in screens:
        pk = screen_name_to_page_key(s["name"])
        if not pk:
            continue
        label_by_page[pk] = s["name"]
        if s.get("description"):
            desc_by_page[pk] = s["description"]

    nav: List[Dict[str, str]] = []
    for pk in PAGE_KEYS:
        defaults = PAGE_DEFAULTS[pk]
        nav.append({
            "path": defaults["path"],
            "label": label_by_page.get(pk, defaults["label"]),
            "icon": defaults["icon"],
            "page": pk,
        })
    bundle["navigation"] = nav

    pages = bundle.setdefault("pages", {})
    if title:
        pages.setdefault("dashboard", {})["title"] = f"{title.strip()} — Dashboard"
    for pk, desc in desc_by_page.items():
        pages.setdefault(pk, {})["subtitle"] = desc

    overview = _first_paragraph(uiux) or _first_paragraph(prd) or _first_paragraph(requirement)
    if overview:
        bundle.setdefault("branding", {})["loginSubtitle"] = overview[:200]
        bundle["branding"]["headerTitle"] = overview[:80] if len(overview) <= 80 else f"{title or 'Monitoring'} hub"

    # KPI labels from UI/UX when mentioned
    if re.search(r"mean time to detect|mttd", combined, re.I):
        pages.setdefault("reports", {}).setdefault("summaryCards", [])
        for card in pages["reports"].get("summaryCards", []):
            if card.get("key") == "mttd_minutes":
                card["label"] = "Mean time to detect"

    bundle["meta"] = {
        "metricProfile": profile,
        "screenCount": len(screens),
        "sources": {
            "requirement": bool(requirement),
            "prd": bool(prd),
            "uiux": bool(uiux),
            "architecture": bool(architecture),
        },
    }
    return bundle


def merge_theme_colors(
    colors: Dict[str, str],
    design_tokens: Optional[Dict[str, Any]] = None,
    uiux: str = "",
) -> Dict[str, str]:
    from app_builder.services.fullstack_frontend_generator import theme_from_tokens

    merged = theme_from_tokens(design_tokens, uiux=uiux)
    out = {**merged, **{k: v for k, v in (colors or {}).items() if v}}
    if out.get("danger"):
        out["anomaly_marker"] = out["danger"]
    return out
