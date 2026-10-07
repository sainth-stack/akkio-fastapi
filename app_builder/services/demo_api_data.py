"""
Realistic demo API payloads for preview when the generated FastAPI backend is not running.
Used by dynamic_app_router and seed_data.json fallbacks.
"""
from __future__ import annotations

import datetime
import random
from typing import Any, Dict, List


def _time_series(metric_id: str, base: float, spread: float, spike: float, points: int = 48) -> List[Dict[str, Any]]:
    rng = random.Random(sum(ord(c) for c in metric_id))
    anomaly_idx = set(rng.sample(range(8, points - 4), min(3, max(1, points // 16))))
    out: List[Dict[str, Any]] = []
    now = datetime.datetime.utcnow()
    for i in range(points):
        val = spike * (0.9 + rng.random() * 0.2) if i in anomaly_idx else max(0.0, base + rng.gauss(0, spread))
        ts = now - datetime.timedelta(minutes=(points - i) * 5)
        out.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "value": round(val, 2),
            "is_anomaly": i in anomaly_idx,
            "anomaly_score": round(3.2 + rng.random() * 2.5, 2) if i in anomaly_idx else round(rng.random() * 0.6, 2),
        })
    return out


def _rng_for_project(project_id: str = "") -> random.Random:
    seed = sum(ord(c) for c in (project_id or "akkio-default")) % (2**32)
    return random.Random(seed)


def anomaly_dashboard_payload(project_id: str = "") -> Dict[str, Any]:
    """Shape expected by anomaly DashboardPage (apiFetch('/api/dashboard'))."""
    rng = _rng_for_project(project_id)
    metrics_cfg = [
        ("spindle_speed", "Spindle Speed", "RPM", 4200.0, 1800.0, 120.0),
        ("feed_rate", "Feed Rate", "mm/min", 450.0, 120.0, 25.0),
        ("tool_wear", "Tool Wear Index", "%", 85.0, 35.0, 8.0),
        ("vibration", "Vibration", "mm/s", 12.0, 4.0, 1.2),
        ("coolant_temp", "Coolant Temperature", "°C", 65.0, 42.0, 3.5),
    ]
    metrics: List[Dict[str, Any]] = []
    for mid, name, unit, hi, base, spread in metrics_cfg:
        metrics.append({
            "metric_id": mid,
            "metric_name": name,
            "unit": unit,
            "threshold_low": 0.0,
            "threshold_high": hi,
            "threshold_min": 0.0,
            "threshold_max": hi,
            "zscore_threshold": 3.0,
            "enabled": True,
            "data": _time_series(mid, base, spread, hi * 1.08),
        })
    return {
        "total_anomalies_today": rng.randint(3, 18),
        "active_alerts": rng.randint(2, 9),
        "metrics_monitored": len(metrics),
        "avg_anomaly_score": round(rng.uniform(2.4, 5.2), 2),
        "metrics": metrics,
    }


def anomaly_anomalies_list() -> Dict[str, Any]:
    items = [
        {"id": "a1", "timestamp": "2026-10-07T14:22:00Z", "metric_name": "Spindle Speed", "metric_id": "spindle_speed",
         "value": 4680, "expected_min": 800, "expected_max": 4200, "severity": "Critical", "status": "Open", "anomaly_score": 5.1},
        {"id": "a2", "timestamp": "2026-10-07T11:05:00Z", "metric_name": "Tool Wear Index", "metric_id": "tool_wear",
         "value": 91.2, "expected_min": 0, "expected_max": 85, "severity": "High", "status": "Acknowledged", "anomaly_score": 4.0},
        {"id": "a3", "timestamp": "2026-10-07T08:44:00Z", "metric_name": "Vibration", "metric_id": "vibration",
         "value": 14.8, "expected_min": 0, "expected_max": 12, "severity": "High", "status": "Open", "anomaly_score": 3.7},
        {"id": "a4", "timestamp": "2026-10-06T22:18:00Z", "metric_name": "Feed Rate", "metric_id": "feed_rate",
         "value": 512, "expected_min": 50, "expected_max": 450, "severity": "Medium", "status": "Resolved", "anomaly_score": 3.1},
    ]
    return {"items": items, "total": len(items)}


def anomaly_thresholds_list() -> Dict[str, Any]:
    items = [
        {"metric_id": "spindle_speed", "metric_name": "Spindle Speed", "unit": "RPM", "threshold_min": 0, "threshold_max": 4200, "zscore_threshold": 3.0, "enabled": True},
        {"metric_id": "feed_rate", "metric_name": "Feed Rate", "unit": "mm/min", "threshold_min": 0, "threshold_max": 450, "zscore_threshold": 2.5, "enabled": True},
        {"metric_id": "tool_wear", "metric_name": "Tool Wear Index", "unit": "%", "threshold_min": 0, "threshold_max": 85, "zscore_threshold": 3.0, "enabled": True},
        {"metric_id": "vibration", "metric_name": "Vibration", "unit": "mm/s", "threshold_min": 0, "threshold_max": 12, "zscore_threshold": 2.5, "enabled": True},
        {"metric_id": "coolant_temp", "metric_name": "Coolant Temperature", "unit": "°C", "threshold_min": 35, "threshold_max": 65, "zscore_threshold": 3.0, "enabled": True},
    ]
    return {"items": items, "total": len(items)}


def anomaly_reports_summary() -> Dict[str, Any]:
    daily = [{"date": f"Oct {d}", "count": max(1, (d % 5) + 2)} for d in range(1, 31)]
    return {
        "total_anomalies": 156,
        "anomaly_rate": 0.0031,
        "mttd_minutes": 4.7,
        "daily_counts": daily,
        "by_severity": [
            {"name": "Critical", "value": 28},
            {"name": "High", "value": 54},
            {"name": "Medium", "value": 48},
            {"name": "Low", "value": 26},
        ],
        "top_metrics": [
            {"metric_name": "Spindle Speed", "count": 42, "severity": "Critical"},
            {"metric_name": "Tool Wear Index", "count": 38, "severity": "High"},
            {"metric_name": "Vibration", "count": 31, "severity": "High"},
        ],
    }


def anomaly_seed_data(project_id: str = "") -> Dict[str, Any]:
    dash = anomaly_dashboard_payload(project_id)
    return {
        "dashboard": dash,
        "kpis": dash,
        "dashboard/kpis": dash,
        "anomalies": anomaly_anomalies_list(),
        "thresholds": anomaly_thresholds_list(),
        "alerts": {"items": [], "total": 0},
        "reports/summary": anomaly_reports_summary(),
    }


def supply_chain_dashboard_kpis(project_id: str = "") -> Dict[str, Any]:
    rng = _rng_for_project(project_id)
    return {
        "total_lots": rng.randint(90, 220),
        "pending_inspections": rng.randint(8, 45),
        "released": rng.randint(70, 160),
        "held": rng.randint(4, 22),
        "rejected": rng.randint(3, 18),
        "open_capa": rng.randint(2, 14),
        "incoming_lots": rng.randint(10, 35),
        "total_items": rng.randint(1800, 5200),
        "low_stock_alerts": rng.randint(5, 28),
        "total_orders": rng.randint(120, 420),
        "pending_orders": rng.randint(12, 55),
        "total_suppliers": rng.randint(40, 95),
    }


def generic_app_dashboard(project_id: str = "") -> Dict[str, Any]:
    """Non-zero stats for generic AI dashboards."""
    rng = _rng_for_project(project_id)
    base = rng.randint(800, 2400)
    return {
        "total": base,
        "active": rng.randint(int(base * 0.5), int(base * 0.95)),
        "pending": rng.randint(20, 180),
        "completed": rng.randint(int(base * 0.4), base),
        "total_sales": rng.randint(120000, 890000),
        "orders_today": rng.randint(12, 89),
        "revenue": rng.randint(120000, 890000),
        "growth_rate": round(rng.uniform(4.0, 22.0), 1),
        "conversion_rate": round(rng.uniform(1.5, 8.5), 1),
        "active_users": rng.randint(200, 3200),
        "trend": [rng.randint(40, 140) for _ in range(10)],
    }


def _dashboard_payload_usable(payload: Dict[str, Any]) -> bool:
    if not payload:
        return False
    metrics = payload.get("metrics")
    if metrics is not None:
        return isinstance(metrics, list) and len(metrics) > 0
    numeric_keys = (
        "total_lots", "pending_inspections", "released", "total_anomalies_today",
        "active_alerts", "metrics_monitored", "total_sales", "revenue", "orders_today",
        "total", "active",
    )
    present = [k for k in numeric_keys if k in payload]
    if not present:
        return len(payload) > 2
    return any(payload.get(k) not in (0, 0.0, None, "") for k in present)


def default_dashboard_for_project(project_id: str, seed: Dict[str, Any]) -> Dict[str, Any]:
    if _project_looks_like_anomaly(project_id, seed):
        return anomaly_dashboard_payload(project_id)
    return {**generic_app_dashboard(project_id), **supply_chain_dashboard_kpis(project_id)}


def build_default_seed(
    files: Dict[str, str],
    requirement: str = "",
    prd: str = "",
    project_id: str = "",
) -> Dict[str, Any]:
    """Default seed_data.json for every generated app (preview + dynamic API)."""
    if project_looks_like_anomaly_from_files(files) or (
        requirement and any(w in requirement.lower() for w in ("anomaly", "cnc", "lathe", "monitor", "sensor"))
    ):
        return anomaly_seed_data(project_id)
    dash = default_dashboard_for_project(project_id, {})
    kpis = supply_chain_dashboard_kpis(project_id)
    return {
        "dashboard": dash,
        "kpis": {**kpis, **{k: dash[k] for k in dash if k not in kpis}},
        "dashboard/kpis": kpis,
        "stats": generic_app_dashboard(project_id),
        "anomalies": anomaly_anomalies_list(),
        "thresholds": anomaly_thresholds_list(),
        "reports/summary": anomaly_reports_summary(),
    }


def project_looks_like_anomaly_from_files(files: Dict[str, str]) -> bool:
    for path in files:
        if path.endswith("AnomalyFeedPage.tsx") or path.endswith("AlertConfigPage.tsx"):
            return True
    dash = files.get("frontend/src/pages/DashboardPage.tsx", "")
    if "metrics_monitored" in dash or "total_anomalies_today" in dash:
        return True
    for content in (files.get("frontend/src/layout/AppLayout.tsx", ""), files.get("frontend/src/App.tsx", "")):
        if "Anomaly Feed" in content or "Alert Config" in content:
            return True
    return False


def resolve_special_collection(collection: str, seed: Dict[str, Any], project_id: str = "") -> Dict[str, Any] | None:
    """Return payload for REST-style collections, or None if not a special API route."""
    key = collection.lower()
    if key in seed and isinstance(seed[key], dict):
        if key == "dashboard" and not _dashboard_payload_usable(seed[key]):
            return default_dashboard_for_project(project_id, seed)
        return seed[key]
    if key == "dashboard":
        if "dashboard" in seed and isinstance(seed["dashboard"], dict) and _dashboard_payload_usable(seed["dashboard"]):
            return seed["dashboard"]
        return default_dashboard_for_project(project_id, seed)
    if key == "anomalies":
        return seed.get("anomalies") or anomaly_anomalies_list()
    if key == "thresholds":
        return seed.get("thresholds") or anomaly_thresholds_list()
    if key == "alerts":
        return seed.get("alerts") or {"items": [], "total": 0}
    if key == "metrics":
        return seed.get("metrics") or anomaly_thresholds_list()
    return None


def resolve_special_subresource(collection: str, subpath: str, seed: Dict[str, Any]) -> Dict[str, Any] | None:
    compound = f"{collection}/{subpath}"
    if compound in seed and isinstance(seed[compound], dict):
        return seed[compound]
    if subpath in seed and isinstance(seed[subpath], dict):
        return seed[subpath]
    if collection == "reports" and subpath == "summary":
        return seed.get("reports/summary") or anomaly_reports_summary()
    if subpath in ("kpis", "stats", "summary", "overview", "metrics", "counters"):
        if _project_looks_like_anomaly("", seed):
            return anomaly_dashboard_payload("")
        kpis = seed.get("kpis")
        if isinstance(kpis, dict) and _dashboard_payload_usable(kpis):
            return kpis
        return supply_chain_dashboard_kpis("")
    return None


def _project_looks_like_anomaly(project_id: str, seed: Dict[str, Any]) -> bool:
    if seed.get("dashboard") and isinstance(seed["dashboard"], dict):
        if "metrics" in seed["dashboard"]:
            return True
    pid = (project_id or "").lower()
    if any(k in pid for k in ("anomaly", "cnc", "lathe", "monitor")):
        return True
    if project_id:
        try:
            from app_builder.services.runtime_paths import resolve_project_root
            import os

            root = resolve_project_root(project_id)
            anomaly_markers = (
                "frontend/src/pages/AnomalyFeedPage.tsx",
                "frontend/src/pages/AlertConfigPage.tsx",
            )
            if any(os.path.exists(os.path.join(root, p)) for p in anomaly_markers):
                return True
            dash_path = os.path.join(root, "frontend/src/pages/DashboardPage.tsx")
            if os.path.exists(dash_path):
                with open(dash_path, "r", encoding="utf-8", errors="replace") as fh:
                    body = fh.read()
                if "metrics_monitored" in body or "total_anomalies_today" in body:
                    return True
                if "Anomaly Feed" in body or "Alert Config" in body:
                    return True
            layout_path = os.path.join(root, "frontend/src/layout/AppLayout.tsx")
            if os.path.exists(layout_path):
                with open(layout_path, "r", encoding="utf-8", errors="replace") as fh:
                    nav = fh.read()
                if "Anomaly Feed" in nav or "Alert Config" in nav:
                    return True
        except Exception:
            pass
    return False
