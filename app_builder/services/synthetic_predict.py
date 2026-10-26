"""Deterministic predictions over synthetic maintenance / operations inputs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _float(val: Any, default: float) -> float:
    try:
        if isinstance(val, str):
            val = val.replace(",", "").replace(" h", "").replace("%", "").strip()
        return float(val)
    except (TypeError, ValueError):
        return default


def predict_from_features(
    features: Dict[str, Any],
    *,
    app_definition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Lightweight synthetic model — stable demo output for UI + chat."""
    mtbf = _float(features.get("mtbf_hours"), 240.0)
    mttr = _float(features.get("mttr_hours"), 1.8)
    downtime = _float(features.get("downtime_hours"), 2.5)
    work_orders = _float(features.get("open_work_orders"), 48.0)
    load = _float(features.get("machine_load_pct"), 78.0)
    output = _float(features.get("production_output"), 850.0)

    load_factor = max(0.6, min(1.35, load / 80.0))
    reliability = max(0.5, min(1.2, mtbf / 280.0))
    wo_stress = max(0.8, min(1.4, work_orders / 60.0))

    pred_downtime = round(downtime * load_factor * wo_stress / reliability, 2)
    pred_mtbf = round(mtbf * (0.92 + (100 - load) / 500), 1)
    pred_output = round(output * (1 - pred_downtime / 48) * reliability, 1)

    if pred_downtime >= 5 or work_orders >= 120:
        risk = "High"
    elif pred_downtime >= 2.5 or work_orders >= 80:
        risk = "Medium"
    else:
        risk = "Low"

    drivers: List[str] = []
    if load >= 85:
        drivers.append("High machine load")
    if work_orders >= 80:
        drivers.append("Elevated open work orders")
    if mtbf < 180:
        drivers.append("Below-target MTBF")
    if mttr > 3:
        drivers.append("Long MTTR")
    if not drivers:
        drivers.append("Stable operating conditions in synthetic baseline")

    rec = (
        "Schedule preventive maintenance on top-loaded assets and review open work orders."
        if risk == "High"
        else "Monitor downtime trend; keep spare parts for high-failure equipment."
        if risk == "Medium"
        else "Continue current maintenance cadence; re-run prediction after next shift."
    )

    return {
        "predicted_downtime_hours_next_period": pred_downtime,
        "predicted_mtbf_hours": pred_mtbf,
        "predicted_production_output": pred_output,
        "risk_level": risk,
        "confidence": round(0.75 + min(0.2, mtbf / 2000), 2),
        "drivers": drivers,
        "recommendation": rec,
        "inputs_used": {
            "mtbf_hours": mtbf,
            "mttr_hours": mttr,
            "downtime_hours": downtime,
            "open_work_orders": work_orders,
            "machine_load_pct": load,
            "production_output": output,
        },
    }


def default_features_from_definition(defn: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Seed predict form from KPI bundles in app data."""
    if not defn:
        return {
            "mtbf_hours": 240.0,
            "mttr_hours": 1.8,
            "downtime_hours": 2.5,
            "open_work_orders": 48.0,
            "machine_load_pct": 78.0,
            "production_output": 850.0,
        }
    data = defn.get("data") or defn.get("mock") or {}
    kpis = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if str(k).startswith("kpis_") and isinstance(v, dict):
                kpis.update(v)
    def _pick(keys: List[str], default: float) -> float:
        for key in keys:
            if key in kpis:
                return _float(kpis[key], default)
        return default

    return {
        "mtbf_hours": _pick(["mtbf_hours", "mtbf", "MTBF"], 240.0),
        "mttr_hours": _pick(["mttr_hours", "mttr", "MTTR"], 1.8),
        "downtime_hours": _pick(["downtime_hours", "downtime"], 2.5),
        "open_work_orders": _pick(["open_work_orders", "work_orders"], 48.0),
        "machine_load_pct": _pick(["machine_load", "oee", "utilization"], 78.0),
        "production_output": _pick(["production_output", "output", "units"], 850.0),
    }
