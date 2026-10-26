"""Live predictions from app KPIs + optional natural-language scenario prompts."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _float(val: Any, default: float) -> float:
    try:
        if isinstance(val, str):
            s = val.replace(",", "").strip().lower()
            if "min" in s:
                num = re.sub(r"[^\d.]", "", s.split("min")[0])
                return float(num or 0) / 60.0
            if "h" in s and "%" not in s:
                num = re.sub(r"[^\d.]", "", s.split("h")[0])
                return float(num or 0)
            s = s.replace(" h", "").replace("%", "").strip()
            return float(re.sub(r"[^\d.]", "", s) or 0)
        return float(val)
    except (TypeError, ValueError):
        return default


def _collect_kpi_bag(defn: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not defn:
        return {}
    data = defn.get("data") or defn.get("mock") or {}
    kpis: Dict[str, Any] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if str(k).startswith("kpis_") and isinstance(v, dict):
                for kk, vv in v.items():
                    kpis[str(kk).lower()] = vv
    return kpis


def default_features_from_definition(defn: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Seed features from all KPI bundles in the live app definition."""
    kpis = _collect_kpi_bag(defn)
    if not kpis:
        return {
            "mtbf_hours": 240.0,
            "mttr_hours": 1.8,
            "downtime_hours": 2.5,
            "open_work_orders": 48.0,
            "machine_load_pct": 78.0,
            "production_output": 850.0,
        }

    def _pick(keys: List[str], default: float) -> float:
        for key in keys:
            lk = key.lower()
            if lk in kpis:
                return _float(kpis[lk], default)
            for kk, vv in kpis.items():
                if lk in kk.replace("_", " ") or kk == lk:
                    return _float(vv, default)
        return default

    throughput = _pick(["throughput", "production_output", "output", "units"], 850.0)
    if throughput > 5000:
        throughput = throughput / 10.0

    downtime = _pick(["downtime_hours", "downtime", "unplanned_downtime"], 2.5)
    if downtime > 24 and downtime < 500:
        downtime = downtime / 60.0

    load = _pick(["machine_load_pct", "machine_load", "oee", "utilization", "load"], 78.0)

    return {
        "mtbf_hours": _pick(["mtbf_hours", "mtbf"], 240.0),
        "mttr_hours": _pick(["mttr_hours", "mttr"], 1.8),
        "downtime_hours": downtime,
        "open_work_orders": _pick(["open_work_orders", "work_orders", "active_alerts", "alerts"], 48.0),
        "machine_load_pct": load,
        "production_output": throughput,
    }


def apply_prompt_to_features(
    prompt: str,
    features: Dict[str, float],
    baseline: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """Apply what-if phrases from the user prompt on top of current features."""
    text = (prompt or "").strip()
    if not text:
        return dict(features), []

    out = dict(features)
    notes: List[str] = []
    low = text.lower()
    base = baseline or features

    def set_num(key: str, val: float, note: str) -> None:
        out[key] = val
        notes.append(note)

    pairs = [
        (r"(?:machine\s+)?load(?:\s+at|\s+to|\s*=|:)?\s*(\d+(?:\.\d+)?)\s*%?", "machine_load_pct", "Machine load"),
        (r"oee\s*(?:at|to|=|:)?\s*(\d+(?:\.\d+)?)\s*%?", "machine_load_pct", "OEE / load"),
        (r"mtbf\s*(?:at|to|=|:)?\s*(\d+(?:\.\d+)?)", "mtbf_hours", "MTBF"),
        (r"mttr\s*(?:at|to|=|:)?\s*(\d+(?:\.\d+)?)", "mttr_hours", "MTTR"),
        (r"downtime\s*(?:at|to|=|:)?\s*(\d+(?:\.\d+)?)", "downtime_hours", "Downtime"),
        (r"(\d+(?:\.\d+)?)\s*%\s*(?:machine\s+)?load", "machine_load_pct", "Machine load"),
        (r"(\d+(?:\.\d+)?)\s+open\s+work\s+orders?", "open_work_orders", "Open work orders"),
        (r"work\s+orders?\s*(?:at|to|=|:)?\s*(\d+(?:\.\d+)?)", "open_work_orders", "Work orders"),
        (r"output\s*(?:at|to|=|:)?\s*(\d+(?:\.\d+)?)", "production_output", "Output"),
        (r"throughput\s*(?:at|to|=|:)?\s*(\d+(?:\.\d+)?)", "production_output", "Throughput"),
    ]
    for pattern, key, label in pairs:
        m = re.search(pattern, low)
        if m:
            set_num(key, float(m.group(1)), f"{label} set to {m.group(1)} from your prompt")

    if re.search(r"\b(double|2x|twice)\b.*\bdowntime\b", low) or re.search(r"\bdowntime\b.*\b(double|2x)\b", low):
        set_num("downtime_hours", base.get("downtime_hours", 2.5) * 2, "Downtime doubled (what-if)")

    inc = re.search(r"(?:increase|raise|up)\s+(?:downtime|load|work\s+orders?).*?(\d+)\s*%", low)
    if inc:
        pct = float(inc.group(1)) / 100.0
        if "downtime" in inc.group(0):
            set_num("downtime_hours", out.get("downtime_hours", 2.5) * (1 + pct), f"Downtime +{inc.group(1)}%")
        elif "load" in inc.group(0):
            set_num("machine_load_pct", min(100.0, out.get("machine_load_pct", 78) * (1 + pct)), f"Load +{inc.group(1)}%")

    if re.search(r"\bwhat\s+if\b", low) and not notes:
        notes.append("What-if scenario using live dashboard KPIs as baseline")

    return out, notes


def predict_from_features(
    features: Dict[str, Any],
    *,
    app_definition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
        drivers.append("Stable operating conditions vs live baseline")

    rec = (
        "Schedule preventive maintenance on top-loaded assets and review open work orders."
        if risk == "High"
        else "Monitor downtime trend; keep spare parts for high-failure equipment."
        if risk == "Medium"
        else "Continue current maintenance cadence; re-run after KPIs change."
    )

    return {
        "predicted_downtime_hours_next_period": pred_downtime,
        "predicted_mtbf_hours": pred_mtbf,
        "predicted_production_output": pred_output,
        "risk_level": risk,
        "confidence": round(0.72 + min(0.23, mtbf / 1800), 2),
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


def _delta_pct(predicted: float, baseline: float) -> Optional[float]:
    if baseline == 0:
        return None
    return round((predicted - baseline) / baseline * 100, 1)


def build_prediction_indicators(
    prediction: Dict[str, Any],
    baseline: Dict[str, float],
) -> List[Dict[str, Any]]:
    used = prediction.get("inputs_used") or {}
    risk = str(prediction.get("risk_level") or "Low")
    risk_sev = {"High": "error", "Medium": "warning", "Low": "success"}.get(risk, "info")

    def metric(
        id_: str,
        label: str,
        predicted: float,
        baseline_key: str,
        unit: str,
        *,
        higher_is_bad: bool = False,
    ) -> Dict[str, Any]:
        base = float(baseline.get(baseline_key, predicted))
        delta = _delta_pct(predicted, base)
        trend = "up" if delta and delta > 0 else "down" if delta and delta < 0 else "flat"
        if delta is not None and higher_is_bad:
            status = "error" if delta > 5 else "warning" if delta > 0 else "success"
        elif delta is not None:
            status = "success" if delta > 0 else "warning" if delta < -5 else "info"
        else:
            status = "info"
        return {
            "id": id_,
            "kind": "metric",
            "label": label,
            "value": predicted,
            "unit": unit,
            "baseline": base,
            "delta_pct": delta,
            "trend": trend,
            "severity": status,
        }

    conf = float(prediction.get("confidence") or 0.75)
    return [
        {
            "id": "risk",
            "kind": "status",
            "label": "Operational risk",
            "value": risk,
            "severity": risk_sev,
            "confidence": conf,
        },
        metric(
            "downtime",
            "Predicted downtime (next period)",
            float(prediction.get("predicted_downtime_hours_next_period") or 0),
            "downtime_hours",
            "h",
            higher_is_bad=True,
        ),
        metric(
            "mtbf",
            "Predicted MTBF",
            float(prediction.get("predicted_mtbf_hours") or 0),
            "mtbf_hours",
            "h",
        ),
        metric(
            "output",
            "Predicted output",
            float(prediction.get("predicted_production_output") or 0),
            "production_output",
            "units",
        ),
        {
            "id": "confidence",
            "kind": "gauge",
            "label": "Model confidence",
            "value": round(conf * 100, 0),
            "unit": "%",
            "severity": "info" if conf >= 0.8 else "warning",
        },
    ]


def run_live_prediction(
    defn: Optional[Dict[str, Any]],
    *,
    features: Optional[Dict[str, Any]] = None,
    prompt: str = "",
) -> Dict[str, Any]:
    """Re-read KPIs from definition, merge form + prompt, then forecast."""
    baseline = default_features_from_definition(defn)
    merged = dict(baseline)
    if features:
        for k, v in features.items():
            merged[k] = _float(v, merged.get(k, 0.0))
    merged, scenario_notes = apply_prompt_to_features(prompt, merged, baseline)
    core = predict_from_features(merged, app_definition=defn)
    return {
        **core,
        "baseline": baseline,
        "indicators": build_prediction_indicators(core, baseline),
        "scenario_notes": scenario_notes,
        "prompt": (prompt or "").strip(),
        "computed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
