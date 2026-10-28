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


FEATURE_KEYS = (
    "mtbf_hours",
    "mttr_hours",
    "downtime_hours",
    "open_work_orders",
    "machine_load_pct",
    "production_output",
)


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def clamp_scenario_to_baseline(
    features: Dict[str, float],
    baseline: Dict[str, float],
) -> Tuple[Dict[str, float], List[str]]:
    """
    Keep what-if inputs in a plausible band around live dashboard KPIs (not arbitrary randoms).
    """
    notes: List[str] = []
    out = dict(baseline)
    out.update(features)

    # Per-metric bands (fraction of baseline)
    bands = {
        "mtbf_hours": (0.35, 2.0),
        "mttr_hours": (0.5, 3.0),
        "downtime_hours": (0.25, 3.0),
        "open_work_orders": (0.0, 3.5),
        "machine_load_pct": (max(5.0, baseline.get("machine_load_pct", 78) * 0.5), min(100.0, baseline.get("machine_load_pct", 78) * 1.35)),
        "production_output": (0.4, 2.0),
    }

    for key in FEATURE_KEYS:
        b = float(baseline.get(key, out.get(key, 0)) or 0)
        raw = float(out.get(key, b) or 0)
        lo_frac, hi_frac = bands.get(key, (0.5, 2.0))
        if key == "machine_load_pct":
            lo, hi = lo_frac, hi_frac
        elif key == "open_work_orders":
            lo = 0.0
            hi = max(b * hi_frac, b + 50, 20.0)
        else:
            lo = max(b * lo_frac, 0.01) if b > 0 else raw
            hi = max(b * hi_frac, b * 1.01) if b > 0 else raw * 2

        clamped = _clamp(raw, lo, hi)
        if abs(clamped - raw) > 1e-6:
            notes.append(
                f"{key.replace('_', ' ')} adjusted from {round(raw, 2)} to {round(clamped, 2)} "
                f"(allowed band vs dashboard baseline)"
            )
        out[key] = round(clamped, 2)

    return out, notes


def merge_explicit_features(
    baseline: Dict[str, float],
    features: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """Use dashboard baseline; override only fields the client actually sent (non-empty)."""
    merged = dict(baseline)
    if not features:
        return merged
    for key in FEATURE_KEYS:
        if key not in features:
            continue
        raw = features[key]
        if raw is None or raw == "":
            continue
        merged[key] = _float(raw, merged.get(key, 0.0))
    return merged


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
        (r"(\d+(?:\.\d+)?)\s*h(?:ours)?\s+mtbf", "mtbf_hours", "MTBF"),
        (r"(\d+(?:\.\d+)?)\s*h(?:ours)?\s+downtime", "downtime_hours", "Downtime"),
        (r"(\d+(?:\.\d+)?)\s*%\s*oee", "machine_load_pct", "OEE"),
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


def parse_numeric_features_from_prompt(
    text: str,
    baseline: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Extract numeric field overrides from natural-language text."""
    base = baseline or default_features_from_definition(None)
    after, _ = apply_prompt_to_features(text, dict(base), base)
    keys = (
        "mtbf_hours",
        "mttr_hours",
        "downtime_hours",
        "open_work_orders",
        "machine_load_pct",
        "production_output",
    )
    return {k: after[k] for k in keys if after.get(k) != base.get(k)}


def predict_from_features(
    features: Dict[str, Any],
    *,
    baseline: Optional[Dict[str, float]] = None,
    app_definition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Forecast next period — stays in range of dashboard baseline + scenario inputs."""
    base = baseline or default_features_from_definition(
        app_definition if isinstance(app_definition, dict) else None
    )
    mtbf = _float(features.get("mtbf_hours"), base["mtbf_hours"])
    mttr = _float(features.get("mttr_hours"), base["mttr_hours"])
    downtime = _float(features.get("downtime_hours"), base["downtime_hours"])
    work_orders = _float(features.get("open_work_orders"), base["open_work_orders"])
    load = _float(features.get("machine_load_pct"), base["machine_load_pct"])
    output = _float(features.get("production_output"), base["production_output"])

    def _rel_delta(cur: float, ref: float) -> float:
        if ref <= 0:
            return 0.0
        return (cur - ref) / ref

    d_load = _rel_delta(load, base["machine_load_pct"])
    d_wo = _rel_delta(work_orders, base["open_work_orders"])
    d_mtbf = _rel_delta(base["mtbf_hours"], mtbf)  # lower MTBF vs baseline → positive stress
    d_down = _rel_delta(downtime, base["downtime_hours"])
    d_out = _rel_delta(output, base["production_output"])

    # Small, interpretable adjustments vs baseline (typical next-period band ~±35%)
    pred_downtime = base["downtime_hours"] * (
        1.0 + 0.35 * d_down + 0.28 * d_load + 0.22 * d_wo + 0.25 * d_mtbf
    )
    pred_downtime = _clamp(
        pred_downtime,
        base["downtime_hours"] * 0.55,
        base["downtime_hours"] * 2.0,
    )

    pred_mtbf = base["mtbf_hours"] * (
        1.0 - 0.12 * d_load - 0.08 * d_wo + 0.45 * _rel_delta(mtbf, base["mtbf_hours"])
    )
    pred_mtbf = _clamp(
        pred_mtbf,
        base["mtbf_hours"] * 0.72,
        base["mtbf_hours"] * 1.28,
    )

    pred_output = base["production_output"] * (
        1.0 + 0.15 * d_out - 0.18 * _rel_delta(pred_downtime, base["downtime_hours"])
    )
    pred_output = _clamp(
        pred_output,
        base["production_output"] * 0.75,
        base["production_output"] * 1.25,
    )

    pred_downtime = round(pred_downtime, 2)
    pred_mtbf = round(pred_mtbf, 1)
    pred_output = round(pred_output, 1)

    ref_down = max(base["downtime_hours"], 0.1)
    if pred_downtime >= base["downtime_hours"] * 1.6 or work_orders >= base["open_work_orders"] * 1.5 + 20:
        risk = "High"
    elif pred_downtime >= base["downtime_hours"] * 1.2 or work_orders >= base["open_work_orders"] * 1.2:
        risk = "Medium"
    else:
        risk = "Low"

    drivers: List[str] = []
    if d_load > 0.08:
        drivers.append("Higher load vs dashboard baseline")
    if d_wo > 0.15:
        drivers.append("More open work orders than baseline")
    if mtbf < base["mtbf_hours"] * 0.9:
        drivers.append("MTBF below dashboard baseline")
    if mttr > base["mttr_hours"] * 1.2:
        drivers.append("MTTR above dashboard baseline")
    if not drivers:
        drivers.append("Scenario aligned with current dashboard operating range")

    rec = (
        "Schedule preventive maintenance on top-loaded assets and review open work orders."
        if risk == "High"
        else "Monitor downtime trend; keep spare parts for high-failure equipment."
        if risk == "Medium"
        else "Continue current maintenance cadence; re-run after KPIs change."
    )

    spread = abs(d_load) + abs(d_wo) + abs(d_mtbf) + abs(d_down)
    confidence = round(_clamp(0.78 - spread * 0.12, 0.62, 0.92), 2)

    return {
        "predicted_downtime_hours_next_period": pred_downtime,
        "predicted_mtbf_hours": pred_mtbf,
        "predicted_production_output": pred_output,
        "risk_level": risk,
        "confidence": confidence,
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
        "baseline_reference": base,
    }


def _delta_pct(predicted: float, baseline: float) -> Optional[float]:
    if baseline == 0:
        return None
    return round((predicted - baseline) / baseline * 100, 1)


def build_prediction_indicators(
    prediction: Dict[str, Any],
    scenario_inputs: Dict[str, float],
) -> List[Dict[str, Any]]:
    used = prediction.get("inputs_used") or {}
    dash = prediction.get("baseline_reference") or scenario_inputs
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
        base = float(dash.get(baseline_key, used.get(baseline_key, predicted)))
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
    """Merge dashboard baseline → form fields → prompt, clamp to KPI band, then forecast."""
    baseline = default_features_from_definition(defn)
    merged = merge_explicit_features(baseline, features)
    merged, prompt_notes = apply_prompt_to_features(prompt, merged, baseline)
    merged, clamp_notes = clamp_scenario_to_baseline(merged, baseline)
    scenario_notes = list(prompt_notes) + clamp_notes

    core = predict_from_features(merged, baseline=baseline, app_definition=defn)
    return {
        **core,
        "baseline": baseline,
        "scenario_inputs": merged,
        "indicators": build_prediction_indicators(core, merged),
        "scenario_notes": scenario_notes,
        "prompt": (prompt or "").strip(),
        "computed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
