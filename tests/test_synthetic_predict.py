from app_builder.services.synthetic_predict import (
    apply_prompt_to_features,
    clamp_scenario_to_baseline,
    default_features_from_definition,
    predict_from_features,
    run_live_prediction,
)


def test_predict_returns_stable_shape():
    base = {
        "mtbf_hours": 200.0,
        "mttr_hours": 2.0,
        "downtime_hours": 4.0,
        "open_work_orders": 50.0,
        "machine_load_pct": 85.0,
        "production_output": 800.0,
    }
    result = predict_from_features(base, baseline=base)
    assert result["risk_level"] in ("Low", "Medium", "High")
    assert "predicted_downtime_hours_next_period" in result
    assert result["drivers"]
    pd = result["predicted_downtime_hours_next_period"]
    assert base["downtime_hours"] * 0.5 <= pd <= base["downtime_hours"] * 2.1


def test_default_features_from_definition():
    defn = {
        "data": {
            "kpis_dashboard": {
                "downtime": "42 min",
                "oee": "87.4%",
            },
            "kpis_analytics": {
                "mtbf": "128 h",
            },
        }
    }
    feats = default_features_from_definition(defn)
    assert feats["mtbf_hours"] == 128.0
    assert abs(feats["downtime_hours"] - 0.7) < 0.01
    assert feats["machine_load_pct"] == 87.4


def test_prompt_overrides_load():
    base = default_features_from_definition(None)
    merged, notes = apply_prompt_to_features("what if machine load is 92%", base, base)
    assert merged["machine_load_pct"] == 92.0
    assert notes


def test_run_live_prediction_has_indicators():
    out = run_live_prediction(
        {"data": {"kpis_analytics": {"mtbf": "200 h"}}},
        prompt="forecast next period",
    )
    assert out["indicators"]
    assert out["computed_at"]


def test_clamp_extreme_inputs():
    base = default_features_from_definition(
        {"data": {"kpis_analytics": {"mtbf": "128 h", "downtime": "42 min"}}}
    )
    wild = dict(base)
    wild["mtbf_hours"] = 9999
    wild["downtime_hours"] = 0.001
    clamped, notes = clamp_scenario_to_baseline(wild, base)
    assert clamped["mtbf_hours"] <= base["mtbf_hours"] * 2.01
    assert notes


def test_different_inputs_different_outputs():
    base = default_features_from_definition(None)
    low = predict_from_features(base, baseline=base)
    stressed = dict(base)
    stressed["machine_load_pct"] = min(100, base["machine_load_pct"] * 1.2)
    stressed["open_work_orders"] = base["open_work_orders"] * 1.4
    high = predict_from_features(stressed, baseline=base)
    assert (
        low["predicted_downtime_hours_next_period"]
        != high["predicted_downtime_hours_next_period"]
        or low["risk_level"] != high["risk_level"]
    )
