from app_builder.services.synthetic_predict import (
    apply_prompt_to_features,
    default_features_from_definition,
    predict_from_features,
    run_live_prediction,
)


def test_predict_returns_stable_shape():
    result = predict_from_features(
        {
            "mtbf_hours": 200,
            "mttr_hours": 2,
            "downtime_hours": 4,
            "open_work_orders": 100,
            "machine_load_pct": 90,
            "production_output": 800,
        }
    )
    assert result["risk_level"] in ("Low", "Medium", "High")
    assert "predicted_downtime_hours_next_period" in result
    assert result["drivers"]


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
