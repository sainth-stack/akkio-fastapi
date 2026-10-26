from app_builder.services.synthetic_predict import default_features_from_definition, predict_from_features


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
                "mtbf_hours": 300,
                "downtime_hours": 1.5,
            }
        }
    }
    feats = default_features_from_definition(defn)
    assert feats["mtbf_hours"] == 300.0
    assert feats["downtime_hours"] == 1.5
