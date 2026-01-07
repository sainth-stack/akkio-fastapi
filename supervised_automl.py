import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def _safe_json(obj: Any) -> Any:
    """Convert numpy/pandas objects to JSON-serializable Python natives."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    return str(obj)


def _rmse(y_true, y_pred) -> float:
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def detect_task_type(y: pd.Series, numeric_threshold: float = 0.75) -> str:
    """
    Decide classification vs regression based on target distribution.
    Returns: "classification" | "regression"
    """
    try:
        if y is None or y.empty:
            return "classification"
        y_non_null = y.dropna()
        if y_non_null.empty:
            return "classification"

        # Try numeric conversion (CSV-safe)
        y_numeric = pd.to_numeric(y_non_null, errors="coerce")
        numeric_ratio = float(y_numeric.notna().mean())

        # If NOT mostly numeric → classification
        if numeric_ratio < numeric_threshold:
            return "classification"

        # From here: target is numeric
        y_clean = y_numeric.dropna()
        unique_count = int(pd.Series(y_clean).nunique())

        # Strict classification checks ONLY
        if unique_count == 2:
            return "classification"
        if unique_count <= 10:
            y_min = float(y_clean.min())
            y_max = float(y_clean.max())
            if y_min >= 0 and y_max <= unique_count:
                return "classification"

        return "regression"
    except Exception:
        # safest fallback: regression (avoid forcing classification)
        return "regression"


def _coerce_datetime_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Convert datetime64 columns to numeric seconds since epoch so they can be modeled."""
    df = df.copy()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64", "datetimetz"]).columns.tolist()
    for c in datetime_cols:
        # keep NaT as NaN
        s = pd.to_datetime(df[c], errors="coerce")
        # seconds since epoch as float; NaT -> NaN (avoid -9223372036... sentinel)
        ns = s.view("int64").astype("float64")
        ns[pd.isna(s)] = np.nan
        df[c] = ns / 1e9
    return df, datetime_cols


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_features = X.select_dtypes(include=["int64", "float64", "Int64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features


def _available_models(task_type: str) -> Tuple[Dict[str, Any], str]:
    """
    Returns model dict and metric name.
    Metric: RMSE for regression (lower better), F1 for classification (higher better).
    """
    models: Dict[str, Any] = {}
    metric_name = "RMSE" if task_type == "regression" else "F1"

    if task_type == "regression":
        models["LinearRegression"] = LinearRegression()
        models["RandomForest"] = RandomForestRegressor(n_estimators=200, random_state=42)
        # Optional: XGBoost / LightGBM
        try:
            from xgboost import XGBRegressor  # type: ignore

            models["XGBoost"] = XGBRegressor(
                n_estimators=300, learning_rate=0.05, random_state=42, objective="reg:squarederror"
            )
        except Exception:
            pass
        try:
            from lightgbm import LGBMRegressor  # type: ignore

            models["LightGBM"] = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
        except Exception:
            pass
    else:
        models["LogisticRegression"] = LogisticRegression(max_iter=2000)
        models["RandomForest"] = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        try:
            from xgboost import XGBClassifier  # type: ignore

            models["XGBoost"] = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                random_state=42,
                eval_metric="logloss",
            )
        except Exception:
            pass
        try:
            from lightgbm import LGBMClassifier  # type: ignore

            models["LightGBM"] = LGBMClassifier(random_state=42)
        except Exception:
            pass

    return models, metric_name


def train_supervised_automl(
    df: pd.DataFrame,
    target_col: str,
    model_dir: str,
) -> Dict[str, Any]:
    """
    Train multiple supervised models, select best by metric, and persist under model_dir.
    Creates:
      - metrics.json (best_model, per-model scores)
      - deployment.json (feature metadata, task type, etc.)
      - <ModelName>.pkl bundles (pipeline + optional label encoder)
    """
    os.makedirs(model_dir, exist_ok=True)

    metrics_path = os.path.join(model_dir, "metrics.json")
    deployment_path = os.path.join(model_dir, "deployment.json")

    # Reuse if already trained
    if os.path.exists(metrics_path) and os.path.exists(deployment_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
            with open(deployment_path, "r", encoding="utf-8") as f:
                deployment = json.load(f)
            return {
                "status": True,
                "reused": True,
                "model_dir": model_dir,
                "task_type": registry.get("task_type"),
                "metric_type": registry.get("metric_type"),
                "best_model": registry.get("best_model"),
                "metric": (registry.get("models", {}) or {}).get(registry.get("best_model"), {}).get("output", {}).get(
                    registry.get("metric_type")
                ),
                "all_models": registry.get("models", {}),
                "feature_columns": deployment.get("feature_names", []),
                "row_data": deployment.get("row_data", {}),
                "skipped_models": registry.get("skipped_models", []),
            }
        except Exception:
            # fall through to retrain if cache is corrupted
            pass

    if target_col not in df.columns:
        return {"status": False, "message": f"Target column '{target_col}' not found"}

    df = df.copy()
    df = df[df[target_col].notna()].copy()
    if df.empty:
        return {"status": False, "message": "Target column empty after NaN removal"}

    # Convert datetime cols → numeric (seconds) to allow modeling
    df, datetime_cols = _coerce_datetime_features(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    task_type = detect_task_type(y)
    preprocessor, numerical_features, categorical_features = build_preprocessor(X)

    label_encoder: Optional[LabelEncoder] = None
    if task_type == "classification":
        label_encoder = LabelEncoder()
        y_enc = label_encoder.fit_transform(y.astype(str))
        y_for_split = y_enc
    else:
        y_for_split = pd.to_numeric(y, errors="coerce")

    # Basic sanity
    if task_type == "regression" and pd.Series(y_for_split).dropna().empty:
        return {"status": False, "message": "Target is not numeric enough for regression"}

    # Split safely
    stratify = None
    if task_type == "classification":
        try:
            # need at least 2 classes and enough per class
            if len(np.unique(y_for_split)) >= 2:
                stratify = y_for_split
        except Exception:
            stratify = None

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_for_split,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    models, metric_name = _available_models(task_type)

    registry: Dict[str, Any] = {
        "task_type": task_type,
        "metric_type": metric_name,
        "best_model": None,
        "models": {},
        "skipped_models": [],
        "trained_at": datetime.now().isoformat(),
    }

    if task_type == "regression":
        best_score = float("inf")
        is_better = lambda s, b: s < b  # noqa: E731
    else:
        best_score = -1.0
        is_better = lambda s, b: s > b  # noqa: E731

    for name, model in models.items():
        try:
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)

            if task_type == "regression":
                score = round(_rmse(y_val, preds), 6)
            else:
                score = round(float(f1_score(y_val, preds, average="weighted", zero_division=0)), 6)

            model_bundle = {"pipeline": pipeline, "label_encoder": label_encoder}
            joblib.dump(model_bundle, os.path.join(model_dir, f"{name}.pkl"))

            registry["models"][name] = {
                "input": {
                    "hyperparameters": _safe_json(model.get_params() if hasattr(model, "get_params") else {}),
                    "train_size": int(len(X_train)),
                    "validation_size": int(len(X_val)),
                },
                "output": {metric_name: score},
            }

            if is_better(score, best_score):
                best_score = score
                registry["best_model"] = name
        except Exception as e:
            registry["skipped_models"].append({"model": name, "error": str(e)})
            continue

    if not registry.get("best_model"):
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(_safe_json(registry), f, indent=2)
        return {"status": False, "message": "No models could be trained", "details": registry.get("skipped_models")}

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_safe_json(registry), f, indent=2)

    # Sample row for form prefilling (same idea as RF)
    row_data: Dict[str, Any] = {}
    for col in X.columns:
        v = X.iloc[0][col]
        if pd.isna(v):
            row_data[col] = None
        elif isinstance(v, (np.integer, np.int64, np.int32)):
            row_data[col] = int(v)
        elif isinstance(v, (np.floating, np.float64, np.float32)):
            row_data[col] = float(v)
        else:
            row_data[col] = str(v)

    deployment = {
        "target_column": target_col,
        "task_type": task_type,
        "best_model": registry["best_model"],
        "metric_type": metric_name,
        "metric": (registry["models"][registry["best_model"]]["output"][metric_name]),
        "feature_names": X.columns.tolist(),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "datetime_features": datetime_cols,
        "row_data": row_data,
        "trained_at": registry.get("trained_at"),
    }

    with open(deployment_path, "w", encoding="utf-8") as f:
        json.dump(_safe_json(deployment), f, indent=2)

    return {
        "status": True,
        "reused": False,
        "model_dir": model_dir,
        "task_type": task_type,
        "best_model": registry["best_model"],
        "metric_type": metric_name,
        "metric": deployment["metric"],
        "all_models": registry["models"],
        "feature_columns": deployment["feature_names"],
        "row_data": row_data,
        "skipped_models": registry.get("skipped_models", []),
    }


def load_best_model(model_dir: str) -> Tuple[Pipeline, Optional[LabelEncoder], Dict[str, Any]]:
    metrics_path = os.path.join(model_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError("metrics.json not found")

    with open(metrics_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    model_name = registry.get("best_model")
    if not model_name:
        raise ValueError("No best_model selected in metrics.json")

    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_name}.pkl not found")

    model_bundle = joblib.load(model_path)
    pipeline = model_bundle["pipeline"]
    label_encoder = model_bundle.get("label_encoder")
    return pipeline, label_encoder, registry


def predict_supervised(model_dir: str, input_df: pd.DataFrame) -> Dict[str, Any]:
    pipeline, label_encoder, registry = load_best_model(model_dir)
    preds = pipeline.predict(input_df)

    if label_encoder is not None:
        try:
            preds_out = label_encoder.inverse_transform(preds.astype(int))
        except Exception:
            preds_out = preds
    else:
        preds_out = preds

    response: Dict[str, Any] = {"predictions": _safe_json(pd.Series(preds_out).tolist())}

    # probabilities if classification and supported
    if registry.get("task_type") == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(input_df)
            if label_encoder is not None:
                class_labels = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))
                response["probabilities"] = [dict(zip(map(str, class_labels), map(float, p))) for p in probs]
            else:
                response["probabilities"] = _safe_json(probs.tolist())
        except Exception:
            pass

    return response


