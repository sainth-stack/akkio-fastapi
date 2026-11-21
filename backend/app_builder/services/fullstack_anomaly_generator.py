"""Anomaly Detection app generator for Agentic Builder — config-driven SaaS template."""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from app_builder.services.anomaly_saas_frontend import (
    build_anomaly_backend_files,
    build_anomaly_frontend_files,
    build_data_bundle,
)

# ──────────────────────────────────────────────────────────────────────────────
# Domain detection
# ──────────────────────────────────────────────────────────────────────────────

ANOMALY_KEYWORDS = (
    "anomaly detection", "anomaly monitoring", "anomaly alert",
    "outlier detection", "outlier monitoring",
    "threshold alert", "threshold breach",
    "time series anomaly", "metric monitoring",
    "z-score detection", "zscore anomaly",
    "infrastructure monitoring", "observability platform",
    "predictive monitoring", "anomaly detection system",
    "anomaly detection platform", "anomaly detection app",
)

ANOMALY_SUPPORTING = (
    "anomaly", "anomalies", "outlier", "outliers",
    "threshold", "metric", "metrics",
    "alert", "alerts", "monitoring",
    "time series", "time-series",
    "cpu", "memory", "latency", "error rate",
    "detection", "sensor", "iot", "telemetry",
    "z-score", "zscore", "severity",
)


def is_anomaly_detection_domain(requirement: str, prd: str = "", uiux: str = "") -> bool:
    text = "\n".join([requirement or "", prd or "", uiux or ""]).lower()
    try:
        from app_builder.services.json_app_routing import is_operations_dashboard_prompt

        if is_operations_dashboard_prompt(requirement, prd, uiux):
            return False
    except ImportError:
        pass
    if re.search(r"\banomaly detection\b|\banomaly monitoring\b|\banomaly platform\b", text):
        pass  # continue normal scoring
    elif re.search(r"maintenance|manufacturing", text) and re.search(
        r"dashboard|analytics|reports?", text
    ):
        return False
    strong = sum(1 for k in ANOMALY_KEYWORDS if k in text)
    if strong >= 1:
        return True
    support = sum(1 for k in ANOMALY_SUPPORTING if k in text)
    return support >= 4


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def anomaly_detection_frontend_files(
    title: str,
    colors: Dict[str, str],
    requirement: str = "",
    prd: str = "",
    uiux: str = "",
    architecture: Optional[Dict[str, Any]] = None,
    design_tokens: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Return frontend files: stable pages + generated appConfig.ts and mockData.ts."""
    return build_anomaly_frontend_files(
        title,
        colors,
        requirement=requirement,
        prd=prd,
        uiux=uiux,
        architecture=architecture,
        design_tokens=design_tokens,
    )


def anomaly_detection_backend_files(
    title: str,
    requirement: str = "",
    prd: str = "",
    colors: Dict[str, str] | None = None,
    uiux: str = "",
    architecture: Optional[Dict[str, Any]] = None,
    design_tokens: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Return backend with app_data.json shared with frontend mock data."""
    return build_anomaly_backend_files(
        title,
        colors or {},
        requirement=requirement,
        prd=prd,
        uiux=uiux,
        architecture=architecture,
        design_tokens=design_tokens,
    )


__all__ = [
    "ANOMALY_KEYWORDS",
    "ANOMALY_SUPPORTING",
    "is_anomaly_detection_domain",
    "anomaly_detection_frontend_files",
    "anomaly_detection_backend_files",
    "build_data_bundle",
]
