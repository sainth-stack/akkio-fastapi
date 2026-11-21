"""Parse PRD chart bullets into chart blocks with meaningful series and axes."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_CHART_COLORS = ("#2563eb", "#16a34a", "#ea580c", "#7c3aed", "#0891b2", "#dc2626", "#64748b")


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:32] or "value"


def _detect_metrics(text: str) -> List[Tuple[str, str]]:
    """Return (dataKey, displayName) pairs mentioned in chart description."""
    low = text.lower()
    catalog: List[Tuple[str, str, str]] = [
        (r"\boee\b", "oee", "OEE %"),
        (r"first.?pass|fpy", "fpy", "First-pass yield"),
        (r"throughput", "throughput", "Throughput"),
        (r"scrap", "scrap_rate", "Scrap rate"),
        (r"defect", "defects", "Defects"),
        (r"yield", "yield_pct", "Yield %"),
        (r"downtime", "downtime", "Downtime"),
        (r"output", "output", "Output"),
        (r"performance", "performance", "Performance"),
        (r"operational", "operational_index", "Operational index"),
        (r"planned|plan\b", "planned", "Planned"),
        (r"actual", "actual", "Actual"),
        (r"target|benchmark", "target", "Target"),
        (r"utilization", "utilization", "Utilization %"),
        (r"energy", "energy", "Energy / unit"),
        (r"mtbf", "mtbf", "MTBF"),
        (r"alerts?", "alerts", "Alerts"),
    ]
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for pat, key, name in catalog:
        if re.search(pat, low) and key not in seen:
            seen.add(key)
            out.append((key, name))
    return out


def _pie_segments(text: str) -> List[str]:
    low = text.lower()
    if " vs " in low or " versus " in low:
        parts = re.split(r"\s+vs\.?\s+|\s+versus\s+", text, flags=re.I)
        return [p.strip(" -") for p in parts if p.strip()]
    if re.search(r"passed|rework|scrap|hold", low):
        return ["Passed", "Rework", "Scrapped", "Hold", "Other"]
    chunks = re.split(r",|/|\band\b", text)
    labels = [c.strip(" -") for c in chunks if 2 < len(c.strip()) < 40]
    return labels[:8] if labels else ["Segment A", "Segment B", "Segment C", "Other"]


def infer_x_axis_key(description: str, chart_type: str) -> str:
    low = description.lower()
    if re.search(r"over time|trend|daily|weekly|monthly|by date|timeline", low):
        return "period"
    if re.search(r"by shift|per shift|shift", low):
        return "shift"
    if re.search(r"by line|production line|per line", low):
        return "category"
    if re.search(r"by category|pareto|defect type|by owner|by operator", low):
        return "category"
    if chart_type == "barChart":
        return "category"
    return "period"


def chart_block_from_bullet(line: str, page_id: str, index: int) -> Dict[str, Any]:
    raw = re.sub(r"^\*+|\*+$", "", line).strip()
    title = re.sub(r"^(line|bar|donut|pie)\s+chart\s*:\s*", "", raw, flags=re.I).strip()
    title = title[:80] or "Trend"
    desc = title
    dk_base = f"{page_id}_{index}"

    if re.search(r"donut|pie\b", line, re.I):
        return {
            "type": "pieChart",
            "title": title,
            "dataKey": f"pie_{dk_base}",
            "height": 300,
            "rich": True,
            "pieLabels": _pie_segments(desc),
        }

    metrics = _detect_metrics(desc)
    x_key = infer_x_axis_key(desc, "barChart" if re.search(r"bar|pareto|comparison", line, re.I) else "lineChart")

    if re.search(r"bar chart|pareto|comparison|by\s+\w+", line, re.I):
        if not metrics:
            metrics = [("actual", "Actual"), ("benchmark", "Benchmark")]
        series = [
            {"dataKey": k, "name": n, "color": _CHART_COLORS[i % len(_CHART_COLORS)]}
            for i, (k, n) in enumerate(metrics[:4])
        ]
        return {
            "type": "barChart",
            "title": title,
            "dataKey": f"series_{dk_base}_bar",
            "xAxisKey": x_key,
            "series": series,
            "height": 300,
            "rich": True,
        }

    if not metrics:
        metrics = [("value", "Value"), ("target", "Target")]
    elif len(metrics) == 1:
        metrics.append(("target", "Target"))
    series = [
        {"dataKey": k, "name": n, "color": _CHART_COLORS[i % len(_CHART_COLORS)]}
        for i, (k, n) in enumerate(metrics[:5])
    ]
    return {
        "type": "lineChart",
        "title": title,
        "dataKey": f"series_{dk_base}_line",
        "xAxisKey": x_key,
        "series": series,
        "height": 300,
        "rich": True,
    }
