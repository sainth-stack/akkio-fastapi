"""Page-specific block layouts — each screen gets distinct charts, tables, and KPIs."""
from __future__ import annotations

from typing import Any, Dict, List

from app_builder.services.page_mock_catalog import infer_page_profile


def _sid(screen: Dict[str, str]) -> str:
    return screen.get("id") or "page"


def _title(screen: Dict[str, str]) -> str:
    return screen.get("title") or "Page"


def blocks_for_screen(
    screen: Dict[str, str],
    index: int,
    total: int,
    *,
    domain: str = "",
    meta_description: str = "",
) -> List[Dict[str, Any]]:
    sid = _sid(screen)
    title = _title(screen)
    ctx = f"{title} {screen.get('purpose') or ''}".lower()
    profile = infer_page_profile(sid, title, "", meta_description, domain)

    if profile == "sessions" or (sid in ("sessions", "session") or (ctx.strip() == "sessions")):
        return _sessions_blocks(sid, title)
    if profile == "reports" or "report" in ctx:
        return _reports_blocks(sid, title)
    if profile == "analytics" or "analytic" in ctx:
        return _analytics_blocks(sid, title)
    if index == 0 or profile in ("dashboard", "manufacturing", "transformer_monitoring") or "dashboard" in ctx:
        return _dashboard_blocks(sid, title, profile)

    if any(k in ctx for k in ("list", "table", "catalog", "inventory", "contact", "deal", "ticket")):
        return [_records_table(sid, title)]

    return [{"type": "text", "content": screen.get("purpose") or title}]


def _kpi_row(sid: str, items: List[Dict[str, str]]) -> Dict[str, Any]:
    return {"type": "kpiRow", "dataKey": f"kpis_{sid}", "items": items}


def _dashboard_blocks(sid: str, title: str, profile: str) -> List[Dict[str, Any]]:
    if profile == "transformer_monitoring":
        kpis = [
            {"label": "Load (MVA)", "valueKey": "load_mva", "color": "primary"},
            {"label": "Oil temperature", "valueKey": "oil_temp", "color": "warning"},
            {"label": "Efficiency", "valueKey": "efficiency", "color": "success"},
            {"label": "Active alarms", "valueKey": "active_alarms", "color": "error"},
            {"label": "Tap position", "valueKey": "tap_position", "color": "info"},
            {"label": "Partial discharge", "valueKey": "pd_level", "color": "default"},
        ]
        trend_series = [
            {"dataKey": "load_mva", "name": "Load (MVA)", "color": "#1976d2"},
            {"dataKey": "target", "name": "Rated load", "color": "#94a3b8"},
        ]
        bar_series = [
            {"dataKey": "oil_temp", "name": "Oil temp °C", "color": "#ed6c02"},
            {"dataKey": "ambient", "name": "Ambient °C", "color": "#0288d1"},
        ]
        alert_title = "Active transformer alarms"
    else:
        kpis = [
            {"label": "OEE", "valueKey": "oee", "color": "primary"},
            {"label": "Throughput", "valueKey": "throughput", "color": "success"},
            {"label": "Downtime (today)", "valueKey": "downtime", "color": "warning"},
            {"label": "Active alerts", "valueKey": "active_alerts", "color": "error"},
            {"label": "Scrap rate", "valueKey": "scrap_rate", "color": "info"},
            {"label": "Energy / unit", "valueKey": "energy_per_unit", "color": "default"},
        ]
        trend_series = [
            {"dataKey": "oee", "name": "OEE %", "color": "#1976d2"},
            {"dataKey": "target", "name": "Target", "color": "#94a3b8"},
        ]
        bar_series = [
            {"dataKey": "output", "name": "Actual output", "color": "#1976d2"},
            {"dataKey": "planned", "name": "Planned", "color": "#9c27b0"},
        ]
        alert_title = "Plant floor alerts"

    return [
        _kpi_row(sid, kpis),
        {
            "type": "lineChart",
            "title": f"{title} — performance trend",
            "dataKey": f"series_{sid}_performance",
            "xAxisKey": "period",
            "series": trend_series,
            "height": 320,
        },
        {
            "type": "barChart",
            "title": f"{title} — shift comparison",
            "dataKey": f"series_{sid}_shift",
            "xAxisKey": "period",
            "series": bar_series,
            "height": 300,
        },
        {
            "type": "pieChart",
            "title": f"{title} — alert distribution",
            "dataKey": f"pie_{sid}_alerts",
            "height": 280,
        },
        {
            "type": "dataTable",
            "title": alert_title,
            "dataKey": f"table_{sid}_alerts",
            "columns": [
                {"field": "id", "header": "ID"},
                {"field": "unit", "header": "Unit / line"},
                {"field": "severity", "header": "Severity"},
                {"field": "type", "header": "Type"},
                {"field": "message", "header": "Message"},
                {"field": "occurred_at", "header": "Occurred"},
            ],
            "searchable": True,
        },
    ]


def _sessions_blocks(sid: str, title: str) -> List[Dict[str, Any]]:
    return [
        _kpi_row(
            sid,
            [
                {"label": "Active sessions", "valueKey": "active_sessions", "color": "primary"},
                {"label": "Avg duration", "valueKey": "avg_duration", "color": "success"},
                {"label": "Units online", "valueKey": "units_online", "color": "info"},
                {"label": "Exceptions", "valueKey": "exceptions", "color": "error"},
            ],
        ),
        {
            "type": "lineChart",
            "title": f"{title} — utilization (24h)",
            "dataKey": f"series_{sid}_utilization",
            "xAxisKey": "period",
            "series": [
                {"dataKey": "utilization", "name": "Utilization %", "color": "#1976d2"},
                {"dataKey": "capacity", "name": "Capacity %", "color": "#94a3b8"},
            ],
            "height": 300,
        },
        {
            "type": "dataTable",
            "title": "Operator sessions",
            "dataKey": f"table_{sid}",
            "columns": [
                {"field": "id", "header": "Session"},
                {"field": "name", "header": "Unit"},
                {"field": "operator", "header": "Operator"},
                {"field": "duration", "header": "Duration"},
                {"field": "status", "header": "Status"},
                {"field": "started_at", "header": "Started"},
            ],
            "searchable": True,
        },
    ]


def _analytics_blocks(sid: str, title: str) -> List[Dict[str, Any]]:
    return [
        _kpi_row(
            sid,
            [
                {"label": "MTBF", "valueKey": "mtbf", "color": "primary"},
                {"label": "First-pass yield", "valueKey": "first_pass_yield", "color": "success"},
                {"label": "Scrap rate", "valueKey": "scrap_rate", "color": "warning"},
                {"label": "Variance", "valueKey": "variance", "color": "info"},
            ],
        ),
        {
            "type": "barChart",
            "title": f"{title} — output vs defects",
            "dataKey": f"series_{sid}_quality",
            "xAxisKey": "period",
            "series": [
                {"dataKey": "output", "name": "Output (units)", "color": "#1976d2"},
                {"dataKey": "defects", "name": "Defects", "color": "#d32f2f"},
            ],
            "height": 300,
        },
        {
            "type": "lineChart",
            "title": f"{title} — yield trend",
            "dataKey": f"series_{sid}_yield",
            "xAxisKey": "period",
            "series": [
                {"dataKey": "yield_pct", "name": "Yield %", "color": "#2e7d32"},
                {"dataKey": "target", "name": "Target", "color": "#94a3b8"},
            ],
            "height": 280,
        },
        {
            "type": "dataTable",
            "title": "Metric breakdown",
            "dataKey": f"table_{sid}",
            "columns": [
                {"field": "id", "header": "ID"},
                {"field": "name", "header": "Metric"},
                {"field": "value", "header": "Value", "align": "right"},
                {"field": "trend", "header": "Trend"},
                {"field": "status", "header": "Status"},
            ],
            "searchable": True,
        },
    ]


def _reports_blocks(sid: str, title: str) -> List[Dict[str, Any]]:
    return [
        _kpi_row(
            sid,
            [
                {"label": "Published (30d)", "valueKey": "reports_published", "color": "primary"},
                {"label": "On-time delivery", "valueKey": "on_time_delivery", "color": "success"},
                {"label": "Open drafts", "valueKey": "open_drafts", "color": "warning"},
            ],
        ),
        {
            "type": "barChart",
            "title": f"{title} — delivery by owner",
            "dataKey": f"series_{sid}_delivery",
            "xAxisKey": "period",
            "series": [
                {"dataKey": "published", "name": "Published", "color": "#1976d2"},
                {"dataKey": "draft", "name": "Draft", "color": "#ed6c02"},
            ],
            "height": 280,
        },
        {
            "type": "dataTable",
            "title": "Report library",
            "dataKey": f"table_{sid}",
            "columns": [
                {"field": "id", "header": "ID"},
                {"field": "name", "header": "Report"},
                {"field": "period", "header": "Period"},
                {"field": "owner", "header": "Owner"},
                {"field": "status", "header": "Status"},
            ],
            "searchable": True,
        },
    ]


def _records_table(sid: str, title: str) -> Dict[str, Any]:
    return {
        "type": "dataTable",
        "title": title,
        "dataKey": f"table_{sid}",
        "columns": [
            {"field": "id", "header": "ID"},
            {"field": "name", "header": "Name"},
            {"field": "status", "header": "Status"},
            {"field": "updated_at", "header": "Updated"},
        ],
        "searchable": True,
    }
