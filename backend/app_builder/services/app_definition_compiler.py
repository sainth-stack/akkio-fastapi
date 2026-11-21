"""Compile AppDefinition from plan_json + requirement (deterministic, no TS codegen)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app_builder.schemas.app_definition import (
    AppDefinition,
    AppMeta,
    AppTheme,
    BarChartBlock,
    ChartSeriesDef,
    DataTableBlock,
    KpiItemDef,
    KpiRowBlock,
    LineChartBlock,
    NavItemDef,
    PageDef,
    TableColumnDef,
)


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return s or "page"


def _blueprint_domain(blueprint: Any) -> str:
    if not isinstance(blueprint, dict):
        return ""
    return str(blueprint.get("selected_domain") or blueprint.get("domain") or "").lower()


def _is_inventory_intent(text: str) -> bool:
    return any(
        k in text
        for k in (
            "inventory",
            "supply chain",
            "sku",
            "warehouse",
            "purchase order",
            "materials / skus",
            "supplier portal",
            "stock replenishment",
        )
    )


def _is_monitoring_analytics_intent(text: str) -> bool:
    return any(
        k in text
        for k in (
            "monitoring",
            "analytics",
            "oee",
            "production",
            "plant",
            "factory",
            "machine",
            "work order",
            "quality control",
            "manufacturing",
        )
    )


def _theme_from_tokens(design_tokens: Optional[Dict[str, Any]]) -> AppTheme:
    if not design_tokens:
        return AppTheme()
    light = design_tokens.get("light") or {}
    if isinstance(light, dict) and not light.get("primary"):
        colors = design_tokens.get("colors") or {}
        if isinstance(colors, dict):
            light = {**colors, **light}
    palette = design_tokens.get("chart_palette") or design_tokens.get("chartPalette") or []
    return AppTheme(
        primary=light.get("primary", "#1976d2"),
        primaryDark=light.get("primary_dark", light.get("primaryDark", "#1565c0")),
        primaryLight=light.get("primary_light", "#42a5f5"),
        secondary=light.get("secondary", "#9c27b0"),
        background=light.get("background", "#f5f7fb"),
        surface=light.get("surface", "#ffffff"),
        text=light.get("text", "#0f172a"),
        muted=light.get("muted", "#64748b"),
        danger=light.get("danger", light.get("error", "#d32f2f")),
        success=light.get("success", "#2e7d32"),
        warning=light.get("warning", "#ed6c02"),
        info=light.get("info", "#0288d1"),
        fontFamily=(design_tokens.get("font_family") or "Inter").split(",")[0].strip("'\""),
        chartPalette=palette if isinstance(palette, list) and palette else AppTheme().chartPalette,
    )


def _default_inventory_app(app_name: str, domain: str) -> AppDefinition:
    """Sensible demo app for supply chain / inventory / generic SaaS prompts."""
    mock: Dict[str, Any] = {
        "kpis_dashboard": {
            "total_skus": "1,248",
            "inventory_value": "$4.2M",
            "low_stock_alerts": 17,
            "excess_stock": 6,
            "turnover": "6.4×",
            "supplier_otd": "94.2%",
        },
        "series_stock_trend": [
            {"name": "Jan", "consumption": 4200, "onHand": 12800},
            {"name": "Feb", "consumption": 3900, "onHand": 12100},
            {"name": "Mar", "consumption": 5100, "onHand": 11800},
            {"name": "Apr", "consumption": 4700, "onHand": 11200},
            {"name": "May", "consumption": 5300, "onHand": 10950},
            {"name": "Jun", "consumption": 4900, "onHand": 10400},
        ],
        "table_materials": [
            {"id": "SKU-1001", "name": "Steel Plate 4mm", "onHand": 420, "reorder": 200, "safety": 150, "status": "OK"},
            {"id": "SKU-2044", "name": "Hydraulic Seal Kit", "onHand": 38, "reorder": 80, "safety": 40, "status": "Low"},
            {"id": "SKU-3302", "name": "PCB Controller v3", "onHand": 12, "reorder": 50, "safety": 25, "status": "Critical"},
            {"id": "SKU-4410", "name": "Copper Wire 2.5mm", "onHand": 890, "reorder": 400, "safety": 200, "status": "OK"},
            {"id": "SKU-5521", "name": "Bearing 6205", "onHand": 156, "reorder": 100, "safety": 60, "status": "OK"},
        ],
        "table_suppliers": [
            {"id": "SUP-01", "name": "Apex Metals", "leadTimeDays": 12, "onTimePct": 96, "score": 4.6},
            {"id": "SUP-02", "name": "Nova Components", "leadTimeDays": 8, "onTimePct": 91, "score": 4.2},
            {"id": "SUP-03", "name": "Global Polymers", "leadTimeDays": 21, "onTimePct": 88, "score": 3.9},
        ],
        "table_orders": [
            {"id": "PO-8841", "supplier": "Apex Metals", "status": "In Transit", "lines": 14, "value": 48200},
            {"id": "PO-8836", "supplier": "Nova Components", "status": "Received", "lines": 8, "value": 22100},
            {"id": "PO-8829", "supplier": "Global Polymers", "status": "Draft", "lines": 22, "value": 91500},
        ],
        "table_alerts": [
            {"id": "AL-01", "type": "Low stock", "subject": "Hydraulic Seal Kit", "severity": "High", "when": "2h ago"},
            {"id": "AL-02", "type": "Delayed PO", "subject": "PO-8841", "severity": "Medium", "when": "5h ago"},
            {"id": "AL-03", "type": "Abnormal consumption", "subject": "PCB Controller v3", "severity": "High", "when": "1d ago"},
        ],
        "kpis_reports": {
            "total_anomalies": "142",
            "fill_rate": "97.1%",
            "avg_lead_time": "11.2d",
        },
        "table_report_detail": [
            {"material": "Steel Plate 4mm", "category": "Raw", "value": 182000, "turns": 8.2},
            {"material": "Hydraulic Seal Kit", "category": "MRO", "value": 42000, "turns": 5.1},
            {"material": "PCB Controller v3", "category": "Electronics", "value": 96000, "turns": 4.4},
        ],
    }

    nav = [
        NavItemDef(label="Dashboard", path="/", icon="Dashboard", pageId="dashboard"),
        NavItemDef(label="Materials", path="/materials", icon="Inventory", pageId="materials"),
        NavItemDef(label="Suppliers", path="/suppliers", icon="LocalShipping", pageId="suppliers"),
        NavItemDef(label="Purchase Orders", path="/orders", icon="ShoppingCart", pageId="orders"),
        NavItemDef(label="Alerts", path="/alerts", icon="NotificationsActive", pageId="alerts"),
        NavItemDef(label="Reports", path="/reports", icon="Assessment", pageId="reports"),
    ]

    pages = [
        PageDef(
            id="dashboard",
            title="Operations dashboard",
            subtitle="Real-time inventory health and consumption trends",
            blocks=[
                KpiRowBlock(
                    items=[
                        KpiItemDef(label="Total SKUs", valueKey="total_skus", color="primary"),
                        KpiItemDef(label="Inventory value", valueKey="inventory_value", color="success"),
                        KpiItemDef(label="Low-stock alerts", valueKey="low_stock_alerts", color="warning"),
                        KpiItemDef(label="Excess stock", valueKey="excess_stock", color="error"),
                        KpiItemDef(label="Turnover", valueKey="turnover", color="info"),
                        KpiItemDef(label="Supplier OTD", valueKey="supplier_otd", color="success"),
                    ],
                ).model_dump(),
                LineChartBlock(
                    title="Stock vs consumption",
                    dataKey="series_stock_trend",
                    xAxisKey="name",
                    series=[
                        ChartSeriesDef(dataKey="consumption", name="Consumption", color="#1976d2"),
                        ChartSeriesDef(dataKey="onHand", name="On hand", color="#2e7d32"),
                    ],
                ).model_dump(),
            ],
        ),
        PageDef(
            id="materials",
            title="Materials / SKUs",
            subtitle="Search, filter, and monitor stock status",
            blocks=[
                DataTableBlock(
                    title="Materials",
                    dataKey="table_materials",
                    columns=[
                        TableColumnDef(field="id", header="SKU"),
                        TableColumnDef(field="name", header="Name"),
                        TableColumnDef(field="onHand", header="On hand", align="right"),
                        TableColumnDef(field="reorder", header="Reorder pt.", align="right"),
                        TableColumnDef(field="safety", header="Safety", align="right"),
                        TableColumnDef(field="status", header="Status"),
                    ],
                ).model_dump(),
            ],
        ),
        PageDef(
            id="suppliers",
            title="Suppliers",
            blocks=[
                DataTableBlock(
                    dataKey="table_suppliers",
                    columns=[
                        TableColumnDef(field="name", header="Supplier"),
                        TableColumnDef(field="leadTimeDays", header="Lead time (d)", align="right"),
                        TableColumnDef(field="onTimePct", header="On-time %", align="right"),
                        TableColumnDef(field="score", header="Score", align="right"),
                    ],
                ).model_dump(),
            ],
        ),
        PageDef(
            id="orders",
            title="Purchase orders",
            blocks=[
                DataTableBlock(
                    dataKey="table_orders",
                    columns=[
                        TableColumnDef(field="id", header="PO"),
                        TableColumnDef(field="supplier", header="Supplier"),
                        TableColumnDef(field="status", header="Status"),
                        TableColumnDef(field="lines", header="Lines", align="right"),
                        TableColumnDef(field="value", header="Value", align="right"),
                    ],
                ).model_dump(),
            ],
        ),
        PageDef(
            id="alerts",
            title="Alerts",
            blocks=[
                DataTableBlock(
                    dataKey="table_alerts",
                    columns=[
                        TableColumnDef(field="type", header="Type"),
                        TableColumnDef(field="subject", header="Subject"),
                        TableColumnDef(field="severity", header="Severity"),
                        TableColumnDef(field="when", header="When"),
                    ],
                ).model_dump(),
            ],
        ),
        PageDef(
            id="reports",
            title="Reports",
            subtitle="Summary KPIs and material-level breakdown",
            blocks=[
                KpiRowBlock(
                    items=[
                        KpiItemDef(label="Events (30d)", valueKey="total_anomalies", color="primary"),
                        KpiItemDef(label="Fill rate", valueKey="fill_rate", color="success"),
                        KpiItemDef(label="Avg lead time", valueKey="avg_lead_time", color="info"),
                    ],
                ).model_dump(),
                DataTableBlock(
                    title="Detail by material",
                    dataKey="table_report_detail",
                    columns=[
                        TableColumnDef(field="material", header="Material"),
                        TableColumnDef(field="category", header="Category"),
                        TableColumnDef(field="value", header="Value", align="right"),
                        TableColumnDef(field="turns", header="Turns", align="right"),
                    ],
                ).model_dump(),
            ],
        ),
    ]

    return AppDefinition(
        meta=AppMeta(appName=app_name, description="Inventory and supply chain workspace", domain=domain),
        navigation=nav,
        pages=pages,
        mock=mock,
    )


def _manufacturing_monitoring_app(app_name: str) -> AppDefinition:
    """Dashboard + Analytics + Reports — matches Agentic Manufacturing domain prompts."""
    mock: Dict[str, Any] = {
        "kpis_dashboard": {
            "oee": "87.4%",
            "throughput": "1,240 u/h",
            "downtime": "42 min",
            "active_alerts": 5,
            "scrap_rate": "0.8%",
            "energy_per_unit": "12.4 kWh",
        },
        "series_oee_trend": [
            {"shift": "Mon A", "oee": 84.2, "target": 85},
            {"shift": "Mon B", "oee": 86.1, "target": 85},
            {"shift": "Tue A", "oee": 88.4, "target": 85},
            {"shift": "Tue B", "oee": 87.0, "target": 85},
            {"shift": "Wed A", "oee": 89.2, "target": 85},
            {"shift": "Wed B", "oee": 86.8, "target": 85},
        ],
        "kpis_analytics": {
            "mtbf": "128 h",
            "mttr": "18 min",
            "first_pass_yield": "96.2%",
            "planned_stops": 3,
        },
        "series_output_defects": [
            {"week": "W1", "output": 8200, "defects": 62},
            {"week": "W2", "output": 8450, "defects": 58},
            {"week": "W3", "output": 8100, "defects": 71},
            {"week": "W4", "output": 8720, "defects": 49},
            {"week": "W5", "output": 8890, "defects": 44},
            {"week": "W6", "output": 9010, "defects": 41},
        ],
        "table_machines": [
            {"id": "M-01", "name": "CNC Cell A", "status": "Running", "oee": "91.2%", "alerts": 0},
            {"id": "M-02", "name": "Press Line 2", "status": "Idle", "oee": "78.4%", "alerts": 1},
            {"id": "M-03", "name": "Assembly Robot 4", "status": "Running", "oee": "88.9%", "alerts": 0},
            {"id": "M-04", "name": "Paint Booth", "status": "Maintenance", "oee": "—", "alerts": 2},
        ],
        "table_quality_events": [
            {"id": "Q-101", "line": "Press Line 2", "defect": "Surface scratch", "severity": "Medium", "when": "3h ago"},
            {"id": "Q-102", "line": "CNC Cell A", "defect": "Tolerance drift", "severity": "High", "when": "5h ago"},
            {"id": "Q-103", "line": "Assembly Robot 4", "defect": "Torque out of spec", "severity": "Low", "when": "1d ago"},
        ],
        "kpis_reports": {
            "reports_generated": "24",
            "on_time_delivery": "94.8%",
            "cost_per_unit": "$18.42",
        },
        "table_report_runs": [
            {"id": "R-2406", "name": "Weekly OEE summary", "period": "Jun W2", "owner": "Ops", "status": "Published"},
            {"id": "R-2405", "name": "Scrap pareto", "period": "Jun W1", "owner": "Quality", "status": "Published"},
            {"id": "R-2404", "name": "Energy intensity", "period": "May", "owner": "Sustainability", "status": "Draft"},
        ],
    }

    nav = [
        NavItemDef(label="Dashboard", path="/", icon="Dashboard", pageId="dashboard"),
        NavItemDef(label="Analytics", path="/analytics", icon="ShowChart", pageId="analytics"),
        NavItemDef(label="Reports", path="/reports", icon="Assessment", pageId="reports"),
    ]

    pages = [
        PageDef(
            id="dashboard",
            title="Plant dashboard",
            subtitle="Live OEE, throughput, and alert summary",
            blocks=[
                KpiRowBlock(
                    items=[
                        KpiItemDef(label="OEE", valueKey="oee", color="primary"),
                        KpiItemDef(label="Throughput", valueKey="throughput", color="success"),
                        KpiItemDef(label="Downtime (today)", valueKey="downtime", color="warning"),
                        KpiItemDef(label="Active alerts", valueKey="active_alerts", color="error"),
                        KpiItemDef(label="Scrap rate", valueKey="scrap_rate", color="warning"),
                        KpiItemDef(label="Energy / unit", valueKey="energy_per_unit", color="info"),
                    ],
                ).model_dump(),
                LineChartBlock(
                    title="OEE by shift",
                    dataKey="series_oee_trend",
                    xAxisKey="shift",
                    series=[
                        ChartSeriesDef(dataKey="oee", name="OEE %", color="#1976d2"),
                        ChartSeriesDef(dataKey="target", name="Target", color="#94a3b8"),
                    ],
                ).model_dump(),
            ],
        ),
        PageDef(
            id="analytics",
            title="Production analytics",
            subtitle="Output, defects, and machine-level performance",
            blocks=[
                KpiRowBlock(
                    items=[
                        KpiItemDef(label="MTBF", valueKey="mtbf", color="primary"),
                        KpiItemDef(label="MTTR", valueKey="mttr", color="info"),
                        KpiItemDef(label="First-pass yield", valueKey="first_pass_yield", color="success"),
                        KpiItemDef(label="Planned stops", valueKey="planned_stops", color="warning"),
                    ],
                ).model_dump(),
                BarChartBlock(
                    title="Weekly output vs defects",
                    dataKey="series_output_defects",
                    xAxisKey="week",
                    series=[
                        ChartSeriesDef(dataKey="output", name="Output (units)", color="#1976d2"),
                        ChartSeriesDef(dataKey="defects", name="Defects", color="#d32f2f"),
                    ],
                ).model_dump(),
                DataTableBlock(
                    title="Machine status",
                    dataKey="table_machines",
                    columns=[
                        TableColumnDef(field="id", header="ID"),
                        TableColumnDef(field="name", header="Machine"),
                        TableColumnDef(field="status", header="Status"),
                        TableColumnDef(field="oee", header="OEE", align="right"),
                        TableColumnDef(field="alerts", header="Alerts", align="right"),
                    ],
                ).model_dump(),
                DataTableBlock(
                    title="Recent quality events",
                    dataKey="table_quality_events",
                    columns=[
                        TableColumnDef(field="line", header="Line"),
                        TableColumnDef(field="defect", header="Defect"),
                        TableColumnDef(field="severity", header="Severity"),
                        TableColumnDef(field="when", header="When"),
                    ],
                ).model_dump(),
            ],
        ),
        PageDef(
            id="reports",
            title="Reports",
            subtitle="Published operational and quality reports",
            blocks=[
                KpiRowBlock(
                    items=[
                        KpiItemDef(label="Reports (30d)", valueKey="reports_generated", color="primary"),
                        KpiItemDef(label="On-time delivery", valueKey="on_time_delivery", color="success"),
                        KpiItemDef(label="Cost / unit", valueKey="cost_per_unit", color="info"),
                    ],
                ).model_dump(),
                DataTableBlock(
                    title="Report library",
                    dataKey="table_report_runs",
                    columns=[
                        TableColumnDef(field="name", header="Report"),
                        TableColumnDef(field="period", header="Period"),
                        TableColumnDef(field="owner", header="Owner"),
                        TableColumnDef(field="status", header="Status"),
                    ],
                ).model_dump(),
            ],
        ),
    ]

    return AppDefinition(
        meta=AppMeta(
            appName=app_name,
            description="Manufacturing monitoring — dashboard, analytics, and reports",
            domain="manufacturing",
        ),
        navigation=nav,
        pages=pages,
        mock=mock,
    )


def _transformer_demo_app(app_name: str) -> AppDefinition:
    mock = {
        "kpis_main": {
            "load_mva": "42.8",
            "efficiency": "98.2%",
            "oil_temp": "62°C",
            "anomalies": 3,
        },
        "series_load": [
            {"hour": "00:00", "load": 38.2},
            {"hour": "04:00", "load": 35.1},
            {"hour": "08:00", "load": 44.5},
            {"hour": "12:00", "load": 48.9},
            {"hour": "16:00", "load": 46.2},
            {"hour": "20:00", "load": 41.0},
        ],
        "table_events": [
            {"id": "E-01", "metric": "Oil temperature", "value": "78°C", "severity": "High"},
            {"id": "E-02", "metric": "Tap position", "value": "14", "severity": "Medium"},
        ],
    }
    nav = [
        NavItemDef(label="Dashboard", path="/", icon="Dashboard", pageId="dashboard"),
        NavItemDef(label="Reports", path="/reports", icon="Assessment", pageId="reports"),
    ]
    pages = [
        PageDef(
            id="dashboard",
            title="Transformer monitoring",
            subtitle="KPIs and load profile",
            blocks=[
                KpiRowBlock(
                    items=[
                        KpiItemDef(label="Load (MVA)", valueKey="load_mva", color="primary"),
                        KpiItemDef(label="Efficiency", valueKey="efficiency", color="success"),
                        KpiItemDef(label="Oil temp", valueKey="oil_temp", color="warning"),
                        KpiItemDef(label="Anomalies", valueKey="anomalies", color="error"),
                    ],
                ).model_dump(),
                LineChartBlock(
                    title="Load profile (24h)",
                    dataKey="series_load",
                    xAxisKey="hour",
                    series=[ChartSeriesDef(dataKey="load", name="Load MVA")],
                ).model_dump(),
            ],
        ),
        PageDef(
            id="reports",
            title="Reports",
            blocks=[
                DataTableBlock(
                    dataKey="table_events",
                    columns=[
                        TableColumnDef(field="id", header="ID"),
                        TableColumnDef(field="metric", header="Metric"),
                        TableColumnDef(field="value", header="Value"),
                        TableColumnDef(field="severity", header="Severity"),
                    ],
                ).model_dump(),
            ],
        ),
    ]
    return AppDefinition(
        meta=AppMeta(appName=app_name, description="Transformer KPIs and reporting", domain="analytics_dashboard"),
        navigation=nav,
        pages=pages,
        mock=mock,
    )


def compile_app_definition(
    requirement: str = "",
    prd: str = "",
    uiux: str = "",
    plan_json: Optional[Dict[str, Any]] = None,
    app_name: Optional[str] = None,
) -> AppDefinition:
    """Build AppDefinition without LLM (structure + demo mock)."""
    plan_json = plan_json or {}
    blueprint = plan_json.get("blueprint") or {}
    if isinstance(blueprint, str):
        blueprint = {}
    design_tokens = plan_json.get("design_tokens")

    selected_domain = _blueprint_domain(blueprint)
    domain = selected_domain or (
        (blueprint.get("domain") or "generic") if isinstance(blueprint, dict) else "generic"
    )
    combined = "\n".join([requirement or "", prd or "", uiux or ""]).lower()

    name = app_name or (blueprint.get("app_name") if isinstance(blueprint, dict) else None) or "Workspace"
    if len(name) > 80:
        name = name[:80]

    if "transformer" in combined or "substation" in combined:
        app = _transformer_demo_app(name)
    elif selected_domain == "manufacturing":
        app = _manufacturing_monitoring_app(name)
    elif selected_domain == "retail" and _is_inventory_intent(combined):
        app = _default_inventory_app(name, "retail")
    elif _is_inventory_intent(combined) and not _is_monitoring_analytics_intent(combined):
        app = _default_inventory_app(name, domain)
    elif _is_monitoring_analytics_intent(combined):
        app = _manufacturing_monitoring_app(name)
    else:
        app = _manufacturing_monitoring_app(name) if selected_domain == "manufacturing" else _default_inventory_app(
            name, domain
        )

    app.theme = _theme_from_tokens(design_tokens if isinstance(design_tokens, dict) else None)

    return app
