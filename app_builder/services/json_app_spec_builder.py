"""Build AppSpec v1 from planning artifacts (no TS/React codegen)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app_builder.schemas.json_app_spec import (
    AppSpecV1,
    BrandingSpec,
    ChartSeriesSpec,
    ChartSpec,
    EntityFieldSpec,
    EntitySpec,
    KpiSpec,
    NavItemSpec,
    ScreenSpec,
    TableColumnSpec,
    ThemeSpec,
    validate_app_spec,
)
from app_builder.services.fullstack_frontend_generator import theme_from_tokens


def _slug(s: str) -> str:
    x = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return x or "screen"


def _title_from_requirement(requirement: str, fallback: str = "Application") -> str:
    line = (requirement or "").strip().split("\n")[0].strip()
    if len(line) > 8 and len(line) < 100:
        return line[:90]
    return fallback


def _default_inventory_spec(app_name: str, theme: ThemeSpec) -> AppSpecV1:
    materials = [
        {"id": "M-1001", "sku": "SKU-8842", "name": "Transformer Core Steel", "onHand": 420, "reorder": 200, "safety": 150, "value": 84000, "status": "OK"},
        {"id": "M-1002", "sku": "SKU-3310", "name": "HV Bushing Assembly", "onHand": 38, "reorder": 60, "safety": 40, "value": 22800, "status": "Low"},
        {"id": "M-1003", "sku": "SKU-1199", "name": "Insulating Oil (IBC)", "onHand": 12, "reorder": 25, "safety": 20, "value": 9600, "status": "Critical"},
        {"id": "M-1004", "sku": "SKU-5501", "name": "Copper Winding Wire", "onHand": 890, "reorder": 400, "safety": 300, "value": 133500, "status": "OK"},
        {"id": "M-1005", "sku": "SKU-7702", "name": "Cooling Fan Module", "onHand": 145, "reorder": 80, "safety": 50, "value": 21750, "status": "OK"},
    ]
    suppliers = [
        {"id": "S-01", "name": "Apex Magnetics Ltd", "leadDays": 14, "onTimePct": 96, "score": 4.7},
        {"id": "S-02", "name": "Grid Components Co", "leadDays": 21, "onTimePct": 88, "score": 4.2},
        {"id": "S-03", "name": "Voltex Insulation", "leadDays": 10, "onTimePct": 92, "score": 4.5},
    ]
    pos = [
        {"id": "PO-2401", "supplier": "Apex Magnetics Ltd", "status": "In Transit", "lines": 3, "total": 45200},
        {"id": "PO-2402", "supplier": "Voltex Insulation", "status": "Draft", "lines": 1, "total": 12800},
        {"id": "PO-2398", "supplier": "Grid Components Co", "status": "Received", "lines": 5, "total": 91000},
    ]
    trend = [
        {"label": "Jan", "consumption": 420, "stock": 3800},
        {"label": "Feb", "consumption": 390, "stock": 3650},
        {"label": "Mar", "consumption": 450, "stock": 3520},
        {"label": "Apr", "consumption": 410, "stock": 3480},
        {"label": "May", "consumption": 480, "stock": 3310},
        {"label": "Jun", "consumption": 465, "stock": 3180},
    ]
    nav = [
        NavItemSpec(id="dashboard", label="Dashboard", icon="Dashboard", route="/", screenId="dashboard"),
        NavItemSpec(id="materials", label="Materials", icon="Inventory", route="/materials", screenId="materials"),
        NavItemSpec(id="suppliers", label="Suppliers", icon="LocalShipping", route="/suppliers", screenId="suppliers"),
        NavItemSpec(id="orders", label="Purchase Orders", icon="ShoppingCart", route="/orders", screenId="orders"),
        NavItemSpec(id="alerts", label="Alerts", icon="WarningAmber", route="/alerts", screenId="alerts"),
        NavItemSpec(id="reports", label="Reports", icon="Assessment", route="/reports", screenId="reports"),
    ]
    screens = [
        ScreenSpec(
            id="dashboard",
            type="dashboard",
            title=f"{app_name} — Operations",
            subtitle="Real-time inventory and supply health",
            kpis=[
                KpiSpec(id="skus", label="Total SKUs", value=128, tone="primary", icon="Inventory"),
                KpiSpec(id="value", label="Inventory value", value="$1.24M", tone="success", icon="AttachMoney"),
                KpiSpec(id="low", label="Low / critical SKUs", value=7, tone="warning", icon="WarningAmber"),
                KpiSpec(id="ontime", label="Supplier on-time %", value="91.2%", tone="primary", icon="LocalShipping"),
            ],
            charts=[
                ChartSpec(
                    id="trend",
                    title="Consumption vs on-hand trend",
                    type="line",
                    xAxisKey="label",
                    series=[
                        ChartSeriesSpec(dataKey="consumption", name="Consumption", color=theme.primary),
                        ChartSeriesSpec(dataKey="stock", name="On-hand units", color=theme.secondary),
                    ],
                    data=trend,
                ),
            ],
        ),
        ScreenSpec(
            id="materials",
            type="table",
            title="Materials / SKUs",
            entityId="material",
            columns=[
                TableColumnSpec(key="sku", label="SKU"),
                TableColumnSpec(key="name", label="Material"),
                TableColumnSpec(key="onHand", label="On hand", align="right"),
                TableColumnSpec(key="reorder", label="Reorder pt", align="right"),
                TableColumnSpec(key="status", label="Status"),
            ],
            rows=materials,
        ),
        ScreenSpec(
            id="suppliers",
            type="table",
            title="Suppliers",
            entityId="supplier",
            columns=[
                TableColumnSpec(key="name", label="Supplier"),
                TableColumnSpec(key="leadDays", label="Lead time (days)", align="right"),
                TableColumnSpec(key="onTimePct", label="On-time %", align="right"),
                TableColumnSpec(key="score", label="Score", align="right"),
            ],
            rows=suppliers,
        ),
        ScreenSpec(
            id="orders",
            type="table",
            title="Purchase orders",
            entityId="purchase_order",
            columns=[
                TableColumnSpec(key="id", label="PO #"),
                TableColumnSpec(key="supplier", label="Supplier"),
                TableColumnSpec(key="status", label="Status"),
                TableColumnSpec(key="lines", label="Lines", align="right"),
                TableColumnSpec(key="total", label="Total ($)", align="right"),
            ],
            rows=pos,
        ),
        ScreenSpec(
            id="alerts",
            type="alerts",
            title="Alerts",
            alerts=[
                {"severity": "Critical", "title": "SKU-1199 below safety stock", "detail": "Insulating oil at 12 units (safety 20)"},
                {"severity": "High", "title": "PO-2401 delayed", "detail": "In transit 3 days past ETA"},
                {"severity": "Medium", "title": "Abnormal consumption", "detail": "Copper wire usage +18% vs 30d avg"},
            ],
        ),
        ScreenSpec(
            id="reports",
            type="reports",
            title="Inventory reports",
            reportSummary=[
                KpiSpec(id="turnover", label="Inventory turnover", value="4.2x", tone="primary"),
                KpiSpec(id="fill", label="Fill rate", value="96.4%", tone="success"),
                KpiSpec(id="excess", label="Excess stock SKUs", value=5, tone="warning"),
            ],
            columns=[
                TableColumnSpec(key="sku", label="SKU"),
                TableColumnSpec(key="name", label="Material"),
                TableColumnSpec(key="onHand", label="On hand", align="right"),
                TableColumnSpec(key="value", label="Value ($)", align="right"),
            ],
            rows=materials,
        ),
    ]
    entities = [
        EntitySpec(id="material", label="Material", fields=[
            EntityFieldSpec(key="sku", label="SKU"), EntityFieldSpec(key="name", label="Name"),
            EntityFieldSpec(key="onHand", label="On hand", type="number"),
        ]),
        EntitySpec(id="supplier", label="Supplier", fields=[EntityFieldSpec(key="name", label="Name")]),
        EntitySpec(id="purchase_order", label="Purchase Order", fields=[EntityFieldSpec(key="status", label="Status")]),
    ]
    return AppSpecV1(
        meta={"domain": "inventory", "profile": "supply_chain"},
        branding=BrandingSpec(appName=app_name, headerTitle="Supply chain control tower", tagline=""),
        theme=theme,
        navigation=nav,
        screens=screens,
        entities=entities,
    )


def _default_analytics_spec(app_name: str, theme: ThemeSpec) -> AppSpecV1:
    trend = [{"label": f"W{i+1}", "output": 82 + i * 2, "defects": max(1, 5 - i // 2)} for i in range(12)]
    nav = [
        NavItemSpec(id="dashboard", label="Dashboard", icon="Dashboard", route="/", screenId="dashboard"),
        NavItemSpec(id="reports", label="Reports", icon="Assessment", route="/reports", screenId="reports"),
    ]
    screens = [
        ScreenSpec(
            id="dashboard",
            type="dashboard",
            title=app_name,
            subtitle="KPI overview and trends",
            kpis=[
                KpiSpec(id="k1", label="OEE", value="87.4%", tone="success", icon="Speed"),
                KpiSpec(id="k2", label="Throughput", value="1,240 units", tone="primary", icon="Timeline"),
                KpiSpec(id="k3", label="Defect rate", value="0.8%", tone="warning", icon="BugReport"),
                KpiSpec(id="k4", label="Downtime", value="42 min", tone="error", icon="WarningAmber"),
            ],
            charts=[
                ChartSpec(
                    id="output",
                    title="Weekly output vs defects",
                    type="bar",
                    xAxisKey="label",
                    series=[
                        ChartSeriesSpec(dataKey="output", name="Output", color=theme.primary),
                        ChartSeriesSpec(dataKey="defects", name="Defects", color="#C62828"),
                    ],
                    data=trend,
                ),
            ],
        ),
        ScreenSpec(
            id="reports",
            type="reports",
            title="Reports",
            reportSummary=[
                KpiSpec(id="r1", label="MTBF", value="128 h", tone="primary"),
                KpiSpec(id="r2", label="Energy / unit", value="12.4 kWh", tone="success"),
            ],
            columns=[
                TableColumnSpec(key="label", label="Period"),
                TableColumnSpec(key="output", label="Output", align="right"),
                TableColumnSpec(key="defects", label="Defects", align="right"),
            ],
            rows=trend,
        ),
    ]
    return AppSpecV1(
        meta={"domain": "analytics_dashboard"},
        branding=BrandingSpec(appName=app_name, headerTitle="Performance hub"),
        theme=theme,
        navigation=nav,
        screens=screens,
        entities=[],
    )


def _screens_from_architecture(architecture: Optional[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    if not architecture:
        return names
    for s in architecture.get("screens") or []:
        if isinstance(s, dict) and s.get("name"):
            names.append(str(s["name"]))
        elif isinstance(s, str):
            names.append(s)
    fs = architecture.get("frontend_structure") or {}
    for p in fs.get("pages") or []:
        if isinstance(p, dict):
            n = p.get("name") or p.get("component")
            if n:
                names.append(str(n))
    return names


def build_app_spec_v1(
    *,
    requirement: str,
    prd: str = "",
    uiux: str = "",
    architecture: Optional[Dict[str, Any]] = None,
    design_tokens: Optional[Dict[str, Any]] = None,
    app_name: Optional[str] = None,
    domain: str = "generic",
    app_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return validated AppSpec v1 dict."""
    colors = theme_from_tokens(design_tokens, uiux=uiux)
    theme = ThemeSpec(
        primary=colors.get("primary", "#1565C0"),
        primaryDark=colors.get("primary_dark", "#0D47A1"),
        secondary=colors.get("secondary", "#7B1FA2"),
        background=colors.get("background", "#F5F5F5"),
        surface=colors.get("surface", "#FFFFFF"),
        text=colors.get("text", "#212121"),
        muted=colors.get("muted", "#757575"),
    )
    name = app_name or _title_from_requirement(requirement, "Agentic Application")
    combined = "\n".join([requirement, prd, uiux]).lower()

    if domain in ("inventory",) or any(k in combined for k in ("inventory", "supply chain", "sku", "warehouse", "purchase order")):
        spec = _default_inventory_spec(name, theme)
    elif domain in ("analytics_dashboard",) or any(k in combined for k in ("transformer", "kpi", "dashboard", "report")):
        spec = _default_analytics_spec(name, theme)
    else:
        spec = _default_analytics_spec(name, theme)

    spec.branding.appName = name
    if uiux:
        first = re.sub(r"#+\s*", "", uiux.strip().split("\n")[0])[:120]
        if first:
            spec.branding.tagline = first

    arch_screens = _screens_from_architecture(architecture)
    if arch_screens:
        spec.meta["plannedScreens"] = arch_screens

    spec.meta["appId"] = str(app_id) if app_id else None
    spec.meta["domain"] = domain
    return validate_app_spec(spec.to_dict()).to_dict()
