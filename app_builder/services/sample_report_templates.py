"""
Domain-aware sample report templates for App Builder UI and generated app Reports pages.
"""
from __future__ import annotations

import datetime
import json
from typing import Any, Dict, List, Optional, Tuple

from app_builder.services.demo_api_data import (
    anomaly_anomalies_list,
    anomaly_dashboard_payload,
    anomaly_reports_summary,
    generic_app_dashboard,
    project_looks_like_anomaly_from_files,
)
from app_builder.services.supply_chain_demo_data import (
    looks_like_supply_chain,
    supply_chain_seed_data,
)

TemplateMeta = Dict[str, str]


def detect_report_domain(
    requirement: str = "",
    prd: str = "",
    files: Optional[Dict[str, str]] = None,
    seed: Optional[Dict[str, Any]] = None,
) -> str:
    seed = seed or {}
    if looks_like_supply_chain(requirement, prd, files):
        return "supply_chain"
    if project_looks_like_anomaly_from_files(files or {}, requirement=requirement, prd=prd):
        return "anomaly"
    if seed.get("dashboard") and isinstance(seed["dashboard"], dict) and seed["dashboard"].get("metrics"):
        return "anomaly"
    if seed.get("materials") or (isinstance(seed.get("dashboard"), dict) and "total_inventory_value" in seed["dashboard"]):
        return "supply_chain"
    try:
        from app_builder.services.fullstack_ecommerce_generator import is_ecommerce_domain

        if is_ecommerce_domain(requirement, prd):
            return "ecommerce"
    except Exception:
        pass
    text = f"{requirement} {prd}".lower()
    analytics_kw = (
        "sales ai", "sales pipeline", "revenue forecast", "crm", "lead scoring",
        "analytics dashboard", "kpi dashboard", "business intelligence", "forecast",
    )
    if any(k in text for k in analytics_kw):
        return "analytics"
    if seed.get("products") or seed.get("orders"):
        return "ecommerce"
    return "generic"


def template_catalog(domain: str) -> List[TemplateMeta]:
    catalogs: Dict[str, List[TemplateMeta]] = {
        "supply_chain": [
            {"id": "inventory_operations_summary_v1", "title": "Inventory Operations Summary", "subtitle": "KPIs, aging buckets, and category turnover"},
            {"id": "inventory_aging_excess_v1", "title": "Inventory Aging & Excess", "subtitle": "Value by age bucket and excess exposure"},
            {"id": "inventory_value_turnover_v1", "title": "Value & Turnover", "subtitle": "Inventory value and turns by material category"},
        ],
        "anomaly": [
            {"id": "anomaly_operations_summary_v1", "title": "Anomaly Operations Summary", "subtitle": "Detections, MTTD, and severity mix"},
            {"id": "anomaly_metric_breakdown_v1", "title": "Metric Breakdown", "subtitle": "Top contributing signals and daily trend"},
        ],
        "ecommerce": [
            {"id": "ecommerce_sales_summary_v1", "title": "Sales & Orders Summary", "subtitle": "Revenue, conversion, and order funnel"},
            {"id": "ecommerce_product_performance_v1", "title": "Product Performance", "subtitle": "Top SKUs and category mix"},
        ],
        "analytics": [
            {"id": "analytics_pipeline_summary_v1", "title": "Pipeline Summary", "subtitle": "Stages, win rate, and forecast"},
            {"id": "analytics_revenue_forecast_v1", "title": "Revenue Forecast", "subtitle": "Monthly trend and growth drivers"},
        ],
        "generic": [
            {"id": "operations_summary_v1", "title": "Operations Summary", "subtitle": "Core KPIs and activity snapshot"},
            {"id": "activity_snapshot_v1", "title": "Activity Snapshot", "subtitle": "Recent items and status breakdown"},
        ],
    }
    return catalogs.get(domain, catalogs["generic"])


def _org_line(domain: str) -> str:
    return {
        "supply_chain": "Manufacturing Co. · Supply Chain",
        "anomaly": "Plant Operations · Anomaly Monitoring",
        "ecommerce": "Retail Co. · Commerce",
        "analytics": "Revenue Team · Analytics",
        "generic": "Your Organization · Operations",
    }.get(domain, "Your Organization · Operations")


def _hint_line(domain: str) -> str:
    return {
        "supply_chain": "Sample report layouts for inventory apps — always render with demo data in preview and on the Reports page.",
        "anomaly": "Sample anomaly report layouts — demonstration data for dashboards and alert workflows.",
        "ecommerce": "Sample commerce report layouts — orders, revenue, and product performance with demo data.",
        "analytics": "Sample analytics report layouts — pipeline and revenue views with demonstration data.",
        "generic": "Sample report layouts ship with every generated app and always use demonstration data.",
    }.get(domain, "Sample report layouts with demonstration data.")


def _supply_chain_document(template_id: str, project_id: str, seed: Dict[str, Any]) -> Dict[str, Any]:
    sc = seed if seed.get("reports/summary") else supply_chain_seed_data(project_id)
    dash = sc.get("dashboard") or {}
    summary = sc.get("reports/summary") or {}
    materials = (sc.get("materials") or {}).get("items") or []
    templates = template_catalog("supply_chain")
    meta = next((t for t in templates if t["id"] == template_id), templates[0])
    kpis = [
        {"label": "SKUs", "value": str(dash.get("total_skus", len(materials)))},
        {"label": "Inventory value", "value": f"${int(dash.get('total_inventory_value', 0)):,}"},
        {"label": "Low-stock alerts", "value": str(dash.get("low_stock_alerts", 0))},
        {"label": "Turnover", "value": f"{dash.get('inventory_turnover', 0)}×"},
        {"label": "Supplier on-time", "value": f"{dash.get('supplier_on_time_pct', 0)}%"},
    ]
    sections: List[Dict[str, Any]] = []
    aging = summary.get("inventory_aging") or []
    turnover = summary.get("turnover_by_category") or []
    if template_id in ("inventory_operations_summary_v1", "inventory_aging_excess_v1") and aging:
        sections.append({
            "type": "table",
            "title": "Inventory aging",
            "columns": ["Bucket", "Value ($)", "% of total"],
            "rows": [[r.get("bucket"), f"{int(r.get('value', 0)):,}", f"{r.get('pct')}%"] for r in aging],
        })
    if template_id == "inventory_aging_excess_v1":
        sections.append({
            "type": "callout",
            "title": "Excess & stock-out (30 days)",
            "body": (
                f"Excess stock value: ${int(summary.get('excess_stock_value', 0)):,} · "
                f"Stock-out events: {summary.get('stockout_events_last_30d', 0)}"
            ),
        })
    if template_id in ("inventory_operations_summary_v1", "inventory_value_turnover_v1") and turnover:
        sections.append({
            "type": "table",
            "title": "Turnover by category",
            "columns": ["Category", "Turns / year"],
            "rows": [[r.get("category"), str(r.get("turns"))] for r in turnover],
        })
    sections.append({
        "type": "table",
        "title": "Materials snapshot",
        "columns": ["SKU", "Name", "Category", "On hand", "Status"],
        "rows": [
            [m.get("sku"), m.get("name"), m.get("category"), str(m.get("on_hand_qty")), m.get("status")]
            for m in materials[:8]
        ],
    })
    exports = [
        {"label": "Download materials (CSV)", "path": "/api/materials", "filename": "materials.csv"},
        {"label": "Download aging (CSV)", "path": "/api/reports/summary", "filename": "inventory-aging.csv"},
    ]
    return _wrap_document("supply_chain", meta, kpis, sections, exports)


def _anomaly_document(template_id: str, project_id: str, seed: Dict[str, Any]) -> Dict[str, Any]:
    dash = seed.get("dashboard") if isinstance(seed.get("dashboard"), dict) else anomaly_dashboard_payload(project_id)
    rep = seed.get("reports/summary") or anomaly_reports_summary()
    anomalies = (seed.get("anomalies") or anomaly_anomalies_list()).get("items") or []
    templates = template_catalog("anomaly")
    meta = next((t for t in templates if t["id"] == template_id), templates[0])
    kpis = [
        {"label": "Anomalies today", "value": str(dash.get("total_anomalies_today", rep.get("total_anomalies", 0)))},
        {"label": "Active alerts", "value": str(dash.get("active_alerts", 0))},
        {"label": "Metrics monitored", "value": str(dash.get("metrics_monitored", len(dash.get("metrics") or [])))},
        {"label": "Avg anomaly score", "value": str(dash.get("avg_anomaly_score", rep.get("anomaly_rate", 0)))},
        {"label": "MTTD (min)", "value": str(rep.get("mttd_minutes", 4.7))},
    ]
    sections: List[Dict[str, Any]] = []
    if template_id == "anomaly_operations_summary_v1":
        by_sev = rep.get("by_severity") or []
        sections.append({
            "type": "table",
            "title": "Anomalies by severity",
            "columns": ["Severity", "Count"],
            "rows": [[r.get("name"), str(r.get("value"))] for r in by_sev],
        })
    if template_id == "anomaly_metric_breakdown_v1":
        top = rep.get("top_metrics") or []
        sections.append({
            "type": "table",
            "title": "Top contributing metrics",
            "columns": ["Metric", "Count", "Severity"],
            "rows": [[r.get("metric_name"), str(r.get("count")), r.get("severity")] for r in top],
        })
    sections.append({
        "type": "table",
        "title": "Recent anomalies",
        "columns": ["Time", "Metric", "Value", "Severity", "Status"],
        "rows": [
            [a.get("timestamp", "")[:16], a.get("metric_name"), str(a.get("value")), a.get("severity"), a.get("status")]
            for a in anomalies[:8]
        ],
    })
    exports = [{"label": "Download anomalies (CSV)", "path": "/api/anomalies", "filename": "anomalies.csv"}]
    return _wrap_document("anomaly", meta, kpis, sections, exports)


def _ecommerce_document(template_id: str, project_id: str, seed: Dict[str, Any]) -> Dict[str, Any]:
    dash = generic_app_dashboard(project_id)
    if isinstance(seed.get("dashboard"), dict):
        dash = {**dash, **{k: v for k, v in seed["dashboard"].items() if v}}
    products = (seed.get("products") or {}).get("items") if isinstance(seed.get("products"), dict) else seed.get("products")
    if not products:
        products = [
            {"sku": "SKU-101", "name": "Wireless Headphones", "category": "Electronics", "price": 89.99, "units_sold": 420},
            {"sku": "SKU-204", "name": "Organic Cotton Tee", "category": "Apparel", "price": 24.5, "units_sold": 890},
            {"sku": "SKU-318", "name": "Stainless Water Bottle", "category": "Home", "price": 32.0, "units_sold": 310},
        ]
    templates = template_catalog("ecommerce")
    meta = next((t for t in templates if t["id"] == template_id), templates[0])
    revenue = int(dash.get("revenue") or dash.get("total_sales") or 420000)
    kpis = [
        {"label": "Revenue", "value": f"${revenue:,}"},
        {"label": "Orders today", "value": str(dash.get("orders_today", 48))},
        {"label": "Conversion", "value": f"{dash.get('conversion_rate', 3.2)}%"},
        {"label": "Active users", "value": str(dash.get("active_users", 1240))},
        {"label": "Growth", "value": f"{dash.get('growth_rate', 12.4)}%"},
    ]
    sections: List[Dict[str, Any]] = []
    if template_id == "ecommerce_sales_summary_v1":
        sections.append({
            "type": "callout",
            "title": "Order funnel (demo)",
            "body": "Cart → Checkout → Paid: 1,240 → 890 → 742 (59.8% checkout completion)",
        })
    sections.append({
        "type": "table",
        "title": "Top products",
        "columns": ["SKU", "Name", "Category", "Price", "Units sold"],
        "rows": [
            [p.get("sku"), p.get("name"), p.get("category"), f"${p.get('price', 0)}", str(p.get("units_sold", 0))]
            for p in (products or [])[:8]
        ],
    })
    exports = [{"label": "Download products (CSV)", "path": "/api/products", "filename": "products.csv"}]
    return _wrap_document("ecommerce", meta, kpis, sections, exports)


def _analytics_document(template_id: str, project_id: str, seed: Dict[str, Any]) -> Dict[str, Any]:
    dash = generic_app_dashboard(project_id)
    templates = template_catalog("analytics")
    meta = next((t for t in templates if t["id"] == template_id), templates[0])
    kpis = [
        {"label": "Pipeline value", "value": f"${int(dash.get('revenue', 500000) * 1.4):,}"},
        {"label": "Win rate", "value": "32%"},
        {"label": "Open deals", "value": str(dash.get("pending", 84))},
        {"label": "Forecast (Q)", "value": f"${int(dash.get('revenue', 400000) * 1.1):,}"},
        {"label": "Growth", "value": f"{dash.get('growth_rate', 14.2)}%"},
    ]
    stages = [
        ["Prospect", "42", "$1.2M"],
        ["Qualified", "28", "$890K"],
        ["Proposal", "16", "$620K"],
        ["Negotiation", "9", "$410K"],
        ["Closed Won", "12", "$380K"],
    ]
    sections: List[Dict[str, Any]] = []
    if template_id == "analytics_pipeline_summary_v1":
        sections.append({
            "type": "table",
            "title": "Pipeline by stage",
            "columns": ["Stage", "Deals", "Value"],
            "rows": stages,
        })
    if template_id == "analytics_revenue_forecast_v1":
        trend = dash.get("trend") or [80, 92, 88, 101, 110, 105, 118, 124, 130, 128]
        sections.append({
            "type": "table",
            "title": "Monthly revenue trend (indexed)",
            "columns": ["Period", "Index"],
            "rows": [[f"Month {i + 1}", str(v)] for i, v in enumerate(trend[:10])],
        })
    sections.append({
        "type": "callout",
        "title": "Highlights",
        "body": "Demo analytics report — connect your CRM or warehouse to replace this sample template.",
    })
    exports = [{"label": "Download pipeline (CSV)", "path": "/api/dashboard/kpis", "filename": "pipeline.csv"}]
    return _wrap_document("analytics", meta, kpis, sections, exports)


def _generic_document(template_id: str, project_id: str, seed: Dict[str, Any]) -> Dict[str, Any]:
    dash = seed.get("dashboard") if isinstance(seed.get("dashboard"), dict) else generic_app_dashboard(project_id)
    templates = template_catalog("generic")
    meta = next((t for t in templates if t["id"] == template_id), templates[0])
    kpis = [
        {"label": "Total records", "value": str(dash.get("total", dash.get("total_items", 1200)))},
        {"label": "Active", "value": str(dash.get("active", 840))},
        {"label": "Pending", "value": str(dash.get("pending", 62))},
        {"label": "Completed", "value": str(dash.get("completed", 980))},
    ]
    sections = [
        {
            "type": "callout",
            "title": "About this sample",
            "body": "This template demonstrates how reports render in your generated app. Data is synthetic for preview.",
        },
        {
            "type": "table",
            "title": "Activity snapshot",
            "columns": ["Metric", "Value"],
            "rows": [[k.replace("_", " ").title(), str(v)] for k, v in list(dash.items())[:8] if not isinstance(v, (list, dict))],
        },
    ]
    exports = [{"label": "Download KPIs (CSV)", "path": "/api/dashboard/kpis", "filename": "kpis.csv"}]
    return _wrap_document("generic", meta, kpis, sections, exports)


def _wrap_document(
    domain: str,
    meta: TemplateMeta,
    kpis: List[Dict[str, str]],
    sections: List[Dict[str, Any]],
    exports: List[Dict[str, str]],
) -> Dict[str, Any]:
    now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    catalog = template_catalog(domain)
    return {
        "domain": domain,
        "organization": _org_line(domain),
        "hint": _hint_line(domain),
        "templates": catalog,
        "template_id": meta["id"],
        "title": meta["title"],
        "subtitle": meta.get("subtitle", ""),
        "report_generated_at": now,
        "kpis": kpis,
        "sections": sections,
        "exports": exports,
    }


def build_sample_report_document(
    domain: str,
    template_id: Optional[str] = None,
    project_id: str = "",
    seed: Optional[Dict[str, Any]] = None,
    requirement: str = "",
    prd: str = "",
    files: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    seed = seed or {}
    if not domain or domain == "auto":
        domain = detect_report_domain(requirement, prd, files, seed)
    catalog = template_catalog(domain)
    tid = template_id or (catalog[0]["id"] if catalog else "operations_summary_v1")
    if not any(t["id"] == tid for t in catalog):
        tid = catalog[0]["id"]
    builders = {
        "supply_chain": _supply_chain_document,
        "anomaly": _anomaly_document,
        "ecommerce": _ecommerce_document,
        "analytics": _analytics_document,
        "generic": _generic_document,
    }
    return builders.get(domain, _generic_document)(tid, project_id, seed)


def enrich_seed_with_report_templates(
    seed: Dict[str, Any],
    requirement: str = "",
    prd: str = "",
    files: Optional[Dict[str, str]] = None,
    project_id: str = "",
) -> Dict[str, Any]:
    """Attach reports/templates + default reports/template payload to seed_data.json."""
    out = dict(seed or {})
    domain = detect_report_domain(requirement, prd, files, out)
    catalog = template_catalog(domain)
    doc = build_sample_report_document(domain, catalog[0]["id"], project_id, out, requirement, prd, files)
    out["reports/templates"] = {"templates": catalog, "domain": domain}
    out["reports/template"] = doc
    return out


def load_project_seed(project_name: str) -> Tuple[Dict[str, Any], str, str]:
    """Load seed + requirement hints from generated project if present."""
    requirement = ""
    prd = ""
    seed: Dict[str, Any] = {}
    try:
        from app_builder.services.runtime_paths import resolve_project_root
        import os

        root = resolve_project_root(project_name)
        seed_path = os.path.join(root, "backend", "seed_data.json")
        if os.path.isfile(seed_path):
            with open(seed_path, "r", encoding="utf-8") as fh:
                seed = json.load(fh)
        meta_path = os.path.join(root, "builder_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            requirement = str(meta.get("requirement") or meta.get("prompt") or "")
            prd = str(meta.get("prd") or "")
    except Exception:
        pass
    return seed, requirement, prd


def build_sample_report_for_project(
    project_name: str,
    template_id: Optional[str] = None,
) -> Dict[str, Any]:
    seed, requirement, prd = load_project_seed(project_name)
    domain = detect_report_domain(requirement, prd, None, seed)
    return build_sample_report_document(
        domain, template_id, project_name, seed, requirement, prd, None
    )


def resolve_reports_template_payload(
    seed: Dict[str, Any],
    template_id: Optional[str] = None,
    project_id: str = "",
) -> Dict[str, Any]:
    """Dynamic router: /reports/template and /reports/templates."""
    domain = detect_report_domain("", "", None, seed)
    catalog = template_catalog(domain)
    tid = template_id or catalog[0]["id"]
    if template_id:
        return build_sample_report_document(domain, tid, project_id, seed)
    stored = seed.get("reports/template")
    if isinstance(stored, dict) and stored.get("template_id") == tid:
        return stored
    return build_sample_report_document(domain, tid, project_id, seed)
