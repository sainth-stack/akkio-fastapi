/**
 * Two ready-to-send sample prompts (no placeholders). User can read them in the sidebar and click Use.
 */
export const SAMPLE_PROMPT_TEMPLATES = [
    {
        id: 'operations_records_v1',
        title: 'Operations & records',
        subtitle: 'Inventory / supply chain style operations app',
        prompt: `Build a supply chain inventory management app for a manufacturing company.

Core entities with list and detail pages:
- Materials / SKUs with on-hand quantity, reorder point, and safety stock
- Suppliers with lead time and on-time delivery score
- Purchase orders with status (Draft, In Transit, Received)

Dashboard KPIs with realistic non-zero demo data:
- Total SKUs and total inventory value
- Low-stock and excess-stock alert counts
- Inventory turnover and supplier on-time percentage

Pages:
- Dashboard with KPI cards and a consumption or stock trend chart
- Materials table with search, filters, and row status (OK, Low, Critical)
- Purchase orders and suppliers views
- Alerts for low stock, delayed POs, and abnormal consumption
- Reports page with summary report (KPIs + breakdown table) and detail table (one row per material)

Workflows: material receipt, reorder suggestion, and PO approval (simple statuses).

Use realistic synthetic demo data on every screen by default. No login required.`,
    },
    {
        id: 'procurement_v1',
        title: 'Procurement & spend',
        subtitle: 'Manufacturing procurement (3 pages + RAG, Predict, Observability)',
        prompt: `Build a manufacturing procurement dashboard (Dashboard, Analytics, Reports)
with RAG chat, Predict, and LLM observability to track prompts, outputs, tokens, and model cost.

Use realistic synthetic demo data: purchase orders, vendors, spend, and cycle-time KPIs.
App domain: Manufacturing. No login required.`,
    },
    {
        id: 'monitoring_insights_v1',
        title: 'Monitoring & insights',
        subtitle: 'Equipment / anomaly monitoring style app',
        prompt: `Build an equipment monitoring and anomaly detection app for a CNC production line.

Metrics to track with thresholds:
- Spindle speed (RPM), feed rate (mm/min), tool wear index (%)
- Vibration (mm/s) and coolant temperature (°C)

Dashboard with non-zero demo values:
- Anomalies detected today and active alerts
- Metrics monitored and average anomaly score
- Live trend charts per metric with anomaly markers

Pages:
- Main operations dashboard
- Anomaly feed with timestamp, metric, severity, and status (Open, Acknowledged, Resolved)
- Threshold configuration per metric (min, max, z-score)
- Reports page with summary report (KPIs + severity breakdown) and detail table (one row per anomaly event)

Use realistic synthetic time-series and alert data everywhere. No login required.`,
    },
];
