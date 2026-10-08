"""
Supply chain / inventory demo payloads and mock API for generated apps.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from app_builder.services.demo_api_data import _rng_for_project


def looks_like_supply_chain(requirement: str = "", prd: str = "", files: Dict[str, str] | None = None) -> bool:
    text = f"{requirement} {prd}".lower()
    keywords = (
        "supply chain", "inventory management", "inventory level", "stock-out", "stock out",
        "reorder", "purchase order", "warehouse", "material receipt", "safety stock",
        "supplier lead", "goods receipt", "replenishment", "sku", "turnover report",
        "excess stock", "inventory aging",
    )
    if any(k in text for k in keywords):
        return True
    if not files:
        return False
    markers = (
        "InventoryPage", "MaterialsPage", "PurchaseOrdersPage", "SuppliersPage",
        "WarehousesPage", "ReplenishmentPage", "GoodsReceiptsPage",
    )
    for path in files:
        if any(m in path for m in markers):
            return True
    layout = files.get("frontend/src/layout/AppLayout.tsx", "") + files.get("frontend/src/App.tsx", "")
    return any(w in layout for w in ("Inventory", "Purchase Orders", "Suppliers", "Warehouses", "Replenishment"))


def _materials(rng) -> List[Dict[str, Any]]:
    names = [
        ("MAT-1001", "Steel Rod 12mm", "Raw Material", "kg"),
        ("MAT-1002", "Aluminum Sheet 3mm", "Raw Material", "sheet"),
        ("MAT-2001", "Bearing Assembly A", "Component", "ea"),
        ("MAT-2002", "Hydraulic Seal Kit", "Component", "kit"),
        ("MAT-3001", "Finished Gearbox GX-7", "Finished Good", "ea"),
        ("MAT-3002", "Motor Mount Bracket", "Finished Good", "ea"),
        ("MAT-1100", "Cutting Fluid CF-40", "Consumable", "L"),
        ("MAT-1101", "Carbide Insert CI-22", "Consumable", "ea"),
    ]
    items = []
    for i, (code, name, category, uom) in enumerate(names, 1):
        on_hand = rng.randint(120, 4800)
        reorder = rng.randint(80, 400)
        safety = rng.randint(40, 200)
        items.append({
            "id": i,
            "sku": code,
            "name": name,
            "category": category,
            "uom": uom,
            "on_hand_qty": on_hand,
            "reserved_qty": rng.randint(10, min(200, on_hand // 3)),
            "available_qty": on_hand - rng.randint(10, 80),
            "reorder_point": reorder,
            "safety_stock": safety,
            "status": rng.choice(["OK", "LOW", "CRITICAL", "EXCESS"]),
            "unit_cost": round(rng.uniform(4.5, 420.0), 2),
            "inventory_value": round(on_hand * rng.uniform(4.5, 420.0), 2),
        })
    return items


def _suppliers(rng) -> List[Dict[str, Any]]:
    return [
        {"id": 1, "name": "NorthStar Metals", "lead_time_days": 7, "on_time_pct": 94.2, "quality_score": 92, "active_pos": 3},
        {"id": 2, "name": "Precision Parts Co.", "lead_time_days": 14, "on_time_pct": 88.5, "quality_score": 89, "active_pos": 2},
        {"id": 3, "name": "Global Industrial Supply", "lead_time_days": 21, "on_time_pct": 91.0, "quality_score": 87, "active_pos": 4},
        {"id": 4, "name": "FastFlow Logistics", "lead_time_days": 5, "on_time_pct": 96.8, "quality_score": 90, "active_pos": 1},
    ]


def _purchase_orders(rng) -> List[Dict[str, Any]]:
    statuses = ["Draft", "Submitted", "In Transit", "Partially Received", "Closed"]
    items = []
    for i in range(1, 13):
        items.append({
            "id": i,
            "po_number": f"PO-2026-{1000 + i}",
            "supplier": rng.choice(["NorthStar Metals", "Precision Parts Co.", "Global Industrial Supply"]),
            "order_date": f"2026-09-{rng.randint(1, 28):02d}",
            "expected_date": f"2026-10-{rng.randint(1, 20):02d}",
            "status": rng.choice(statuses),
            "lines": rng.randint(2, 8),
            "total_value": round(rng.uniform(12000, 185000), 2),
            "received_pct": rng.randint(0, 100),
        })
    return items


def _inventory_alerts(rng) -> List[Dict[str, Any]]:
    types = ["LOW_STOCK", "STOCKOUT_RISK", "EXCESS_STOCK", "DELAYED_PO", "ABNORMAL_CONSUMPTION"]
    severities = ["Low", "Medium", "High", "Critical"]
    items = []
    for i in range(1, 11):
        items.append({
            "id": f"al{i}",
            "type": rng.choice(types),
            "severity": rng.choice(severities),
            "material": rng.choice(["Steel Rod 12mm", "Bearing Assembly A", "Cutting Fluid CF-40"]),
            "message": "Threshold breached — review replenishment",
            "timestamp": f"2026-10-07T{rng.randint(8, 18):02d}:00:00Z",
            "status": rng.choice(["Open", "Acknowledged", "Resolved"]),
        })
    return items


def _inventory_anomalies(rng) -> List[Dict[str, Any]]:
    items = []
    for i in range(1, 9):
        items.append({
            "id": f"inv-a{i}",
            "timestamp": f"2026-10-0{rng.randint(1, 7)}T{rng.randint(8, 20):02d}:15:00Z",
            "material": rng.choice(["Aluminum Sheet 3mm", "Hydraulic Seal Kit", "Motor Mount Bracket"]),
            "metric": rng.choice(["consumption_rate", "issue_variance", "cycle_count_delta"]),
            "value": round(rng.uniform(1.2, 4.8), 2),
            "expected_max": 1.0,
            "severity": rng.choice(["Medium", "High", "Critical"]),
            "status": rng.choice(["Open", "Investigating", "Closed"]),
        })
    return items


def supply_chain_seed_data(project_id: str = "") -> Dict[str, Any]:
    rng = _rng_for_project(project_id or "supply-chain")
    materials = _materials(rng)
    suppliers = _suppliers(rng)
    pos = _purchase_orders(rng)
    alerts = _inventory_alerts(rng)
    anomalies = _inventory_anomalies(rng)
    low_stock = sum(1 for m in materials if m["status"] in ("LOW", "CRITICAL"))
    consumption_trend = [{"week": f"W{i}", "consumption": rng.randint(800, 2400)} for i in range(1, 13)]

    dashboard = {
        "total_skus": len(materials),
        "total_inventory_value": round(sum(m["inventory_value"] for m in materials), 2),
        "low_stock_alerts": low_stock,
        "stockout_risk_count": rng.randint(2, 9),
        "open_purchase_orders": sum(1 for p in pos if p["status"] not in ("Closed",)),
        "pending_receipts": rng.randint(3, 12),
        "inventory_turnover": round(rng.uniform(4.2, 11.5), 1),
        "supplier_on_time_pct": round(rng.uniform(88.0, 97.5), 1),
        "excess_stock_count": sum(1 for m in materials if m["status"] == "EXCESS"),
        "consumption_trends": consumption_trend,
        "materials_by_level": materials[:6],
        # Legacy KPI keys some dashboards still read
        "total_lots": rng.randint(90, 220),
        "pending_inspections": rng.randint(8, 45),
        "released": rng.randint(70, 160),
        "held": rng.randint(4, 22),
        "rejected": rng.randint(3, 18),
        "open_capa": rng.randint(2, 14),
        "incoming_lots": rng.randint(10, 35),
        "total_items": sum(m["on_hand_qty"] for m in materials),
        "total_orders": len(pos),
        "pending_orders": sum(1 for p in pos if p["status"] in ("Submitted", "In Transit")),
        "total_suppliers": len(suppliers),
    }

    reports_summary = {
        "inventory_aging": [
            {"bucket": "0-30 days", "value": round(rng.uniform(120000, 280000), 2), "pct": 42},
            {"bucket": "31-60 days", "value": round(rng.uniform(80000, 160000), 2), "pct": 28},
            {"bucket": "61-90 days", "value": round(rng.uniform(40000, 90000), 2), "pct": 18},
            {"bucket": "90+ days", "value": round(rng.uniform(20000, 60000), 2), "pct": 12},
        ],
        "turnover_by_category": [
            {"category": "Raw Material", "turns": round(rng.uniform(6, 12), 1)},
            {"category": "Component", "turns": round(rng.uniform(4, 9), 1)},
            {"category": "Finished Good", "turns": round(rng.uniform(2, 6), 1)},
        ],
        "excess_stock_value": round(rng.uniform(45000, 120000), 2),
        "stockout_events_last_30d": rng.randint(1, 7),
        "report_generated_at": "2026-10-07T12:00:00Z",
        "template": "inventory_operations_summary_v1",
    }

    thresholds = {
        "items": [
            {"id": "low_stock", "name": "Low stock", "threshold_pct": 15, "enabled": True},
            {"id": "excess", "name": "Excess stock", "threshold_pct": 130, "enabled": True},
            {"id": "consumption", "name": "Abnormal consumption", "z_score": 2.5, "enabled": True},
            {"id": "supplier_delay", "name": "Delayed supplier delivery", "days_late": 3, "enabled": True},
            {"id": "stockout", "name": "Stock-out risk", "days_cover": 5, "enabled": True},
        ],
        "total": 5,
    }

    return {
        "dashboard": dashboard,
        "kpis": dashboard,
        "dashboard/kpis": dashboard,
        "materials": {"items": materials, "total": len(materials)},
        "inventory": {"items": materials, "total": len(materials)},
        "inventory_levels": {"items": materials, "total": len(materials)},
        "suppliers": {"items": suppliers, "total": len(suppliers)},
        "purchase_orders": {"items": pos, "total": len(pos)},
        "purchase-orders": {"items": pos, "total": len(pos)},
        "alerts": {"items": alerts, "total": len(alerts)},
        "anomalies": {"items": anomalies, "total": len(anomalies)},
        "consumption_trends": {"series": consumption_trend},
        "reports/summary": reports_summary,
        "reports/inventory-aging": reports_summary,
        "thresholds": thresholds,
    }


def supply_chain_collection_fallback(collection: str, project_id: str = "") -> Dict[str, Any]:
    seed = supply_chain_seed_data(project_id)
    key = collection.replace("-", "_")
    if collection in seed:
        return seed[collection]
    if key in seed:
        return seed[key]
    if key == "purchase_orders" and "purchase_orders" in seed:
        return seed["purchase_orders"]
    return {"items": [], "total": 0}


def reports_page_tsx() -> str:
    """Always-working sample report template with export."""
    return r'''import { useQuery } from '@tanstack/react-query';
import { Box, Button, Card, CardContent, Typography, Table, TableHead, TableRow, TableCell, TableBody, Stack, Chip } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import { apiFetch } from '../api/client';

function downloadCsv(filename: string, rows: Record<string, unknown>[]) {
  if (!rows.length) return;
  const headers = Object.keys(rows[0]);
  const csv = [headers.join(','), ...rows.map((r) => headers.map((h) => JSON.stringify(r[h] ?? '')).join(','))].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function ReportsPage() {
  const { data: summary } = useQuery({
    queryKey: ['reports-summary'],
    queryFn: () => apiFetch<Record<string, unknown>>('/api/reports/summary'),
  });
  const { data: materials } = useQuery({
    queryKey: ['materials-export'],
    queryFn: () => apiFetch<{ items: Record<string, unknown>[] }>('/api/materials'),
  });

  const aging = (summary?.inventory_aging as { bucket: string; value: number; pct: number }[]) || [];
  const turnover = (summary?.turnover_by_category as { category: string; turns: number }[]) || [];

  return (
    <Box sx={{ p: 3 }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight={800}>Inventory Reports</Typography>
        <Chip label="Sample report template" color="primary" variant="outlined" />
      </Stack>
      <Stack direction="row" spacing={2} sx={{ mb: 3 }}>
        <Button
          variant="contained"
          startIcon={<DownloadIcon />}
          onClick={() => downloadCsv('inventory-materials.csv', materials?.items || [])}
        >
          Download materials (CSV)
        </Button>
        <Button
          variant="outlined"
          startIcon={<DownloadIcon />}
          onClick={() => downloadCsv('inventory-aging.csv', aging.map((a) => ({ ...a })))}
        >
          Download aging report (CSV)
        </Button>
      </Stack>
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>Inventory aging</Typography>
          <Table size="small">
            <TableHead><TableRow><TableCell>Bucket</TableCell><TableCell>Value ($)</TableCell><TableCell>%</TableCell></TableRow></TableHead>
            <TableBody>
              {aging.map((row) => (
                <TableRow key={row.bucket}>
                  <TableCell>{row.bucket}</TableCell>
                  <TableCell>{row.value.toLocaleString()}</TableCell>
                  <TableCell>{row.pct}%</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>Turnover by category</Typography>
          <Table size="small">
            <TableHead><TableRow><TableCell>Category</TableCell><TableCell>Turns / year</TableCell></TableRow></TableHead>
            <TableBody>
              {turnover.map((row) => (
                <TableRow key={row.category}>
                  <TableCell>{row.category}</TableCell>
                  <TableCell>{row.turns}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </Box>
  );
}
'''


def supply_chain_mock_ts(project_id: str = "") -> str:
    seed = supply_chain_seed_data(project_id)
    seed_js = json.dumps(seed, ensure_ascii=False)
    return f"""// Supply chain / inventory demo API — auto-generated sample data
const SEED: Record<string, unknown> = {seed_js};

function ok<T>(data: T): Promise<T> {{ return Promise.resolve(data); }}

export async function mockFetch<T = unknown>(path: string, options: RequestInit = {{}}): Promise<T> {{
  const method = (options.method || 'GET').toUpperCase();
  const pathNoQuery = path.split('?')[0];
  const clean = pathNoQuery.replace(/^\\/api\\//, '').replace(/^\\//, '');

  if (method === 'GET') {{
    if (clean === 'dashboard' || clean === 'dashboard/kpis') return ok(SEED.dashboard) as T;
    if (clean === 'reports/summary' || clean === 'reports/inventory-aging') {{
      return ok(SEED['reports/summary'] || SEED.dashboard) as T;
    }}
    if (clean === 'reports/templates') {{
      return ok(SEED['reports/templates'] || {{ templates: [] }}) as T;
    }}
    if (clean.startsWith('reports/template')) {{
      const tpl = SEED['reports/template'];
      if (tpl) return ok(tpl) as T;
    }}
    if (clean === 'consumption_trends') return ok(SEED.consumption_trends) as T;
    const direct = SEED[clean] ?? SEED[clean.replace(/-/g, '_')];
    if (direct) return ok(direct) as T;
    const itemsKey = clean.replace(/-/g, '_');
    if (SEED[itemsKey]) return ok(SEED[itemsKey]) as T;
  }}
  return ok({{ ok: true, path: clean, method }}) as T;
}}
"""
