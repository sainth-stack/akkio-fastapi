/**
 * Deterministic mock API used when the live backend is unavailable.
 * Codegen should extend MOCK_TABLES with domain entities from the PRD.
 */
type Json = Record<string, unknown>;

const now = () => new Date().toISOString();

/**
 * Exact-path overrides — checked BEFORE the generic collection routing.
 * Add entries here for endpoints whose path doesn't map 1-to-1 to a
 * collection name (e.g. nested routes like /api/dashboard/kpis).
 * Keys must be the raw path as passed to mockFetch (e.g. '/api/dashboard/kpis').
 */
const MOCK_ENDPOINTS: Record<string, Json> = {
  // Supply-chain / quality / inventory KPIs
  '/api/dashboard/kpis': {
    total_lots: 142,
    pending_inspections: 23,
    released: 98,
    held: 12,
    rejected: 9,
    open_capa: 7,
    incoming_lots: 18,
    total_items: 3420,
    low_stock_alerts: 14,
    on_time_delivery_rate: 94.2,
    inventory_turnover: 8.3,
    total_orders: 287,
    pending_orders: 34,
    total_suppliers: 67,
    active_suppliers: 54,
  },
  // General dashboard summary
  '/api/dashboard': {
    total_sales: 284500,
    orders_today: 47,
    active_users: 1284,
    revenue: 284500,
    growth_rate: 12.4,
    conversion_rate: 3.8,
    total_lots: 142,
    pending_inspections: 23,
    released: 98,
    held: 12,
    rejected: 9,
    open_capa: 7,
    trend: [60, 72, 80, 85, 91, 94, 100, 110, 118, 128],
  },
  '/api/stats': { total: 1284, active: 987, pending: 142, completed: 1089 },
  '/api/metrics': { total: 500, anomalies: 23, alerts: 7, score: 94.2 },
  '/api/reports/summary': { total_anomalies: 156, mttd_minutes: 4.2, anomaly_rate: 3.1 },
};

const MOCK_TABLES: Record<string, Json[]> = {
  dashboard: [
    {
      total_lots: 200,
      pending_inspections: 18,
      released: 142,
      held: 24,
      rejected: 16,
      open_capa: 9,
      total_sales: 284500,
      orders_today: 47,
      active_users: 1284,
      revenue: 284500,
      growth_rate: 12.4,
      conversion_rate: 3.8,
    },
  ],
  suppliers: [
    { id: 1, code: 'SUP-001', name: 'ABC Auto Components', location: 'Detroit, MI', quality_score: 92, ppm: 42, status: 'ACTIVE' },
    { id: 2, code: 'SUP-002', name: 'Prime Precision', location: 'Stuttgart, DE', quality_score: 88, ppm: 67, status: 'ACTIVE' },
  ],
  materials: [
    { id: 1, code: 'MAT-BD', name: 'Brake Disc', type: 'Casting', criticality: 'HIGH', status: 'ACTIVE' },
    { id: 2, code: 'MAT-GB', name: 'Gear Blank', type: 'Forging', criticality: 'MEDIUM', status: 'ACTIVE' },
  ],
  incoming_lots: [
    { id: 1, lot_number: 'LOT-20260925-001', supplier: 'ABC Auto Components', material: 'Brake Disc', quantity: 240, risk_score: 72, inspection_status: 'PENDING', release_status: 'HOLD' },
  ],
};

function collectionFromPath(path: string): { key: string; id?: string } {
  const clean = path.split('?')[0].replace(/^\/api\//, '/').replace(/^\//, '');
  const parts = clean.split('/').filter(Boolean);
  if (parts[0] === 'dashboard') return { key: 'dashboard' };
  return { key: parts[0] || 'items', id: parts[1] };
}

export async function mockFetch<T = unknown>(path: string, options: RequestInit = {}): Promise<T> {
  const method = (options.method || 'GET').toUpperCase();

  // Exact-path overrides take priority for GET requests.
  // Strip query string for matching.
  if (method === 'GET') {
    const pathNoQuery = path.split('?')[0];
    if (Object.prototype.hasOwnProperty.call(MOCK_ENDPOINTS, pathNoQuery)) {
      return MOCK_ENDPOINTS[pathNoQuery] as T;
    }
  }

  const { key, id } = collectionFromPath(path);
  if (!MOCK_TABLES[key]) MOCK_TABLES[key] = [];
  const rows = MOCK_TABLES[key];

  if (method === 'GET' && !id) {
    if (key === 'dashboard') return (rows[0] || {}) as T;
    return { items: rows, total: rows.length } as T;
  }
  if (method === 'GET' && id) {
    const found = rows.find((r) => String(r.id) === String(id)) || rows[0] || { id };
    return found as T;
  }
  if (method === 'POST') {
    let body: Json = {};
    try {
      body = JSON.parse(String(options.body || '{}'));
    } catch {
      body = {};
    }
    const created = { id: rows.length + 1, ...body, created_at: now() };
    rows.push(created);
    return created as T;
  }
  return { ok: true, mocked: true, path, method } as T;
}
