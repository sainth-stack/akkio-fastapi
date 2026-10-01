/**
 * Deterministic mock API used when the live backend is unavailable.
 * Codegen should extend MOCK_TABLES with domain entities from the PRD.
 */
type Json = Record<string, unknown>;

const now = () => new Date().toISOString();

const MOCK_TABLES: Record<string, Json[]> = {
  dashboard: [
    {
      total_lots: 200,
      pending_inspections: 18,
      released: 142,
      held: 24,
      rejected: 16,
      open_capa: 9,
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
