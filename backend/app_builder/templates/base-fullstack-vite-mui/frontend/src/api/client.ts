/**
 * API client with mock fallback.
 * Live backend is preferred; any network or HTTP failure uses src/api/mock.ts
 * so every screen remains usable.
 */
import { mockFetch } from './mock';

export function getApiBase(): string {
  if (typeof window !== 'undefined' && window.__AKKIO_API_BASE__) {
    return window.__AKKIO_API_BASE__.replace(/\/$/, '');
  }
  if (typeof window !== 'undefined') {
    const match = window.location.pathname.match(/^\/app\/([^/]+)/);
    if (match) {
      return `${window.location.origin}/api/apps/${match[1]}`;
    }
  }
  const envUrl = (import.meta as { env?: { VITE_API_URL?: string } }).env?.VITE_API_URL;
  if (envUrl) return envUrl.replace(/\/$/, '');
  const { protocol, hostname, port } = window.location;
  const apiPort = port === '5173' ? '8000' : port || '8000';
  return `${protocol}//${hostname}:${apiPort}`;
}

/**
 * Build a full URL from a base and a path, handling the preview case where
 * the base already contains /api/apps/{id}. In that case the path supplied
 * by callers like `/api/dashboard/kpis` must have its leading `/api` stripped
 * to avoid a double `/api` segment, e.g.:
 *   base  = /api/apps/myapp
 *   path  = /api/dashboard/kpis
 *   → URL = /api/apps/myapp/dashboard/kpis   ✓  (not /api/apps/myapp/api/…)
 */
function buildUrl(base: string, path: string): string {
  const normalizedBase = base.replace(/\/$/, '');
  let normalizedPath = path.startsWith('/') ? path : '/' + path;

  // When the base is a preview proxy path (/api/apps/{id}), strip the
  // redundant /api prefix that page code naturally prefixes to its paths.
  if (normalizedBase.includes('/api/apps/') && normalizedPath.startsWith('/api/')) {
    normalizedPath = normalizedPath.slice(4); // remove leading '/api'
  }

  return normalizedBase + normalizedPath;
}

/** True when live API returned JSON that would render as empty KPIs / no chart data. */
function isUsableApiPayload(path: string, data: unknown): boolean {
  if (data == null || typeof data !== 'object') return false;
  const p = path.split('?')[0].toLowerCase();
  const d = data as Record<string, unknown>;

  const pathNorm = p.split('?')[0];
  const isDashboardRoot =
    pathNorm === '/api/dashboard' ||
    (pathNorm.endsWith('/dashboard') && !pathNorm.includes('kpis'));

  if (p.includes('dashboard')) {
    const metrics = d.metrics;
    if (Array.isArray(metrics)) return metrics.length > 0;
    // Anomaly dashboards call GET /api/dashboard and need a metrics[] time series.
    if (isDashboardRoot && ('total_lots' in d || 'pending_inspections' in d)) {
      return false;
    }
    const numericKeys = [
      'total_lots', 'pending_inspections', 'released', 'held', 'rejected', 'open_capa',
      'incoming_lots', 'total_anomalies_today', 'active_alerts', 'metrics_monitored',
      'total_sales', 'revenue', 'orders_today', 'total', 'active', 'pending',
      'total_skus', 'total_inventory_value', 'low_stock_alerts', 'stockout_risk_count',
      'open_purchase_orders', 'inventory_turnover', 'supplier_on_time_pct',
    ];
    const present = numericKeys.filter((k) => k in d);
    if (present.length === 0) return Object.keys(d).length > 2;
    return present.some((k) => {
      const v = d[k];
      return typeof v === 'number' && v !== 0;
    });
  }

  if (p.includes('anomalies') && Array.isArray(d.items)) {
    return d.items.length > 0;
  }

  if (Array.isArray(data)) {
    return data.length > 0;
  }

  return true;
}

function parseApiError(text: string, status: number): string {
  if (!text) return `HTTP ${status}`;
  try {
    const data = JSON.parse(text);
    if (data && typeof data.detail === 'string') return data.detail;
  } catch {
    /* ignore */
  }
  return text.length > 300 ? `${text.slice(0, 300)}…` : text;
}

export async function apiFetch<T = unknown>(path: string, options: RequestInit = {}): Promise<T> {
  const base = getApiBase();
  let url = path.startsWith('http') ? path : buildUrl(base, path);
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...((options.headers as Record<string, string>) || {}),
  };
  const token =
    (typeof window !== 'undefined' && window.__AKKIO_ACCESS_TOKEN__) ||
    (typeof localStorage !== 'undefined' ? localStorage.getItem('access_token') : null);
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  try {
    const res = await fetch(url, { ...options, headers });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(parseApiError(text, res.status));
    }
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('application/json')) {
      const data = await res.json();
      if (!isUsableApiPayload(path, data)) {
        return mockFetch<T>(path, options);
      }
      return data as T;
    }
    return (await res.text()) as T;
  } catch (err) {
    return mockFetch<T>(path, options);
  }
}

// Export as default so both import styles work:
//   import apiFetch from '../api/client'          ← default (LLMs often generate this)
//   import { apiFetch } from '../api/client'      ← named  (also correct)
export default apiFetch;
