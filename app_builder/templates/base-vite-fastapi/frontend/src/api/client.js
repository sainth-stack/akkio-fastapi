/**
 * API client — resolves backend URL for local dev and /app/{project} preview.
 */
export function getApiBase() {
  if (typeof window !== 'undefined' && window.__AKKIO_API_BASE__) {
    return window.__AKKIO_API_BASE__.replace(/\/$/, '');
  }
  if (typeof window !== 'undefined') {
    const match = window.location.pathname.match(/^\/app\/([^/]+)/);
    if (match) {
      return `${window.location.origin}/api/apps/${match[1]}`;
    }
  }
  const { protocol, hostname, port } = window.location;
  const apiPort = port === '5173' ? '8000' : port || '8000';
  return `${protocol}//${hostname}:${apiPort}/api/apps`;
}

export async function apiFetch(path, options = {}) {
  const base = getApiBase();
  const url = path.startsWith('http') ? path : `${base}${path.startsWith('/') ? path : `/${path}`}`;
  const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
  const token = typeof window !== 'undefined' && window.__AKKIO_ACCESS_TOKEN__;
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(url, { ...options, headers });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(text || `HTTP ${res.status}`);
  }
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return res.json();
  return res.text();
}
