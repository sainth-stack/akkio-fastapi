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

function parseApiError(text, status) {
  if (!text) return `HTTP ${status}`;
  try {
    const data = JSON.parse(text);
    if (data && typeof data.detail === 'string') return data.detail;
    if (data && data.detail) return JSON.stringify(data.detail);
  } catch {
    /* plain text */
  }
  if (text.includes('Internal Server Error')) {
    return 'Server error — please retry. If this persists, run the app again from Build.';
  }
  return text.length > 300 ? text.slice(0, 300) + '…' : text;
}

export async function apiFetch(path, options = {}) {
  const base = getApiBase();
  let url = path.startsWith('http') ? path : `${base}${path.startsWith('/') ? path : `/${path}`}`;
  const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
  const token = typeof window !== 'undefined' && window.__AKKIO_ACCESS_TOKEN__;
  if (token) {
    headers.Authorization = `Bearer ${token}`;
    if (!path.startsWith('http') && !url.includes('access_token=')) {
      const sep = url.includes('?') ? '&' : '?';
      url = `${url}${sep}access_token=${encodeURIComponent(token)}`;
    }
  }
  const res = await fetch(url, { ...options, headers });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(parseApiError(text, res.status));
  }
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return res.json();
  return res.text();
}
