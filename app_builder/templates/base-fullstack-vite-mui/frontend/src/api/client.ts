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
  let url = path.startsWith('http') ? path : `${base}${path.startsWith('/') ? path : `/${path}`}`;
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
    if (ct.includes('application/json')) return res.json() as Promise<T>;
    return (await res.text()) as T;
  } catch (err) {
    return mockFetch<T>(path, options);
  }
}

// Export as default so both import styles work:
//   import apiFetch from '../api/client'          ← default (LLMs often generate this)
//   import { apiFetch } from '../api/client'      ← named  (also correct)
export default apiFetch;
