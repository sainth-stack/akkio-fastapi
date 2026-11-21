export function getAkkioApiBase(): string | null {
  const w = window as { __AKKIO_API_BASE__?: string };
  if (w.__AKKIO_API_BASE__) {
    return w.__AKKIO_API_BASE__.replace(/\/$/, '');
  }
  const match = window.location.pathname.match(/^\/app\/([^/]+)/);
  return match ? `/api/apps/${match[1]}` : null;
}

export function getAkkioAccessToken(): string | null {
  const w = window as { __AKKIO_ACCESS_TOKEN__?: string };
  return w.__AKKIO_ACCESS_TOKEN__?.trim() || null;
}
