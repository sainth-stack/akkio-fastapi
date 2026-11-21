/** Match SPA nav path without false positives (e.g. /predict vs /predictions). */
export function navPathMatches(pathname: string, itemPath: string): boolean {
  if (itemPath === '/') return pathname === '/';
  return pathname === itemPath || pathname.startsWith(`${itemPath}/`);
}

/** Pick the most specific nav item for the current pathname. */
export function resolveActiveNavItem<T extends { path: string }>(
  navItems: T[],
  pathname: string,
): T | undefined {
  let best: T | undefined;
  let bestLen = -1;
  for (const item of navItems) {
    if (!navPathMatches(pathname, item.path)) continue;
    if (item.path.length > bestLen) {
      best = item;
      bestLen = item.path.length;
    }
  }
  return best;
}

export function isPredictNavItem(item: { pageId?: string; path: string; label?: string }): boolean {
  const pid = (item.pageId || '').toLowerCase();
  if (pid === 'predict') return true;
  if (item.path === '/predict' || item.path.endsWith('/predict')) return true;
  return false;
}

export function isCreateTicketNavItem(item: {
  pageId?: string;
  path: string;
  label?: string;
}): boolean {
  const pid = (item.pageId || '').toLowerCase();
  if (pid === 'create_ticket') return true;
  return item.path === '/create-ticket' || item.path.endsWith('/create-ticket');
}

export function isSimilarTicketsNavItem(item: {
  pageId?: string;
  path: string;
  label?: string;
}): boolean {
  const pid = (item.pageId || '').toLowerCase();
  if (pid === 'similar_tickets') return true;
  return item.path === '/similar-tickets' || item.path.endsWith('/similar-tickets');
}

export function isObservabilityNavItem(item: {
  pageId?: string;
  path: string;
  label?: string;
}): boolean {
  const pid = (item.pageId || '').toLowerCase();
  if (pid === 'observability') return true;
  if (item.path === '/observability' || item.path.endsWith('/observability')) return true;
  return false;
}
