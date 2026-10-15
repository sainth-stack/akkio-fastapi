/** Strip redundant "PageName — …" prefixes from generated titles. */
export function cleanDisplayTitle(title: string, pageTitle?: string): string {
  const t = (title || '').trim();
  if (!t) return t;

  const parts = t.split(/\s*[—–-]\s+/);
  if (parts.length >= 2) {
    const head = parts[0].trim();
    const page = (pageTitle || '').trim();
    if (!page || head.toLowerCase() === page.toLowerCase()) {
      const rest = parts.slice(1).join(' — ').trim();
      if (rest) return rest;
    }
  }
  return t;
}

export function shouldShowPageSubtitle(subtitle: string | undefined, pageTitle: string): boolean {
  const s = (subtitle || '').trim();
  if (!s) return false;
  if (s.toLowerCase() === pageTitle.trim().toLowerCase()) return false;
  if (s.length < 12) return false;
  return true;
}
