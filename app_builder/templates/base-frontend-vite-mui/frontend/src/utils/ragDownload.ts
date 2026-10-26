import type { AppDefinition } from '../renderer/useAppDefinition';
import { getAkkioAccessToken, getAkkioApiBase } from '../renderer/akkioApi';

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

export function buildAppDefinitionPayload(definition: AppDefinition) {
  return {
    meta: definition.meta,
    pages: definition.pages,
    data: definition.data || definition.mock,
    appKind: (definition as { appKind?: string }).appKind,
  };
}

export async function downloadRagDataPdf(
  definition: AppDefinition,
  pageId?: string,
): Promise<void> {
  const apiBase = getAkkioApiBase();
  if (!apiBase) {
    throw new Error('Open this app from the Akkio host (/app/...) to export RAG data.');
  }
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  const token = getAkkioAccessToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(`${apiBase}/chat/rag-export`, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      messages: [{ role: 'user', content: 'export rag data' }],
      page_id: pageId,
      app_definition: buildAppDefinitionPayload(definition),
    }),
  });
  const data = (await res.json()) as { context?: string; appName?: string; detail?: unknown };
  if (!res.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    throw new Error(detail || 'Export failed');
  }
  const appName = data.appName || definition.meta?.appName || 'App';
  const context = data.context || '';
  const html = `<!DOCTYPE html><html><head><title>RAG Data — ${escapeHtml(appName)}</title>
<style>body{font-family:Inter,system-ui,sans-serif;padding:24px;max-width:900px;margin:0 auto}
h1{font-size:20px}pre{white-space:pre-wrap;font-size:12px;line-height:1.5;background:#f8fafc;padding:16px;border-radius:8px}
</style></head><body>
<h1>RAG data export — ${escapeHtml(appName)}</h1>
<p>Grounded context used by the in-app data assistant.</p>
<pre>${escapeHtml(context)}</pre>
</body></html>`;
  const win = window.open('', '_blank');
  if (!win) throw new Error('Allow pop-ups to save the PDF, then use Print → Save as PDF.');
  win.document.write(html);
  win.document.close();
  win.focus();
  setTimeout(() => win.print(), 400);
}
