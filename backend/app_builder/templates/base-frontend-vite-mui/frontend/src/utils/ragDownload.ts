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
@media print{body{padding:12px} pre{background:#fff;border:1px solid #e2e8f0}}
</style></head><body>
<h1>RAG data export — ${escapeHtml(appName)}</h1>
<p>Knowledge, manufacturing sales, and the policies from the app prompt. Open this file in a browser and use Print → Save as PDF if needed.</p>
<pre>${escapeHtml(context)}</pre>
</body></html>`;

  const safeName = String(appName).replace(/[^\w.-]+/g, '-').replace(/-+/g, '-').slice(0, 60) || 'app';
  const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  try {
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${safeName}-rag-data.html`;
    anchor.rel = 'noopener';
    anchor.style.display = 'none';
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
  } finally {
    URL.revokeObjectURL(url);
  }
}
