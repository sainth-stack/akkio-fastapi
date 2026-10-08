"""Inject universal Sample Report Template UI into generated fullstack apps."""
import json
from typing import Any, Dict


def sample_report_template_component_tsx() -> str:
    return r'''import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Box, Button, Card, CardContent, Chip, Stack, Table, TableBody, TableCell,
  TableHead, TableRow, Typography,
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import { apiFetch } from '../api/client';

export type ReportSection =
  | { type: 'table'; title: string; columns: string[]; rows: (string | number | null)[][] }
  | { type: 'callout'; title: string; body: string };

export type SampleReportPayload = {
  domain?: string;
  organization?: string;
  hint?: string;
  templates?: { id: string; title: string; subtitle?: string }[];
  template_id?: string;
  title?: string;
  subtitle?: string;
  report_generated_at?: string;
  kpis?: { label: string; value: string }[];
  sections?: ReportSection[];
  exports?: { label: string; path: string; filename: string }[];
};

function downloadCsv(filename: string, rows: Record<string, unknown>[]) {
  if (!rows.length) return;
  const headers = Object.keys(rows[0]);
  const csv = [headers.join(','), ...rows.map((r) => headers.map((h) => JSON.stringify(r[h] ?? '')).join(','))].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

async function fetchTemplate(templateId: string): Promise<SampleReportPayload> {
  const q = templateId ? `?template_id=${encodeURIComponent(templateId)}` : '';
  return apiFetch<SampleReportPayload>(`/api/reports/template${q}`);
}

export default function SampleReportTemplate({ pageTitle = 'Reports' }: { pageTitle?: string }) {
  const [activeId, setActiveId] = useState<string>('');

  const { data, isLoading } = useQuery({
    queryKey: ['sample-report-template', activeId],
    queryFn: () => fetchTemplate(activeId),
  });

  const templates = data?.templates || [];
  const resolvedId = activeId || data?.template_id || templates[0]?.id || '';
  const activeMeta = useMemo(
    () => templates.find((t) => t.id === resolvedId) || templates[0],
    [templates, resolvedId],
  );

  const kpis = data?.kpis || [];
  const sections = data?.sections || [];
  const generatedAt = data?.report_generated_at;

  return (
    <Box sx={{ p: 3 }}>
      <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" alignItems="flex-start" spacing={2} sx={{ mb: 2 }}>
        <Box>
          <Typography variant="h4" fontWeight={800}>{pageTitle}</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            {data?.hint || 'Sample report templates with demonstration data.'}
          </Typography>
        </Box>
        <Chip label="Sample template" color="primary" variant="outlined" />
      </Stack>

      <Stack direction="row" flexWrap="wrap" gap={1} sx={{ mb: 2 }}>
        {templates.map((t) => (
          <Button
            key={t.id}
            size="small"
            variant={resolvedId === t.id ? 'contained' : 'outlined'}
            onClick={() => setActiveId(t.id)}
          >
            {t.title}
          </Button>
        ))}
      </Stack>

      {(data?.exports || []).length > 0 && (
        <Stack direction="row" flexWrap="wrap" gap={1} sx={{ mb: 2 }}>
          {(data?.exports || []).map((ex) => (
            <Button
              key={ex.filename}
              variant="outlined"
              size="small"
              startIcon={<DownloadIcon />}
              onClick={async () => {
                const payload = await apiFetch<Record<string, unknown> | { items: Record<string, unknown>[] }>(ex.path);
                const items = Array.isArray((payload as { items?: unknown }).items)
                  ? (payload as { items: Record<string, unknown>[] }).items
                  : [payload as Record<string, unknown>];
                downloadCsv(ex.filename, items);
              }}
            >
              {ex.label}
            </Button>
          ))}
        </Stack>
      )}

      <Card variant="outlined" sx={{ borderStyle: 'dashed' }}>
        <Box sx={{ bgcolor: 'primary.main', color: 'primary.contrastText', px: 2, py: 0.75, typography: 'caption', fontWeight: 700 }}>
          Sample template — demonstration data only
        </Box>
        <CardContent>
          <Stack direction="row" justifyContent="space-between" alignItems="flex-start" sx={{ mb: 2 }}>
            <Box>
              <Typography variant="overline" color="text.secondary">{data?.organization}</Typography>
              <Typography variant="h5" fontWeight={700}>{activeMeta?.title || data?.title}</Typography>
              <Typography variant="caption" color="text.secondary">
                Template <code>{resolvedId}</code>
                {generatedAt ? ` · Generated ${new Date(generatedAt).toLocaleString()}` : ''}
              </Typography>
            </Box>
            <Chip label="SAMPLE" size="small" color="secondary" />
          </Stack>

          <Stack direction="row" flexWrap="wrap" gap={2} sx={{ mb: 3 }} aria-busy={isLoading}>
            {kpis.map((k) => (
              <Box key={k.label} sx={{ minWidth: 120 }}>
                <Typography variant="caption" color="text.secondary">{k.label}</Typography>
                <Typography variant="h6" fontWeight={700}>{k.value}</Typography>
              </Box>
            ))}
          </Stack>

          {sections.map((section, idx) => {
            if (section.type === 'callout') {
              return (
                <Box key={idx} sx={{ mb: 2, p: 2, bgcolor: 'action.hover', borderRadius: 1 }}>
                  <Typography variant="subtitle2" gutterBottom>{section.title}</Typography>
                  <Typography variant="body2">{section.body}</Typography>
                </Box>
              );
            }
            return (
              <Box key={idx} sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>{section.title}</Typography>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      {section.columns.map((c) => <TableCell key={c}>{c}</TableCell>)}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {section.rows.map((row, ri) => (
                      <TableRow key={ri}>
                        {row.map((cell, ci) => <TableCell key={ci}>{cell}</TableCell>)}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Box>
            );
          })}
        </CardContent>
      </Card>
    </Box>
  );
}
'''


def reports_page_tsx() -> str:
    return r'''import SampleReportTemplate from '../components/SampleReportTemplate';

export default function ReportsPage() {
  return <SampleReportTemplate pageTitle="Reports" />;
}
'''


def ensure_sample_reports_ui(files: dict) -> dict:
    """Add SampleReportTemplate component and Reports page for every app."""
    out = dict(files or {})
    out["frontend/src/components/SampleReportTemplate.tsx"] = sample_report_template_component_tsx()
    out["frontend/src/pages/ReportsPage.tsx"] = reports_page_tsx()
    out = _ensure_reports_route(out)
    out = _patch_mock_for_report_template(out)
    return out


def _ensure_reports_route(files: dict) -> dict:
    app_path = "frontend/src/App.tsx"
    app = files.get(app_path, "")
    if not app:
        return files
    if "ReportsPage" in app and "/reports" in app:
        return files
    if "ReportsPage" not in app:
        if "from './pages/" in app:
            app = app.replace(
                "import DashboardPage",
                "import ReportsPage from './pages/ReportsPage';\nimport DashboardPage",
                1,
            )
        else:
            app = "import ReportsPage from './pages/ReportsPage';\n" + app
    if "/reports" not in app and "<Route" in app:
        import re

        route_snippet = '<Route path="reports" element={<ReportsPage />} />'
        alt = '<Route path="/reports" element={<ReportsPage />} />'
        if 'path="*"' in app:
            app = app.replace('<Route path="*"', f'{route_snippet}\n        <Route path="*"', 1)
        elif re.search(r'<Route[^>]+index', app):
            app = re.sub(r"(<Route[^>]+index[^/]*/>\s*)", r"\1        " + route_snippet + "\n        ", app, count=1)
        else:
            app = app.replace("</Routes>", f"        {alt}\n      </Routes>", 1)
    files[app_path] = app
    layout = files.get("frontend/src/layout/AppLayout.tsx", "")
    if layout and "Reports" not in layout and "nav" in layout.lower():
        if "Dashboard" in layout:
            layout = layout.replace(
                "{ label: 'Dashboard'",
                "{ label: 'Reports', path: '/reports' },\n    { label: 'Dashboard'",
                1,
            )
            files["frontend/src/layout/AppLayout.tsx"] = layout
    return files


def _patch_mock_for_report_template(files: dict) -> dict:
    mock_path = "frontend/src/api/mock.ts"
    mock = files.get(mock_path, "")
    if not mock or "reports/template" in mock:
        return files
    seed: Dict[str, Any] = {}
    try:
        seed = json.loads(files.get("backend/seed_data.json") or "{}")
    except json.JSONDecodeError:
        seed = {}
    handler = """
    if (clean === 'reports/templates') {
      const tpl = SEED['reports/templates'] as { templates?: unknown[] } | undefined;
      return ok(tpl || { templates: [] }) as T;
    }
    if (clean.startsWith('reports/template')) {
      const tpl = SEED['reports/template'];
      if (tpl) return ok(tpl) as T;
    }
"""
    if "const SEED" in mock:
        mock = mock.replace(
            "  if (method === 'GET') {",
            "  if (method === 'GET') {\n" + handler,
            1,
        )
        files[mock_path] = mock
        return files

    doc = seed.get("reports/template") or {}
    templates_meta = seed.get("reports/templates") or {"templates": doc.get("templates") or []}
    doc_js = json.dumps(doc, ensure_ascii=False)
    meta_js = json.dumps(templates_meta, ensure_ascii=False)
    inline_handler = f"""
  if (method === 'GET' && clean === 'reports/templates') {{
    return ok({meta_js}) as T;
  }}
  if (method === 'GET' && clean.startsWith('reports/template')) {{
    return ok({doc_js}) as T;
  }}
"""
    anchor = "  return ok({ ok: true, mocked: true, path, method }) as T;"
    if anchor in mock:
        mock = mock.replace(anchor, inline_handler + anchor, 1)
        files[mock_path] = mock
    return files
