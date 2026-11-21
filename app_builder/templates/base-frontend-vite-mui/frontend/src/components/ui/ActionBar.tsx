import { useState } from 'react';
import { Alert, Button, Dialog, DialogActions, DialogContent, DialogTitle, Paper, Stack, TextField } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DownloadIcon from '@mui/icons-material/Download';
import type { ExportSheet } from '../../renderer/buildExportSheets';
import { useAppDefinition } from '../../renderer/useAppDefinition';
import { getAkkioAccessToken, getAkkioApiBase } from '../../renderer/akkioApi';

export type ActionDef = {
  id: string;
  label: string;
  actionType?: 'primary' | 'exportCsv' | 'exportPdf' | 'create';
  dataKey?: string;
};

export default function ActionBar({
  actions,
  exportSheets,
  exportFilename = 'export.csv',
}: {
  actions: ActionDef[];
  exportSheets?: ExportSheet[];
  exportFilename?: string;
}) {
  const { definition, applyDefinition } = useAppDefinition();
  const [active, setActive] = useState<ActionDef | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const table = definition?.pages.flatMap((p) => p.blocks).find((b) => b.type === 'dataTable' && b.dataKey === active?.dataKey);
  const columns = (table?.columns || []) as Array<{ field: string; header: string; inputType?: string; required?: boolean }>;

  function exportData(action: ActionDef) {
    const sheets = exportSheets || [];
    if (action.actionType === 'exportPdf') {
      window.print();
      return;
    }
    const cell = (value: unknown) => {
      let text = String(value ?? '');
      if (/^[=+\-@]/.test(text)) text = `'${text}`;
      return `"${text.replace(/"/g, '""')}"`;
    };
    const csv = sheets.map((sheet) => [sheet.columns.map((c) => cell(c.header)).join(','), ...sheet.rows.map((row) => sheet.columns.map((c) => cell(row[c.field])).join(','))].join('\r\n')).join('\r\n\r\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = exportFilename;
    link.click();
    URL.revokeObjectURL(url);
  }

  async function save() {
    if (!definition || !active?.dataKey) return;
    setSaving(true);
    setError('');
    try {
      const record = Object.fromEntries(columns.map((c) => [c.field, c.inputType === 'number' ? (values[c.field] === '' || values[c.field] == null ? null : Number(values[c.field])) : values[c.field] || '']));
      const token = getAkkioAccessToken();
      const response = await fetch(`${getAkkioApiBase()}/data/records/${encodeURIComponent(active.dataKey)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        body: JSON.stringify({ record }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(String(payload.detail || 'Could not save record'));
      applyDefinition(payload.definition);
      setActive(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not save record');
    } finally {
      setSaving(false);
    }
  }

  if (!actions.length) return null;

  return (
    <Paper sx={{ p: 2, mb: 3, borderRadius: 3 }}>
      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
        {actions.map((a) => {
          const type = a.actionType || 'primary';
          const isPrimary = type === 'primary' || type === 'create';
          const onClick = () => {
            if (a.actionType === 'exportCsv' || a.actionType === 'exportPdf') exportData(a);
            else { setValues({}); setError(''); setActive(a); }
          };
          return (
            <Button
              key={a.id}
              variant={isPrimary ? 'contained' : 'outlined'}
              startIcon={isPrimary ? <AddIcon /> : <DownloadIcon />}
              onClick={onClick}
              disabled={isPrimary && (!a.dataKey || !getAkkioApiBase())}
            >
              {a.label}
            </Button>
          );
        })}
      </Stack>
      <Dialog open={Boolean(active)} onClose={() => !saving && setActive(null)} fullWidth maxWidth="sm">
        <DialogTitle>{active?.label}</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ pt: 1 }}>
            {error ? <Alert severity="error">{error}</Alert> : null}
            {columns.map((c) => <TextField key={c.field} label={c.header} type={c.inputType || 'text'} required={c.required} value={values[c.field] || ''} onChange={(e) => setValues((previous) => ({ ...previous, [c.field]: e.target.value }))} InputLabelProps={{ shrink: true }} />)}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setActive(null)} disabled={saving}>Cancel</Button>
          <Button onClick={() => void save()} disabled={saving || !columns.length || columns.some((c) => c.required && !values[c.field])}>Save</Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
}
