import { Button, Paper, Stack } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DownloadIcon from '@mui/icons-material/Download';
import type { ExportSheet } from '../../renderer/buildExportSheets';

export type ActionDef = {
  id: string;
  label: string;
  actionType?: 'primary' | 'exportCsv' | 'exportPdf' | 'create';
};

function escCsv(v: unknown): string {
  return `"${String(v ?? '').replace(/"/g, '""')}"`;
}

function downloadCsvFromSheets(filename: string, sheets: ExportSheet[]) {
  if (!sheets.length) return;
  const lines: string[] = [];
  for (const sheet of sheets) {
    if (!sheet.rows.length || !sheet.columns.length) continue;
    if (sheet.title) {
      lines.push(`# ${String(sheet.title).replace(/#/g, '')}`);
    }
    const fields = sheet.columns.map((c) => c.field);
    lines.push(sheet.columns.map((c) => escCsv(c.header)).join(','));
    for (const row of sheet.rows) {
      lines.push(fields.map((f) => escCsv(row[f])).join(','));
    }
    lines.push('');
  }
  if (lines.length === 0) return;
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function ActionBar({
  actions,
  exportSheets,
  exportFilename = 'export.csv',
}: {
  actions: ActionDef[];
  exportSheets?: ExportSheet[];
  exportFilename?: string;
}) {
  return (
    <Paper sx={{ p: 2, mb: 3, borderRadius: 3 }}>
      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
        {actions.map((a) => {
          const type = a.actionType || 'primary';
          const isPrimary = type === 'primary' || type === 'create';
          const onClick =
            type === 'exportCsv'
              ? () => downloadCsvFromSheets(exportFilename, exportSheets || [])
              : type === 'exportPdf'
                ? () =>
                    downloadCsvFromSheets(
                      exportFilename.replace(/\.csv$/i, '') + '-report.csv',
                      exportSheets || [],
                    )
                : undefined;
          return (
            <Button
              key={a.id}
              variant={isPrimary ? 'contained' : 'outlined'}
              startIcon={isPrimary ? <AddIcon /> : <DownloadIcon />}
              onClick={onClick}
            >
              {a.label}
            </Button>
          );
        })}
      </Stack>
    </Paper>
  );
}
