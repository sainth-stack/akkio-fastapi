import { Button, Paper, Stack } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DownloadIcon from '@mui/icons-material/Download';

export type ActionDef = {
  id: string;
  label: string;
  actionType?: 'primary' | 'exportCsv' | 'exportPdf' | 'create';
};

function downloadCsv(filename: string, rows: Record<string, unknown>[]) {
  if (!rows.length) return;
  const keys = Object.keys(rows[0]);
  const esc = (v: unknown) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const lines = [keys.join(','), ...rows.map((r) => keys.map((k) => esc(r[k])).join(','))];
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
  exportRows,
  exportFilename = 'export.csv',
}: {
  actions: ActionDef[];
  exportRows?: Record<string, unknown>[];
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
              ? () => downloadCsv(exportFilename, exportRows || [])
              : type === 'exportPdf'
                ? () => downloadCsv(exportFilename.replace('.csv', '.csv'), exportRows || [])
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
