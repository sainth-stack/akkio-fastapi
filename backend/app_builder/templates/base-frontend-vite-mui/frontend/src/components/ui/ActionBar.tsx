import { Button, Paper, Stack } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import type { ExportSheet } from '../../renderer/buildExportSheets';

export type ActionDef = {
  id: string;
  label: string;
  actionType?: 'primary' | 'exportCsv' | 'exportPdf' | 'create';
};

const EXPORT_TYPES = new Set(['exportCsv', 'exportPdf', 'export', 'download']);

function isExportAction(a: ActionDef): boolean {
  const type = (a.actionType || '').toLowerCase();
  if (EXPORT_TYPES.has(type)) return true;
  return /\b(export|download|pdf|csv)\b/i.test(a.label || '');
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
  const visible = actions.filter((a) => !isExportAction(a));
  if (!visible.length) return null;

  return (
    <Paper sx={{ p: 2, mb: 3, borderRadius: 3 }}>
      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
        {visible.map((a) => {
          const type = a.actionType || 'primary';
          const isPrimary = type === 'primary' || type === 'create';
          const onClick = undefined;
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
