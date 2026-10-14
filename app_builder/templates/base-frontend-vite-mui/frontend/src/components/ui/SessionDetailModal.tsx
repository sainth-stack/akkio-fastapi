import {
  Dialog,
  DialogContent,
  DialogTitle,
  Divider,
  Stack,
  Typography,
} from '@mui/material';
import StatusChip from './StatusChip';

export default function SessionDetailModal({
  open,
  title,
  row,
  onClose,
}: {
  open: boolean;
  title: string;
  row: Record<string, unknown> | null;
  onClose: () => void;
}) {
  if (!row) return null;
  const entries = Object.entries(row).filter(([k]) => k !== 'actions');

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>{title}</DialogTitle>
      <DialogContent>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Inspection results, defects, and measurements (preview — connect API for live data).
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <Stack spacing={1.5}>
          {entries.map(([key, value]) => (
            <Stack key={key} direction="row" justifyContent="space-between" gap={2}>
              <Typography variant="body2" color="text.secondary" sx={{ textTransform: 'capitalize' }}>
                {key.replace(/_/g, ' ')}
              </Typography>
              {String(key).toLowerCase().includes('status') ? (
                <StatusChip status={String(value)} size="small" />
              ) : (
                <Typography variant="body2" fontWeight={500}>{String(value)}</Typography>
              )}
            </Stack>
          ))}
        </Stack>
      </DialogContent>
    </Dialog>
  );
}
