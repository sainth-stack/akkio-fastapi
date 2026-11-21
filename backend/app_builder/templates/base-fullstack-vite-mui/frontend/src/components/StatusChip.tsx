import { Chip } from '@mui/material';

const COLOR: Record<string, 'default' | 'success' | 'warning' | 'error' | 'info'> = {
  RELEASED: 'success',
  PASS: 'success',
  HOLD: 'warning',
  WARNING: 'warning',
  PENDING: 'info',
  REJECTED: 'error',
  FAIL: 'error',
  ACTIVE: 'success',
};

export default function StatusChip({ status }: { status: string }) {
  const key = (status || '').toUpperCase();
  return <Chip size="small" label={key || 'UNKNOWN'} color={COLOR[key] || 'default'} />;
}
