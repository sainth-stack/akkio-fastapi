import React from 'react';
import { Chip, ChipProps } from '@mui/material';

type StatusColor = ChipProps['color'];

export interface StatusChipProps {
  status: string;
  variant?: 'filled' | 'outlined';
  size?: 'small' | 'medium';
  /** Override the default status→color mapping. */
  statusMap?: Record<string, { color: StatusColor; label?: string }>;
}

const DEFAULT_STATUS_MAP: Record<string, { color: StatusColor; label?: string }> = {
  active:     { color: 'success' },
  enabled:    { color: 'success' },
  open:       { color: 'success' },
  approved:   { color: 'success' },
  complete:   { color: 'success' },
  completed:  { color: 'success' },
  passed:     { color: 'success' },
  inactive:   { color: 'default' },
  disabled:   { color: 'default' },
  closed:     { color: 'default' },
  archived:   { color: 'default' },
  pending:    { color: 'warning' },
  review:     { color: 'warning' },
  draft:      { color: 'warning' },
  'in review':{ color: 'warning' },
  processing: { color: 'info' },
  running:    { color: 'info' },
  'in progress': { color: 'info' },
  failed:     { color: 'error' },
  error:      { color: 'error' },
  rejected:   { color: 'error' },
  cancelled:  { color: 'error' },
  canceled:   { color: 'error' },
};

/**
 * StatusChip — Chip whose color is determined automatically from the status
 * string.  Pass `statusMap` to override defaults.
 */
export default function StatusChip({
  status,
  variant = 'filled',
  size = 'small',
  statusMap,
}: StatusChipProps) {
  const map = statusMap ?? DEFAULT_STATUS_MAP;
  const entry = map[status?.toLowerCase()] ?? { color: 'default' as StatusColor };

  return (
    <Chip
      label={entry.label ?? (status ? status.charAt(0).toUpperCase() + status.slice(1) : '—')}
      color={entry.color}
      variant={variant}
      size={size}
    />
  );
}
