import { Box, Paper, Typography } from '@mui/material';
import type { TooltipProps } from 'recharts';
import { tokens } from '../../../theme/tokens';

export const CHART_COLORS = [...tokens.chartPalette];

export const chartAxisTick = { fontSize: 11, fill: tokens.muted };
export const chartGridStroke = tokens.border;

export function formatAxisLabel(value: unknown): string {
  const raw = String(value ?? '');
  if (/^\d{4}-\d{2}-\d{2}/.test(raw)) {
    const d = new Date(raw);
    if (!Number.isNaN(d.getTime())) {
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    }
  }
  if (raw.length > 14) return `${raw.slice(0, 12)}…`;
  return raw;
}

export function formatTooltipValue(value: unknown): string {
  if (typeof value === 'number') {
    if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
    if (!Number.isInteger(value)) return value.toFixed(1);
    return value.toLocaleString();
  }
  return String(value ?? '');
}

export function ChartTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null;
  return (
    <Paper
      elevation={4}
      sx={{
        px: 1.5,
        py: 1,
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'divider',
        minWidth: 140,
      }}
    >
      <Typography variant="caption" color="text.secondary" fontWeight={600} display="block" mb={0.5}>
        {formatAxisLabel(label)}
      </Typography>
      {payload.map((entry) => (
        <Box key={String(entry.dataKey)} display="flex" alignItems="center" justifyContent="space-between" gap={2}>
          <Typography variant="caption" sx={{ color: entry.color }}>
            {entry.name}
          </Typography>
          <Typography variant="caption" fontWeight={700}>
            {formatTooltipValue(entry.value)}
          </Typography>
        </Box>
      ))}
    </Paper>
  );
}
