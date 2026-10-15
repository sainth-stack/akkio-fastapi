import { Box, Grid } from '@mui/material';
import StatCard, { StatCardProps } from './StatCard';

export interface KpiRowItem extends Omit<StatCardProps, 'sx'> {
  valueKey: string;
}

export interface KpiRowProps {
  items: KpiRowItem[];
  values: Record<string, string | number>;
  meta?: Record<string, string | number>;
}

export default function KpiRow({ items, values, meta = {} }: KpiRowProps) {
  const count = items.length;
  const md = count <= 4 ? 3 : count <= 6 ? 2 : 2;

  return (
    <Grid container spacing={2} sx={{ mb: 3 }}>
      {items.map((item) => {
        const trendKey = item.trendKey || `${item.valueKey}_trend_pct`;
        const targetKey = item.targetKey || `${item.valueKey}_target`;
        const trendRaw = meta[trendKey];
        const target = meta[targetKey];
        const trend =
          typeof trendRaw === 'number'
            ? { value: trendRaw, label: 'vs prior period' }
            : undefined;

        return (
          <Grid item xs={6} sm={4} md={md} key={item.valueKey}>
            <StatCard
              label={item.label}
              value={values[item.valueKey] ?? '—'}
              trend={trend}
              target={target != null ? String(target) : undefined}
              color={item.color}
              sx={{ height: '100%' }}
            />
          </Grid>
        );
      })}
    </Grid>
  );
}
