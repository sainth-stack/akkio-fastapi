import { Box, Paper, Stack, Typography } from '@mui/material';
import {
  BarChart,
  DataTable,
  LineChart,
  PageHeader,
  PieChart,
  StatCard,
} from '../components/ui';
import type { AppDefinition } from './useAppDefinition';
import { dataBag, getMockSlice } from './useAppDefinition';
import { normalizeBlock } from './blockNormalize';
import { resolveRowKey, resolveXAxisKey } from './chartUtils';
import { tokens } from '../theme/tokens';

type Block = Record<string, unknown>;

function kpiValues(
  definition: AppDefinition,
  block: Block,
  pageId?: string,
): Record<string, string | number> {
  const items = (block.items as Array<{ valueKey: string }> | undefined) ?? [];
  const keys = items.map((i) => i.valueKey);
  const bag = dataBag(definition);
  const merged: Record<string, string | number> = {};

  const bundleKey = (block.dataKey as string) || (pageId ? `kpis_${pageId}` : '');
  if (bundleKey && bag[bundleKey] && typeof bag[bundleKey] === 'object' && !Array.isArray(bag[bundleKey])) {
    Object.assign(merged, bag[bundleKey] as Record<string, string | number>);
  }
  if (pageId && bag[`kpis_${pageId}`] && typeof bag[`kpis_${pageId}`] === 'object' && !Array.isArray(bag[`kpis_${pageId}`])) {
    Object.assign(merged, bag[`kpis_${pageId}`] as Record<string, string | number>);
  }
  for (const [k, v] of Object.entries(bag)) {
    if (k.startsWith('kpis_') && v && typeof v === 'object' && !Array.isArray(v)) {
      Object.assign(merged, v as Record<string, string | number>);
    }
  }

  const out: Record<string, string | number> = {};
  for (const k of keys) {
    const flat = bag[k];
    out[k] = merged[k] ?? (typeof flat === 'string' || typeof flat === 'number' ? flat : '—');
  }
  return out;
}

export default function BlockRenderer({
  block,
  definition,
  pageId,
}: {
  block: Block;
  definition: AppDefinition;
  pageId?: string;
}) {
  const normalized = normalizeBlock(block, definition, pageId);
  const type = normalized.type as string;

  if (type === 'kpiRow') {
    const items = (normalized.items as Array<{ label: string; valueKey: string; color?: string }>) || [];
    const values = kpiValues(definition, normalized, pageId);
    if (!items.length) {
      return (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          No KPI configuration for this row.
        </Typography>
      );
    }
    return (
      <Stack direction="row" flexWrap="wrap" gap={2} sx={{ mb: 3 }}>
        {items.map((item) => (
          <Box key={item.valueKey} sx={{ flex: '1 1 220px', minWidth: 200, maxWidth: 320 }}>
            <StatCard
              label={item.label}
              value={values[item.valueKey] ?? '—'}
              color={(item.color as 'primary') || 'primary'}
            />
          </Box>
        ))}
      </Stack>
    );
  }

  if (type === 'lineChart' || type === 'barChart') {
    const data = (getMockSlice(definition, normalized.dataKey as string) as Record<string, unknown>[]) || [];
    const series = (normalized.series as Array<{ dataKey: string; name?: string; color?: string }>) || [];
    const xAxisKey = resolveXAxisKey(data, (normalized.xAxisKey as string) || 'name');
    const Chart = type === 'lineChart' ? LineChart : BarChart;
    return (
      <Paper sx={{ p: 2, mb: 3, borderRadius: 3 }}>
        {normalized.title ? (
          <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>{normalized.title as string}</Typography>
        ) : null}
        {data.length === 0 || !series.length ? (
          <Typography variant="body2" color="text.secondary">No chart data</Typography>
        ) : (
          <Chart
            data={data}
            series={series.map((s, i) => ({
              dataKey: s.dataKey,
              name: s.name || s.dataKey,
              color: s.color || tokens.chartPalette[i % tokens.chartPalette.length],
            }))}
            xAxisKey={xAxisKey}
            height={(normalized.height as number) || 300}
          />
        )}
      </Paper>
    );
  }

  if (type === 'pieChart') {
    const data = (getMockSlice(definition, normalized.dataKey as string) as Array<{ name: string; value: number }>) || [];
    return (
      <Paper sx={{ p: 2, mb: 3, borderRadius: 3 }}>
        {normalized.title ? (
          <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>{normalized.title as string}</Typography>
        ) : null}
        <PieChart data={data} height={(normalized.height as number) || 300} />
      </Paper>
    );
  }

  if (type === 'dataTable') {
    const rows = (getMockSlice(definition, normalized.dataKey as string) as Record<string, unknown>[]) || [];
    const columns = (normalized.columns as Array<{ field: string; header: string; align?: string }>) || [];
    const rowKey = resolveRowKey(rows, columns, normalized.rowKey as string | undefined);
    return (
      <Box sx={{ mb: 3 }}>
        {normalized.title ? <PageHeader title={normalized.title as string} /> : null}
        <DataTable
          columns={columns.map((c) => ({
            field: c.field as keyof Record<string, unknown>,
            header: c.header,
            align: c.align as 'left' | 'center' | 'right' | undefined,
          }))}
          rows={rows}
          rowKey={rowKey as keyof Record<string, unknown> | undefined}
          searchable={Boolean(normalized.searchable ?? true)}
          emptyMessage={(normalized.emptyMessage as string) || 'No records found'}
        />
      </Box>
    );
  }

  if (type === 'text') {
    return (
      <Typography variant="body1" sx={{ mb: 2 }}>{normalized.content as string}</Typography>
    );
  }

  return null;
}
