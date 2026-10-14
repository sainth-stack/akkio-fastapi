import { useMemo, useState } from 'react';
import { Box, Paper, Stack, Typography } from '@mui/material';
import {
  ActionBar,
  BarChart,
  DataTable,
  FilterBar,
  LineChart,
  PageHeader,
  PieChart,
  SessionDetailModal,
  StatCard,
  StatusChip,
} from '../components/ui';
import type { AppDefinition } from './useAppDefinition';
import { dataBag, getDataSlice } from './useAppDefinition';
import { normalizeBlock } from './blockNormalize';
import { resolveRowKey, resolveXAxisKey } from './chartUtils';
import { usePageFilters } from './PageFilterContext';
import { tokens } from '../theme/tokens';

type Block = Record<string, unknown>;

type KpiItem = {
  label: string;
  valueKey: string;
  color?: string;
  trendKey?: string;
  targetKey?: string;
};

function kpiMeta(definition: AppDefinition, pageId?: string, dataKey?: string): Record<string, string | number> {
  const bag = dataBag(definition);
  const keys = [
    dataKey ? `${dataKey}_meta` : '',
    pageId ? `kpis_${pageId}_meta` : '',
  ].filter(Boolean);
  const merged: Record<string, string | number> = {};
  for (const k of keys) {
    const slice = bag[k];
    if (slice && typeof slice === 'object' && !Array.isArray(slice)) {
      Object.assign(merged, slice as Record<string, string | number>);
    }
  }
  return merged;
}

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

  const out: Record<string, string | number> = {};
  for (const k of keys) {
    const flat = bag[k];
    out[k] = merged[k] ?? (typeof flat === 'string' || typeof flat === 'number' ? flat : '—');
  }
  return out;
}

function filterRows(
  rows: Record<string, unknown>[],
  filters: ReturnType<typeof usePageFilters>['filters'],
) {
  let out = rows;
  const q = filters.search.trim().toLowerCase();
  if (q) {
    out = out.filter((row) => Object.values(row).some((v) => String(v).toLowerCase().includes(q)));
  }
  if (filters.status) {
    out = out.filter((row) =>
      Object.entries(row).some(
        ([k, v]) => k.toLowerCase().includes('status') && String(v).toLowerCase().includes(filters.status.toLowerCase()),
      ),
    );
  }
  if (filters.line) {
    out = out.filter((row) =>
      Object.entries(row).some(
        ([k, v]) =>
          (k.toLowerCase().includes('line') || k.toLowerCase().includes('production'))
          && String(v).toLowerCase().includes(filters.line.toLowerCase()),
      ),
    );
  }
  return out;
}

export default function BlockRenderer({
  block,
  definition,
  pageId,
  pageBlocks,
}: {
  block: Block;
  definition: AppDefinition;
  pageId?: string;
  pageBlocks?: Block[];
}) {
  const normalized = normalizeBlock(block, definition, pageId);
  const type = normalized.type as string;
  const { filters } = usePageFilters();
  const [detailRow, setDetailRow] = useState<Record<string, unknown> | null>(null);

  const exportRows = useMemo(() => {
    if (!pageBlocks) return [];
    const all: Record<string, unknown>[] = [];
    for (const b of pageBlocks) {
      if (b.type === 'dataTable' && b.dataKey) {
        const rows = getDataSlice(definition, b.dataKey as string);
        if (Array.isArray(rows)) all.push(...rows);
      }
    }
    return all;
  }, [definition, pageBlocks]);

  if (type === 'filterBar') {
    const filterDefs = (normalized.filters as Array<Record<string, unknown>>) || [];
    return <FilterBar filters={filterDefs as never} definition={definition} />;
  }

  if (type === 'actionBar') {
    const actions = (normalized.actions as Array<Record<string, unknown>>) || [];
    return (
      <ActionBar
        actions={actions as never}
        exportRows={exportRows}
        exportFilename={`${pageId || 'page'}-export.csv`}
      />
    );
  }

  if (type === 'kpiRow') {
    const items = (normalized.items as KpiItem[]) || [];
    const values = kpiValues(definition, normalized, pageId);
    const meta = kpiMeta(definition, pageId, normalized.dataKey as string);
    if (!items.length) {
      return (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          No KPI configuration for this row.
        </Typography>
      );
    }
    return (
      <Stack direction="row" flexWrap="wrap" gap={2} sx={{ mb: 3 }}>
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
            <Box key={item.valueKey} sx={{ flex: '1 1 220px', minWidth: 200, maxWidth: 320 }}>
              <StatCard
                label={item.label}
                value={values[item.valueKey] ?? '—'}
                color={(item.color as 'primary') || 'primary'}
                trend={trend}
              />
              {target ? (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: -1, ml: 0.5 }}>
                  Target: {target}
                </Typography>
              ) : null}
            </Box>
          );
        })}
      </Stack>
    );
  }

  if (type === 'lineChart' || type === 'barChart') {
    const data = (getDataSlice(definition, normalized.dataKey as string) as Record<string, unknown>[]) || [];
    const series = (normalized.series as Array<{ dataKey: string; name?: string; color?: string }>) || [];
    const xAxisKey = resolveXAxisKey(data, (normalized.xAxisKey as string) || 'period');
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
    const data = (getDataSlice(definition, normalized.dataKey as string) as Array<{ name: string; value: number }>) || [];
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
    const rawRows = (getDataSlice(definition, normalized.dataKey as string) as Record<string, unknown>[]) || [];
    const rows = filterRows(rawRows, filters);
    const columns = (normalized.columns as Array<{ field: string; header: string; align?: string; variant?: string }>) || [];
    const rowKey = resolveRowKey(rows, columns, normalized.rowKey as string | undefined);
    const detailView = Boolean(normalized.detailView);
    return (
      <Box sx={{ mb: 3 }}>
        {normalized.title ? <PageHeader title={normalized.title as string} /> : null}
        <DataTable
          columns={columns.map((c) => ({
            field: c.field as keyof Record<string, unknown>,
            header: c.header,
            align: c.align as 'left' | 'center' | 'right' | undefined,
            renderCell:
              c.variant === 'status' || c.field.toLowerCase().includes('status') || c.field.toLowerCase() === 'severity'
                ? (value) => <StatusChip status={String(value)} size="small" />
                : undefined,
          }))}
          rows={rows}
          rowKey={rowKey as keyof Record<string, unknown> | undefined}
          searchable={Boolean(normalized.searchable ?? true)}
          emptyMessage={(normalized.emptyMessage as string) || 'No records found'}
          onRowClick={detailView ? (row) => setDetailRow(row) : undefined}
        />
        {detailView ? (
          <>
            <SessionDetailModal
              open={Boolean(detailRow)}
              title="Inspection session detail"
              row={detailRow}
              onClose={() => setDetailRow(null)}
            />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              Select a row to view inspection results and measurements.
            </Typography>
          </>
        ) : null}
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
