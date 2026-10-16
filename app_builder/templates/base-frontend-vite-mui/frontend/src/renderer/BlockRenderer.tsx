import { useMemo, useState } from 'react';
import { Box, Paper, Typography } from '@mui/material';
import {
  ActionBar,
  BarChart,
  DataTable,
  FilterBar,
  KpiRow,
  LineChart,
  PieChart,
  SessionDetailModal,
  StatusChip,
} from '../components/ui';
import { cleanDisplayTitle } from './displayUtils';
import type { AppDefinition } from './useAppDefinition';
import { buildPageExportSheets } from './buildExportSheets';
import { dataBag, getDataSlice } from './useAppDefinition';
import { normalizeBlock } from './blockNormalize';
import { resolveRowKey, resolveXAxisKey } from './chartUtils';
import { usePageFilters } from './PageFilterContext';
import { useChartPalette } from '../theme/useChartPalette';
import { repairTableRows, rowsLookPlaceholder } from './tableDataRepair';

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

const chartPaperSx = {
  standalone: {
    p: 2.5,
    mb: 3,
    borderRadius: 2.5,
    border: '1px solid',
    borderColor: 'divider',
    bgcolor: 'background.paper',
    boxShadow: '0 1px 2px rgba(15,23,42,0.04)',
  },
  grid: {
    p: 2.5,
    mb: 0,
    height: '100%',
    borderRadius: 2.5,
    border: '1px solid',
    borderColor: 'divider',
    bgcolor: 'background.paper',
    boxShadow: '0 1px 2px rgba(15,23,42,0.04)',
  },
} as const;

function ChartCardTitle({ title }: { title: string }) {
  return (
    <Typography variant="subtitle2" fontWeight={600} color="text.primary" sx={{ mb: 2, letterSpacing: '-0.01em' }}>
      {title}
    </Typography>
  );
}

export default function BlockRenderer({
  block,
  definition,
  pageId,
  pageTitle,
  pageBlocks,
  layout = 'standalone',
}: {
  block: Block;
  definition: AppDefinition;
  pageId?: string;
  pageTitle?: string;
  pageBlocks?: Block[];
  layout?: 'standalone' | 'grid';
}) {
  const chartPalette = useChartPalette();
  const normalized = normalizeBlock(block, definition, pageId);
  const displayTitle = normalized.title
    ? cleanDisplayTitle(String(normalized.title), pageTitle)
    : '';
  const type = normalized.type as string;
  const { filters } = usePageFilters();
  const [detailRow, setDetailRow] = useState<Record<string, unknown> | null>(null);

  const exportSheets = useMemo(
    () => buildPageExportSheets(pageBlocks, definition),
    [definition, pageBlocks],
  );

  if (type === 'filterBar') {
    const filterDefs = (normalized.filters as Array<Record<string, unknown>>) || [];
    return <FilterBar filters={filterDefs as never} definition={definition} />;
  }

  if (type === 'actionBar') {
    const actions = (normalized.actions as Array<Record<string, unknown>>) || [];
    return (
      <ActionBar
        actions={actions as never}
        exportSheets={exportSheets}
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
    return <KpiRow items={items} values={values} meta={meta} />;
  }

  if (type === 'lineChart' || type === 'barChart') {
    const data = (getDataSlice(definition, normalized.dataKey as string) as Record<string, unknown>[]) || [];
    const series = (normalized.series as Array<{ dataKey: string; name?: string; color?: string }>) || [];
    const xAxisKey = resolveXAxisKey(data, (normalized.xAxisKey as string) || 'period');
    const Chart = type === 'lineChart' ? LineChart : BarChart;
    const chartHeight = (normalized.height as number) || (layout === 'grid' ? 280 : 320);
    const barLayout =
      type === 'barChart' && /pareto|defect|category|distribution|by\s+line|comparison/i.test(displayTitle)
        ? 'vertical'
        : 'horizontal';
    return (
      <Paper sx={chartPaperSx[layout]} elevation={0}>
        {displayTitle ? <ChartCardTitle title={displayTitle} /> : null}
        {data.length === 0 || !series.length ? (
          <Typography variant="body2" color="text.secondary">No chart data</Typography>
        ) : (
          <Chart
            data={data}
            series={series.map((s, i) => ({
              dataKey: s.dataKey,
              name: s.name || s.dataKey,
              color: s.color || chartPalette[i % chartPalette.length],
            }))}
            xAxisKey={xAxisKey}
            height={chartHeight}
            layout={type === 'barChart' ? barLayout : undefined}
            rich={Boolean(normalized.rich ?? type === 'lineChart')}
          />
        )}
      </Paper>
    );
  }

  if (type === 'pieChart') {
    const data = (getDataSlice(definition, normalized.dataKey as string) as Array<{ name: string; value: number }>) || [];
    const chartHeight = (normalized.height as number) || (layout === 'grid' ? 280 : 320);
    return (
      <Paper sx={chartPaperSx[layout]} elevation={0}>
        {displayTitle ? <ChartCardTitle title={displayTitle} /> : null}
        <PieChart data={data} height={chartHeight} />
      </Paper>
    );
  }

  if (type === 'dataTable') {
    const rawRows = (getDataSlice(definition, normalized.dataKey as string) as Record<string, unknown>[]) || [];
    const columns = (normalized.columns as Array<{ field: string; header: string; align?: string; variant?: string }>) || [];
    const tableRows = rowsLookPlaceholder(rawRows)
      ? repairTableRows(rawRows, columns)
      : rawRows;
    const rows = filterRows(tableRows, filters);
    const rowKey = resolveRowKey(rows, columns, normalized.rowKey as string | undefined);
    const detailView = Boolean(normalized.detailView);
    return (
      <Box sx={{ mb: 3 }}>
        {displayTitle ? (
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1.5 }}>
            {displayTitle}
          </Typography>
        ) : null}
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
    const body = String(normalized.content ?? '').trim();
    if (!body || normalized.hidden) {
      return null;
    }
    return (
      <Typography variant="body1" sx={{ mb: 2 }}>{body}</Typography>
    );
  }

  return null;
}
