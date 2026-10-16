import type { AppDefinition } from './useAppDefinition';
import { getMockSlice } from './useAppDefinition';

type Block = Record<string, unknown>;

const KPI_COLORS = ['primary', 'success', 'warning', 'info', 'error'] as const;

const PLACEHOLDER_FIELD = /^(col_?\d+|field_?\d+|column_?\d+)$/i;
const PLACEHOLDER_HEADER = /^(field|column|col)\s*\d+$/i;

const MAINTENANCE_TABLE_COLUMNS = [
  { field: 'work_order_id', header: 'Work order' },
  { field: 'machine', header: 'Machine' },
  { field: 'production_line', header: 'Production line' },
  { field: 'equipment', header: 'Equipment' },
  { field: 'shift', header: 'Shift' },
  { field: 'event_type', header: 'Event type' },
  { field: 'root_cause', header: 'Root cause' },
  { field: 'severity', header: 'Severity' },
  { field: 'status', header: 'Status' },
  { field: 'downtime_hours', header: 'Downtime (h)', align: 'right' as const },
  { field: 'occurred_at', header: 'Occurred at' },
];

function columnsLookPlaceholder(cols: Array<Record<string, unknown>>): boolean {
  return cols.every((col) => {
    const field = String(col.field || col.dataKey || '').trim();
    const header = String(col.title || col.header || '').replace(/\*+/g, '').trim();
    return PLACEHOLDER_FIELD.test(field) || PLACEHOLDER_HEADER.test(header);
  });
}

export function humanizeField(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Expand compact LLM blocks (dataKey-only) into full renderer shape. */
export function normalizeBlock(
  block: Block,
  definition: AppDefinition,
  pageId?: string,
): Block {
  const type = block.type as string;
  const next = { ...block };

  if (type === 'kpiRow') {
    const items = block.items as Array<{ label: string; valueKey: string; color?: string }> | undefined;
    if (items?.length) return next;

    let dataKey = block.dataKey as string | undefined;
    if (!dataKey && pageId) dataKey = `kpis_${pageId}`;
    if (!dataKey) return next;

    const slice = getMockSlice(definition, dataKey);
    if (!slice || typeof slice !== 'object' || Array.isArray(slice)) return next;

    next.items = Object.entries(slice as Record<string, unknown>).map(([valueKey], i) => ({
      label: humanizeField(valueKey),
      valueKey,
      color: KPI_COLORS[i % KPI_COLORS.length],
    }));
    return next;
  }

  if (type === 'lineChart' || type === 'barChart') {
    const data = (getMockSlice(definition, block.dataKey as string) as Record<string, unknown>[]) || [];
    const series = block.series as Array<{ dataKey: string; name?: string }> | undefined;
    if (series?.length || !data.length) return next;

    const xKey = (block.xAxisKey as string) || 'label';
    const sample = data[0];
    const xResolved = sample && xKey in sample
      ? xKey
      : ['period', 'label', 'name', 'week', 'shift', 'hour', 'date'].find((k) => sample && k in sample) || xKey;

    const dataKeys = Object.keys(sample).filter((k) => k !== xResolved);
    next.xAxisKey = xResolved;
    next.series = dataKeys.map((dataKey) => ({
      dataKey,
      name: humanizeField(dataKey),
    }));
    return next;
  }

  if (type === 'dataTable') {
    const rawCols = block.columns as Array<Record<string, unknown>> | undefined;
    if (rawCols?.length && columnsLookPlaceholder(rawCols)) {
      next.columns = MAINTENANCE_TABLE_COLUMNS;
      if (String(block.title || '').toLowerCase().includes('records')) {
        next.title = 'Recent maintenance events';
      }
      return next;
    }
    if (rawCols?.length) {
      next.columns = rawCols.map((col, i) => {
        const title = String(col.title || col.header || '').replace(/\*+/g, '').trim();
        let field = String(col.field || col.dataKey || '').trim();
        if (!field && title) {
          field = title.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
        }
        const header = title || humanizeField(field || `column_${i}`);
        if (!field) field = `column_${i}`;
        return {
          field,
          header,
          align: col.align as 'left' | 'center' | 'right' | undefined,
          variant: col.variant,
        };
      });
      return next;
    }

    const rows = (getMockSlice(definition, block.dataKey as string) as Record<string, unknown>[]) || [];
    if (!rows.length) return next;

    const fields = Object.keys(rows[0]);
    next.columns = fields.map((field) => ({
      field,
      header: humanizeField(field),
      align: typeof rows[0][field] === 'number' ? 'right' : undefined,
    }));
    next.searchable = block.searchable ?? true;
    return next;
  }

  return next;
}
