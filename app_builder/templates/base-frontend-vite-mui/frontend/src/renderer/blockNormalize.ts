import type { AppDefinition } from './useAppDefinition';
import { getMockSlice } from './useAppDefinition';

type Block = Record<string, unknown>;

const KPI_COLORS = ['primary', 'success', 'warning', 'info', 'error'] as const;

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
    const columns = block.columns as Array<{ field: string; header: string }> | undefined;
    if (columns?.length) return next;

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
