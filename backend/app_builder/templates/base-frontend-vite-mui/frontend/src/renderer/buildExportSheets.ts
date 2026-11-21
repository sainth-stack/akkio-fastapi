import type { AppDefinition } from './useAppDefinition';
import { getDataSlice } from './useAppDefinition';

type Block = Record<string, unknown> & { type?: string };

export type ExportColumn = { field: string; header: string };

export type ExportSheet = {
  title?: string;
  columns: ExportColumn[];
  rows: Record<string, unknown>[];
};

function humanize(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function chartColumns(block: Block): ExportColumn[] {
  const xKey = (block.xAxisKey as string) || 'period';
  const series = (block.series as Array<{ dataKey: string; name?: string }>) || [];
  const cols: ExportColumn[] = [{ field: xKey, header: humanize(xKey) }];
  for (const s of series) {
    if (!s?.dataKey) continue;
    cols.push({ field: s.dataKey, header: s.name || humanize(s.dataKey) });
  }
  return cols;
}

function tableColumns(block: Block): ExportColumn[] {
  const cols = (block.columns as Array<{ field: string; header?: string }>) || [];
  return cols
    .filter((c) => c?.field)
    .map((c) => ({
      field: String(c.field),
      header: String(c.header || humanize(String(c.field))),
    }));
}

function isPlaceholderColumn(cols: ExportColumn[]): boolean {
  if (!cols.length) return true;
  return cols.every(
    (c) =>
      /^field_\d+$/i.test(c.field) ||
      !c.header ||
      c.header === '**' ||
      /^Field \d+$/i.test(c.header),
  );
}

/** Build export sheets from visible page blocks — human column headers, same rows as UI tables/charts. */
export function buildPageExportSheets(
  pageBlocks: Block[] | undefined,
  definition: AppDefinition,
): ExportSheet[] {
  if (!pageBlocks?.length) return [];

  const sheets: ExportSheet[] = [];

  for (const block of pageBlocks) {
    if (block.type === 'dataTable' && block.dataKey) {
      const columns = tableColumns(block);
      if (isPlaceholderColumn(columns)) continue;
      const raw = getDataSlice(definition, block.dataKey as string);
      const rows = Array.isArray(raw) ? (raw as Record<string, unknown>[]) : [];
      if (!rows.length) continue;
      const normalized = rows.map((row) => {
        const out: Record<string, unknown> = {};
        for (const col of columns) {
          out[col.field] = row[col.field] ?? row[col.header] ?? '';
        }
        return out;
      });
      sheets.push({
        title: (block.title as string) || 'Table data',
        columns,
        rows: normalized,
      });
    }
  }

  for (const block of pageBlocks) {
    if ((block.type === 'lineChart' || block.type === 'barChart') && block.dataKey) {
      const columns = chartColumns(block);
      const raw = getDataSlice(definition, block.dataKey as string);
      const rows = Array.isArray(raw) ? (raw as Record<string, unknown>[]) : [];
      if (!rows.length || columns.length < 2) continue;
      sheets.push({
        title: (block.title as string) || 'Chart data',
        columns,
        rows,
      });
    }
  }

  return sheets;
}
