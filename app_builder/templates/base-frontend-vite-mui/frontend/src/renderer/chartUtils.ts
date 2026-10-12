export function resolveXAxisKey(
  data: Record<string, unknown>[],
  preferred?: string,
): string {
  if (!data?.length) return preferred || 'name';
  const row = data[0];
  if (preferred && preferred in row) return preferred;
  for (const k of ['label', 'name', 'week', 'shift', 'hour', 'date', 'period']) {
    if (k in row) return k;
  }
  return preferred || 'name';
}

export function resolveRowKey(
  rows: Record<string, unknown>[],
  columns: Array<{ field: string }>,
  explicit?: string,
): string | undefined {
  if (explicit && rows.some((r) => r[explicit] != null)) return explicit;
  if (rows.some((r) => r.id != null)) return 'id';
  const firstField = columns[0]?.field;
  if (firstField && rows.some((r) => r[firstField] != null)) return firstField;
  return undefined;
}
