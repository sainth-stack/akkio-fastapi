const PLACEHOLDER_FIELD = /^(col_?\d+|field_?\d+|column_?\d+)$/i;

const MACHINES = ['CNC Mill #3', 'Conveyor A', 'Hydraulic Press 2', 'Packaging Robot'];
const LINES = ['Line 1 - SMT', 'Line 2 - Assembly', 'Line 3 - Test'];
const CAUSES = ['Bearing wear', 'Sensor drift', 'Hydraulic leak', 'Electrical fault'];
const EVENTS = ['Preventive', 'Corrective', 'Breakdown', 'Inspection'];
const STATUSES = ['Complete', 'In progress', 'Scheduled', 'Critical'];

function cellValue(field: string, rowIndex: number): unknown {
  const f = field.toLowerCase();
  if (f.includes('work_order')) return `WO-${24000 + rowIndex}`;
  if (f === 'machine' || f === 'equipment') return MACHINES[rowIndex % MACHINES.length];
  if (f.includes('production_line') || f === 'line') return LINES[rowIndex % LINES.length];
  if (f === 'shift') return ['Day', 'Swing', 'Night'][rowIndex % 3];
  if (f.includes('root_cause')) return CAUSES[rowIndex % CAUSES.length];
  if (f.includes('event_type')) return EVENTS[rowIndex % EVENTS.length];
  if (f === 'severity') return ['Low', 'Medium', 'High', 'Critical'][rowIndex % 4];
  if (f === 'status') return STATUSES[rowIndex % STATUSES.length];
  if (f.includes('downtime')) return (0.5 + (rowIndex % 6) * 0.4).toFixed(1);
  if (f.includes('mtbf')) return (180 + rowIndex * 7).toFixed(0);
  if (f.includes('mttr')) return (0.8 + (rowIndex % 5) * 0.3).toFixed(1);
  if (f.includes('occurred') || f.includes('date') || f.includes('time')) {
    const day = (rowIndex % 28) + 1;
    return `2024-06-${String(day).padStart(2, '0')}T${String(8 + rowIndex % 10).padStart(2, '0')}:15:00Z`;
  }
  return `${field.replace(/_/g, ' ')} ${rowIndex + 1}`;
}

export function rowsLookPlaceholder(rows: Record<string, unknown>[]): boolean {
  if (!rows.length) return true;
  const sample = rows[0];
  return Object.keys(sample).some((k) => PLACEHOLDER_FIELD.test(k))
    || Object.values(sample).some(
      (v) => typeof v === 'string' && /^Field \d+ \d+$/i.test(v),
    );
}

export function repairTableRows(
  rows: Record<string, unknown>[],
  columns: Array<{ field: string }>,
  targetLen = 28,
): Record<string, unknown>[] {
  const fields = columns.map((c) => c.field).filter(Boolean);
  if (!fields.length) return rows;
  const len = Math.max(targetLen, rows.length || targetLen);
  return Array.from({ length: len }, (_, i) =>
    Object.fromEntries(fields.map((f) => [f, cellValue(f, i)])),
  );
}
