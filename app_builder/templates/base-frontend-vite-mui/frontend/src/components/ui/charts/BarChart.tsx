import React from 'react';
import {
  Bar,
  BarChart as ReBarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export interface BarChartSeries {
  dataKey: string;
  color?: string;
  name?: string;
  radius?: number;
}

export interface BarChartProps {
  data: Record<string, unknown>[];
  series: BarChartSeries[];
  xAxisKey?: string;
  height?: number;
  showGrid?: boolean;
  showLegend?: boolean;
  showTooltip?: boolean;
  layout?: 'horizontal' | 'vertical';
}

const COLORS = ['#1976d2', '#2e7d32', '#ed6c02', '#9c27b0', '#0288d1'];

/**
 * BarChart — Recharts BarChart wrapper.
 */
export default function BarChart({
  data,
  series,
  xAxisKey = 'name',
  height = 280,
  showGrid = true,
  showLegend = true,
  showTooltip = true,
  layout = 'horizontal',
}: BarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ReBarChart
        data={data}
        layout={layout}
        margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
      >
        {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />}
        {layout === 'horizontal' ? (
          <>
            <XAxis dataKey={xAxisKey} tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
          </>
        ) : (
          <>
            <XAxis type="number" tick={{ fontSize: 12 }} />
            <YAxis dataKey={xAxisKey} type="category" tick={{ fontSize: 12 }} width={80} />
          </>
        )}
        {showTooltip && <Tooltip />}
        {showLegend && <Legend />}
        {series.map((s, i) => (
          <Bar
            key={s.dataKey}
            dataKey={s.dataKey}
            name={s.name ?? s.dataKey}
            fill={s.color ?? COLORS[i % COLORS.length]}
            radius={[s.radius ?? 4, s.radius ?? 4, 0, 0]}
          />
        ))}
      </ReBarChart>
    </ResponsiveContainer>
  );
}
