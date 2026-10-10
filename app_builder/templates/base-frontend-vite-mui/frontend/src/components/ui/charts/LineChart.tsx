import React from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart as ReLineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export interface LineChartSeries {
  dataKey: string;
  color?: string;
  name?: string;
  strokeWidth?: number;
  dot?: boolean;
}

export interface LineChartProps {
  data: Record<string, unknown>[];
  series: LineChartSeries[];
  xAxisKey?: string;
  height?: number;
  showGrid?: boolean;
  showLegend?: boolean;
  showTooltip?: boolean;
}

const COLORS = ['#1976d2', '#2e7d32', '#ed6c02', '#9c27b0', '#0288d1'];

/**
 * LineChart — Recharts LineChart wrapper with ResponsiveContainer.
 */
export default function LineChart({
  data,
  series,
  xAxisKey = 'name',
  height = 280,
  showGrid = true,
  showLegend = true,
  showTooltip = true,
}: LineChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ReLineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />}
        <XAxis dataKey={xAxisKey} tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 12 }} />
        {showTooltip && <Tooltip />}
        {showLegend && <Legend />}
        {series.map((s, i) => (
          <Line
            key={s.dataKey}
            type="monotone"
            dataKey={s.dataKey}
            name={s.name ?? s.dataKey}
            stroke={s.color ?? COLORS[i % COLORS.length]}
            strokeWidth={s.strokeWidth ?? 2}
            dot={s.dot ?? false}
            activeDot={{ r: 4 }}
          />
        ))}
      </ReLineChart>
    </ResponsiveContainer>
  );
}
