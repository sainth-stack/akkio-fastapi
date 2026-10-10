import React from 'react';
import {
  Cell,
  Legend,
  Pie,
  PieChart as RePieChart,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';

export interface PieChartDataItem {
  name: string;
  value: number;
  color?: string;
}

export interface PieChartProps {
  data: PieChartDataItem[];
  height?: number;
  innerRadius?: number;
  outerRadius?: number;
  showLegend?: boolean;
  showTooltip?: boolean;
  showLabels?: boolean;
}

const DEFAULT_COLORS = [
  '#1976d2', '#2e7d32', '#ed6c02', '#9c27b0', '#0288d1',
  '#d32f2f', '#00796b', '#f57c00', '#7b1fa2', '#0097a7',
];

/**
 * PieChart — Recharts PieChart wrapper.  Supports donut (innerRadius > 0).
 */
export default function PieChart({
  data,
  height = 280,
  innerRadius = 0,
  outerRadius = 80,
  showLegend = true,
  showTooltip = true,
  showLabels = false,
}: PieChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RePieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={innerRadius}
          outerRadius={outerRadius}
          dataKey="value"
          nameKey="name"
          label={showLabels ? ({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%` : false}
          labelLine={showLabels}
        >
          {data.map((entry, i) => (
            <Cell
              key={`cell-${i}`}
              fill={entry.color ?? DEFAULT_COLORS[i % DEFAULT_COLORS.length]}
            />
          ))}
        </Pie>
        {showTooltip && <Tooltip formatter={(val: number) => val.toLocaleString()} />}
        {showLegend && <Legend />}
      </RePieChart>
    </ResponsiveContainer>
  );
}
