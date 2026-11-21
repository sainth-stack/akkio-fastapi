import React from 'react';
import {
  Cell,
  Legend,
  Pie,
  PieChart as RePieChart,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import { ChartTooltip, CHART_COLORS } from './chartTheme';

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

export default function PieChart({
  data,
  height = 280,
  innerRadius = 58,
  outerRadius = 88,
  showLegend = true,
  showTooltip = true,
  showLabels = false,
}: PieChartProps) {
  const total = data.reduce((s, d) => s + (d.value || 0), 0);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <RePieChart>
        <Pie
          data={data}
          cx="42%"
          cy="50%"
          innerRadius={innerRadius}
          outerRadius={outerRadius}
          dataKey="value"
          nameKey="name"
          paddingAngle={2}
          label={showLabels ? ({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%` : false}
          labelLine={showLabels}
        >
          {data.map((entry, i) => (
            <Cell
              key={`cell-${i}`}
              fill={entry.color ?? CHART_COLORS[i % CHART_COLORS.length]}
              stroke="#fff"
              strokeWidth={2}
            />
          ))}
        </Pie>
        {showTooltip && <Tooltip content={<ChartTooltip />} />}
        {showLegend && (
          <Legend
            layout="vertical"
            align="right"
            verticalAlign="middle"
            iconType="circle"
            iconSize={8}
            formatter={(value: string) => {
              const item = data.find((d) => d.name === value);
              const pct = item && total ? Math.round((item.value / total) * 100) : null;
              const label = pct != null ? `${value} (${pct}%)` : value;
              return <span style={{ fontSize: 12, color: '#64748b' }}>{label}</span>;
            }}
          />
        )}
      </RePieChart>
    </ResponsiveContainer>
  );
}
