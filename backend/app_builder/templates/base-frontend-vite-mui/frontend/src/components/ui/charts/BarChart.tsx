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
import { ChartTooltip, CHART_COLORS, chartAxisTick, chartGridStroke, formatAxisLabel } from './chartTheme';

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
  const isGrouped = series.length > 1;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ReBarChart
        data={data}
        layout={layout}
        margin={{ top: 12, right: 8, left: -8, bottom: 4 }}
        barCategoryGap={isGrouped ? '18%' : '28%'}
        barGap={4}
      >
        {showGrid && (
          <CartesianGrid stroke={chartGridStroke} strokeDasharray="4 4" vertical={false} />
        )}
        {layout === 'horizontal' ? (
          <>
            <XAxis
              dataKey={xAxisKey}
              tick={chartAxisTick}
              tickLine={false}
              axisLine={{ stroke: chartGridStroke }}
              tickFormatter={formatAxisLabel}
              minTickGap={20}
            />
            <YAxis tick={chartAxisTick} tickLine={false} axisLine={false} width={42} />
          </>
        ) : (
          <>
            <XAxis type="number" tick={chartAxisTick} tickLine={false} axisLine={false} />
            <YAxis
              dataKey={xAxisKey}
              type="category"
              tick={chartAxisTick}
              tickLine={false}
              axisLine={false}
              width={96}
              tickFormatter={formatAxisLabel}
            />
          </>
        )}
        {showTooltip && <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(37, 99, 235, 0.06)' }} />}
        {showLegend && series.length > 1 && (
          <Legend
            verticalAlign="bottom"
            height={32}
            iconType="circle"
            iconSize={8}
            wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
          />
        )}
        {series.map((s, i) => (
          <Bar
            key={s.dataKey}
            dataKey={s.dataKey}
            name={s.name ?? s.dataKey}
            fill={s.color ?? CHART_COLORS[i % CHART_COLORS.length]}
            radius={[s.radius ?? 5, s.radius ?? 5, 0, 0]}
            maxBarSize={isGrouped ? 22 : 36}
          />
        ))}
      </ReBarChart>
    </ResponsiveContainer>
  );
}
