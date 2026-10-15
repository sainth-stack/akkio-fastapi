import React from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { ChartTooltip, CHART_COLORS, chartAxisTick, chartGridStroke, formatAxisLabel } from './chartTheme';

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
  rich?: boolean;
}

export default function LineChart({
  data,
  series,
  xAxisKey = 'name',
  height = 280,
  showGrid = true,
  showLegend = true,
  showTooltip = true,
  rich = false,
}: LineChartProps) {
  const primaryKey = series.find((s) => !/target|benchmark|capacity/i.test(s.dataKey + (s.name || '')))?.dataKey
    || series[0]?.dataKey;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 12, right: 8, left: -8, bottom: 4 }}>
        {rich && primaryKey ? (
          <defs>
            <linearGradient id={`fill-${primaryKey}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={series[0]?.color || CHART_COLORS[0]} stopOpacity={0.28} />
              <stop offset="100%" stopColor={series[0]?.color || CHART_COLORS[0]} stopOpacity={0.02} />
            </linearGradient>
          </defs>
        ) : null}
        {showGrid && (
          <CartesianGrid stroke={chartGridStroke} strokeDasharray="4 4" vertical={false} />
        )}
        <XAxis
          dataKey={xAxisKey}
          tick={chartAxisTick}
          tickLine={false}
          axisLine={{ stroke: chartGridStroke }}
          tickFormatter={formatAxisLabel}
          minTickGap={24}
        />
        <YAxis
          tick={chartAxisTick}
          tickLine={false}
          axisLine={false}
          width={42}
          tickFormatter={(v) => (typeof v === 'number' && v >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(v))}
        />
        {showTooltip && <Tooltip content={<ChartTooltip />} />}
        {showLegend && series.length > 1 && (
          <Legend
            verticalAlign="bottom"
            height={32}
            iconType="circle"
            iconSize={8}
            wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
          />
        )}
        {rich && primaryKey ? (
          <Area
            type="monotone"
            dataKey={primaryKey}
            stroke="none"
            fill={`url(#fill-${primaryKey})`}
            legendType="none"
            tooltipType="none"
          />
        ) : null}
        {series.map((s, i) => {
          const isRef = /target|benchmark|capacity/i.test(s.dataKey + (s.name || ''));
          return (
            <Line
              key={s.dataKey}
              type="monotone"
              dataKey={s.dataKey}
              name={s.name ?? s.dataKey}
              stroke={s.color ?? CHART_COLORS[i % CHART_COLORS.length]}
              strokeWidth={isRef ? 1.5 : s.strokeWidth ?? 2.5}
              strokeDasharray={isRef ? '6 4' : undefined}
              dot={isRef ? false : s.dot ?? false}
              activeDot={{ r: 4, strokeWidth: 0 }}
            />
          );
        })}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
