import React from 'react';
import { Box, Paper, Skeleton, Stack, SxProps, Theme, Typography } from '@mui/material';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

export interface StatCardTrend {
  value: number;    // positive = up, negative = down
  label?: string;   // e.g. "vs last month"
}

export interface StatCardProps {
  label: string;
  value: string | number;
  icon?: React.ReactNode;
  trend?: StatCardTrend;
  color?: 'primary' | 'success' | 'error' | 'warning' | 'info' | 'default';
  loading?: boolean;
  sx?: SxProps<Theme>;
}

const COLOR_MAP: Record<string, string> = {
  primary: '#1976d2',
  success: '#2e7d32',
  error:   '#d32f2f',
  warning: '#ed6c02',
  info:    '#0288d1',
  default: '#64748b',
};

export default function StatCard({
  label,
  value,
  icon,
  trend,
  color = 'primary',
  loading = false,
  sx,
}: StatCardProps) {
  const accentColor = COLOR_MAP[color] ?? COLOR_MAP.primary;
  const isUp = trend && trend.value >= 0;

  return (
    <Paper
      sx={{
        p: 2.5,
        borderRadius: 3,
        borderLeft: `4px solid ${accentColor}`,
        transition: 'box-shadow 0.2s',
        '&:hover': { boxShadow: '0 4px 12px rgba(0,0,0,0.12)' },
        ...sx,
      }}
    >
      <Stack direction="row" alignItems="flex-start" justifyContent="space-between" spacing={1}>
        <Box flex={1}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            {label}
          </Typography>

          {loading ? (
            <Skeleton variant="text" width={80} height={40} />
          ) : (
            <Typography variant="h4" fontWeight={700} color={`${color}.main`} lineHeight={1.2}>
              {value}
            </Typography>
          )}

          {trend && !loading && (
            <Stack direction="row" alignItems="center" spacing={0.5} mt={0.5}>
              {isUp ? (
                <TrendingUpIcon sx={{ fontSize: 16, color: 'success.main' }} />
              ) : (
                <TrendingDownIcon sx={{ fontSize: 16, color: 'error.main' }} />
              )}
              <Typography
                variant="caption"
                color={isUp ? 'success.main' : 'error.main'}
                fontWeight={600}
              >
                {isUp ? '+' : ''}
                {trend.value}%
              </Typography>
              {trend.label && (
                <Typography variant="caption" color="text.secondary">
                  {trend.label}
                </Typography>
              )}
            </Stack>
          )}
        </Box>

        {icon && (
          <Box
            sx={{
              p: 1.5,
              borderRadius: 2,
              bgcolor: `${accentColor}18`,
              color: accentColor,
              display: 'flex',
              alignItems: 'center',
            }}
          >
            {icon}
          </Box>
        )}
      </Stack>
    </Paper>
  );
}
