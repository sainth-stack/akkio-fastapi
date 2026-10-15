import React from 'react';
import { Box, Paper, Skeleton, Stack, SxProps, Theme, Typography } from '@mui/material';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

export interface StatCardTrend {
  value: number;
  label?: string;
}

export interface StatCardProps {
  label: string;
  value: string | number;
  icon?: React.ReactNode;
  trend?: StatCardTrend;
  target?: string;
  color?: 'primary' | 'success' | 'error' | 'warning' | 'info' | 'default';
  loading?: boolean;
  sx?: SxProps<Theme>;
  trendKey?: string;
  targetKey?: string;
}

export default function StatCard({
  label,
  value,
  trend,
  target,
  loading = false,
  sx,
}: StatCardProps) {
  const isUp = trend && trend.value >= 0;
  const trendGood = trend ? (label.toLowerCase().includes('scrap') || label.toLowerCase().includes('downtime')
    ? !isUp
    : isUp) : true;

  return (
    <Paper
      elevation={0}
      sx={{
        p: 2,
        height: '100%',
        borderRadius: 2.5,
        border: '1px solid',
        borderColor: 'divider',
        bgcolor: 'background.paper',
        transition: 'border-color 0.15s ease, box-shadow 0.15s ease',
        '&:hover': {
          borderColor: 'primary.light',
          boxShadow: '0 4px 14px rgba(15, 23, 42, 0.06)',
        },
        ...sx,
      }}
    >
      <Typography
        variant="caption"
        color="text.secondary"
        fontWeight={600}
        sx={{ textTransform: 'uppercase', letterSpacing: '0.06em', fontSize: '0.68rem' }}
        display="block"
        mb={0.75}
      >
        {label}
      </Typography>

      {loading ? (
        <Skeleton variant="text" width="70%" height={36} />
      ) : (
        <Typography variant="h5" fontWeight={700} color="text.primary" lineHeight={1.15} letterSpacing="-0.02em">
          {value}
        </Typography>
      )}

      <Stack direction="row" alignItems="center" justifyContent="space-between" mt={1} minHeight={20}>
        {trend && !loading ? (
          <Stack direction="row" alignItems="center" spacing={0.35}>
            {isUp ? (
              <TrendingUpIcon sx={{ fontSize: 15, color: trendGood ? 'success.main' : 'error.main' }} />
            ) : (
              <TrendingDownIcon sx={{ fontSize: 15, color: trendGood ? 'success.main' : 'error.main' }} />
            )}
            <Typography
              variant="caption"
              fontWeight={600}
              color={trendGood ? 'success.main' : 'error.main'}
            >
              {isUp ? '+' : ''}
              {trend.value}%
            </Typography>
            {trend.label ? (
              <Typography variant="caption" color="text.secondary" sx={{ ml: 0.25 }}>
                {trend.label}
              </Typography>
            ) : null}
          </Stack>
        ) : (
          <Box />
        )}
        {target && !loading ? (
          <Typography variant="caption" color="text.secondary">
            Target {target}
          </Typography>
        ) : null}
      </Stack>
    </Paper>
  );
}
