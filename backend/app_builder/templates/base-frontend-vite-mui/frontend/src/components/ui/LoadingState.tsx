import React from 'react';
import { Box, CircularProgress, Stack, Typography } from '@mui/material';

export interface LoadingStateProps {
  message?: string;
  size?: 'small' | 'medium' | 'large';
  fullHeight?: boolean;
}

const SIZE_MAP = { small: 24, medium: 40, large: 56 };

/**
 * LoadingState — centered spinner with optional status message.
 */
export default function LoadingState({
  message = 'Loading…',
  size = 'medium',
  fullHeight = true,
}: LoadingStateProps) {
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: fullHeight ? 240 : 'auto',
        py: fullHeight ? 0 : 4,
      }}
    >
      <Stack alignItems="center" spacing={2}>
        <CircularProgress size={SIZE_MAP[size]} />
        {message && (
          <Typography variant="body2" color="text.secondary">
            {message}
          </Typography>
        )}
      </Stack>
    </Box>
  );
}
