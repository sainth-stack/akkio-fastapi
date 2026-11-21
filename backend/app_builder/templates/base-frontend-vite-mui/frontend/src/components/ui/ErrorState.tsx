import React from 'react';
import { Box, Button, Stack, Typography } from '@mui/material';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

export interface ErrorStateProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
  retryLabel?: string;
}

/**
 * ErrorState — error icon + message + optional retry button.
 */
export default function ErrorState({
  title = 'Something went wrong',
  message = 'An unexpected error occurred. Please try again.',
  onRetry,
  retryLabel = 'Retry',
}: ErrorStateProps) {
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 240,
        py: 4,
        px: 3,
        textAlign: 'center',
      }}
    >
      <Stack alignItems="center" spacing={2}>
        <ErrorOutlineIcon sx={{ fontSize: 56, color: 'error.main' }} />
        <Typography variant="h6" fontWeight={600}>
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary" maxWidth={360}>
          {message}
        </Typography>
        {onRetry && (
          <Button variant="contained" color="error" onClick={onRetry}>
            {retryLabel}
          </Button>
        )}
      </Stack>
    </Box>
  );
}
