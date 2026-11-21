import React from 'react';
import { Box, Stack, Typography } from '@mui/material';
import InboxIcon from '@mui/icons-material/Inbox';

export interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
}

/**
 * EmptyState — centered illustration + title + description + optional CTA.
 */
export default function EmptyState({
  icon,
  title,
  description,
  action,
}: EmptyStateProps) {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        py: 8,
        px: 3,
        textAlign: 'center',
      }}
    >
      <Stack alignItems="center" spacing={1.5}>
        <Box color="text.disabled">
          {icon ?? <InboxIcon sx={{ fontSize: 64 }} />}
        </Box>
        <Typography variant="h6" fontWeight={600} color="text.primary">
          {title}
        </Typography>
        {description && (
          <Typography variant="body2" color="text.secondary" maxWidth={360}>
            {description}
          </Typography>
        )}
        {action}
      </Stack>
    </Box>
  );
}
