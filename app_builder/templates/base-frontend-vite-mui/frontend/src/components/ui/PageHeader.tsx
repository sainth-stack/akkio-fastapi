import React from 'react';
import { Box, Breadcrumbs, Divider, Link, Stack, Typography } from '@mui/material';

export interface PageHeaderBreadcrumb {
  label: string;
  href?: string;
}

export interface PageHeaderProps {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
  breadcrumbs?: PageHeaderBreadcrumb[];
}

export default function PageHeader({ title, subtitle, actions, breadcrumbs }: PageHeaderProps) {
  return (
    <Box mb={3}>
      {breadcrumbs && breadcrumbs.length > 0 && (
        <Breadcrumbs sx={{ mb: 1 }}>
          {breadcrumbs.map((bc, i) =>
            bc.href && i < breadcrumbs.length - 1 ? (
              <Link key={i} underline="hover" color="inherit" href={bc.href} sx={{ fontSize: 13 }}>
                {bc.label}
              </Link>
            ) : (
              <Typography key={i} color="text.primary" sx={{ fontSize: 13 }}>
                {bc.label}
              </Typography>
            ),
          )}
        </Breadcrumbs>
      )}

      <Stack direction="row" alignItems="flex-start" justifyContent="space-between" spacing={2}>
        <Box>
          <Typography variant="h5" fontWeight={700} gutterBottom={!!subtitle}>
            {title}
          </Typography>
          {subtitle && (
            <Typography variant="body2" color="text.secondary">
              {subtitle}
            </Typography>
          )}
        </Box>

        {actions && (
          <Stack direction="row" spacing={1} alignItems="center" flexShrink={0}>
            {actions}
          </Stack>
        )}
      </Stack>

      <Divider sx={{ mt: 2 }} />
    </Box>
  );
}
