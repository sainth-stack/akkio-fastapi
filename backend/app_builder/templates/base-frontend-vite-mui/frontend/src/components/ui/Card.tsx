import React from 'react';
import {
  Card as MuiCard,
  CardContent,
  CardHeader,
  Divider,
  SxProps,
  Theme,
} from '@mui/material';

export interface CardProps {
  title?: string;
  subheader?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
  noPadding?: boolean;
  sx?: SxProps<Theme>;
}

/**
 * Card — MUI Card with optional title, subheader, header actions, and
 * content area.  Pass `noPadding` to remove the default CardContent padding.
 */
export default function Card({ title, subheader, actions, children, noPadding = false, sx }: CardProps) {
  return (
    <MuiCard sx={sx}>
      {(title || actions) && (
        <>
          <CardHeader
            title={title}
            subheader={subheader}
            titleTypographyProps={{ variant: 'h6', fontWeight: 600 }}
            subheaderTypographyProps={{ variant: 'body2' }}
            action={actions}
            sx={{ pb: 0 }}
          />
          <Divider sx={{ mt: 1 }} />
        </>
      )}
      {noPadding ? children : <CardContent>{children}</CardContent>}
    </MuiCard>
  );
}
