import React from 'react';
import {
  Button as MuiButton,
  ButtonProps as MuiButtonProps,
  CircularProgress,
} from '@mui/material';

export interface ButtonProps extends Omit<MuiButtonProps, 'ref'> {
  loading?: boolean;
}

/**
 * Button — wraps MUI Button with a built-in loading spinner.
 * All MUI ButtonProps are forwarded.
 */
export default function Button({ loading = false, disabled, children, startIcon, ...props }: ButtonProps) {
  return (
    <MuiButton
      {...props}
      disabled={disabled || loading}
      startIcon={loading ? <CircularProgress size={16} color="inherit" /> : startIcon}
    >
      {children}
    </MuiButton>
  );
}
