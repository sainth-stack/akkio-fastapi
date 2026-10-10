import React from 'react';
import { Alert, Snackbar, SnackbarOrigin } from '@mui/material';

export type ToastSeverity = 'success' | 'error' | 'warning' | 'info';

export interface ToastProps {
  open: boolean;
  onClose: () => void;
  message: string;
  severity?: ToastSeverity;
  autoHideDuration?: number;
  anchorOrigin?: SnackbarOrigin;
}

/**
 * Toast — Snackbar-based notification.  Control via `open`/`onClose`.
 */
export default function Toast({
  open,
  onClose,
  message,
  severity = 'info',
  autoHideDuration = 4000,
  anchorOrigin = { vertical: 'bottom', horizontal: 'right' },
}: ToastProps) {
  return (
    <Snackbar
      open={open}
      autoHideDuration={autoHideDuration}
      onClose={(_e, reason) => reason !== 'clickaway' && onClose()}
      anchorOrigin={anchorOrigin}
    >
      <Alert
        onClose={onClose}
        severity={severity}
        variant="filled"
        sx={{ minWidth: 280, borderRadius: 2 }}
        elevation={6}
      >
        {message}
      </Alert>
    </Snackbar>
  );
}
