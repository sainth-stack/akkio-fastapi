import React from 'react';
import { TextField, TextFieldProps } from '@mui/material';

export interface FormFieldProps extends Omit<TextFieldProps, 'error' | 'ref'> {
  error?: string;
}

/**
 * FormField — MUI TextField with a string `error` prop instead of boolean.
 * Automatically sets `helperText` to the error string when present.
 */
export default function FormField({ error, helperText, ...props }: FormFieldProps) {
  return (
    <TextField
      fullWidth
      variant="outlined"
      size="small"
      {...props}
      error={!!error}
      helperText={error ?? helperText}
    />
  );
}
