/**
 * Design tokens for the generated app.
 * This file is OVERWRITTEN by the deterministic generator — edit the blueprint
 * or design_tokens in the Agentic Builder to change these values.
 */
export const tokens = {
  primary:        '#1976d2',
  primaryDark:    '#1565c0',
  primaryLight:   '#42a5f5',
  secondary:      '#9c27b0',
  accent:         '#00bcd4',
  background:     '#f5f7fb',
  surface:        '#ffffff',
  text:           '#0f172a',
  muted:          '#64748b',
  border:         '#e2e8f0',
  danger:         '#d32f2f',
  success:        '#2e7d32',
  warning:        '#ed6c02',
  info:           '#0288d1',
  fontFamily:     "'Inter', system-ui, -apple-system, sans-serif",
} as const;

export type Tokens = typeof tokens;
