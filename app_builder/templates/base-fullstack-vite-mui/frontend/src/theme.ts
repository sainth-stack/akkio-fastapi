import { createTheme } from '@mui/material/styles';

const tokens = {
  primary: '#1565c0',
  background: '#f5f7fb',
  surface: '#ffffff',
  text: '#0f172a',
  muted: '#64748b',
  danger: '#d32f2f',
  success: '#2e7d32',
  warning: '#ed6c02',
};

export const appTheme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: tokens.primary },
    error: { main: tokens.danger },
    success: { main: tokens.success },
    warning: { main: tokens.warning },
    background: { default: tokens.background, paper: tokens.surface },
    text: { primary: tokens.text, secondary: tokens.muted },
  },
  typography: {
    fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
    h5: { fontWeight: 700 },
    h6: { fontWeight: 600 },
  },
  shape: { borderRadius: 10 },
  components: {
    MuiButton: { styleOverrides: { root: { textTransform: 'none', fontWeight: 600 } } },
    MuiPaper: { styleOverrides: { root: { boxShadow: '0 1px 2px rgba(15, 23, 42, 0.06)' } } },
  },
});

export { tokens };
