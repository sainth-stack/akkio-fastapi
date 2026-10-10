import { createTheme } from '@mui/material/styles';
import { tokens } from './tokens';

export const appTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main:  tokens.primary,
      dark:  tokens.primaryDark,
      light: tokens.primaryLight,
    },
    secondary: { main: tokens.secondary },
    error:     { main: tokens.danger },
    success:   { main: tokens.success },
    warning:   { main: tokens.warning },
    info:      { main: tokens.info },
    background: {
      default: tokens.background,
      paper:   tokens.surface,
    },
    text: {
      primary:   tokens.text,
      secondary: tokens.muted,
    },
    divider: tokens.border,
  },
  typography: {
    fontFamily: tokens.fontFamily,
    h4: { fontWeight: 700 },
    h5: { fontWeight: 700 },
    h6: { fontWeight: 600 },
    subtitle1: { fontWeight: 500 },
    button: { textTransform: 'none', fontWeight: 600 },
  },
  shape: { borderRadius: 10 },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          boxShadow: '0 1px 3px rgba(15, 23, 42, 0.08)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 1px 3px rgba(15, 23, 42, 0.08)',
          borderRadius: 12,
        },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          margin: '2px 8px',
          '&.Mui-selected': {
            backgroundColor: `${tokens.primary}18`,
            color: tokens.primary,
            '& .MuiListItemIcon-root': { color: tokens.primary },
            '&:hover': { backgroundColor: `${tokens.primary}28` },
          },
        },
      },
    },
    MuiTextField: {
      defaultProps: { size: 'small' },
    },
    MuiChip: {
      styleOverrides: { root: { borderRadius: 6 } },
    },
    MuiTableCell: {
      styleOverrides: {
        head: { fontWeight: 600, backgroundColor: tokens.background },
      },
    },
  },
});

export { tokens };
