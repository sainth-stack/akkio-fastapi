import { createTheme } from '@mui/material/styles';
import { tokens as baseTokens } from './tokens';

export interface DefinitionTheme {
  primary?: string;
  primaryDark?: string;
  primaryLight?: string;
  secondary?: string;
  background?: string;
  surface?: string;
  text?: string;
  muted?: string;
  danger?: string;
  success?: string;
  warning?: string;
  info?: string;
  fontFamily?: string;
  chartPalette?: string[];
  shellVariant?: 'light' | 'brand';
  border?: string;
}

export function createAppTheme(definitionTheme?: DefinitionTheme) {
  const t = {
    ...baseTokens,
    ...(definitionTheme || {}),
    primary: definitionTheme?.primary || baseTokens.primary,
    primaryDark: definitionTheme?.primaryDark || baseTokens.primaryDark,
    primaryLight: definitionTheme?.primaryLight || baseTokens.primaryLight,
    secondary: definitionTheme?.secondary || baseTokens.secondary,
    background: definitionTheme?.background || baseTokens.background,
    surface: definitionTheme?.surface || baseTokens.surface,
    text: definitionTheme?.text || baseTokens.text,
    muted: definitionTheme?.muted || baseTokens.muted,
    danger: definitionTheme?.danger || baseTokens.danger,
    success: definitionTheme?.success || baseTokens.success,
    warning: definitionTheme?.warning || baseTokens.warning,
    info: definitionTheme?.info || baseTokens.info,
    fontFamily: definitionTheme?.fontFamily
      ? `'${definitionTheme.fontFamily}', system-ui, -apple-system, sans-serif`
      : baseTokens.fontFamily,
  };

  return createTheme({
    palette: {
      mode: 'light',
      primary: { main: t.primary, dark: t.primaryDark, light: t.primaryLight },
      secondary: { main: t.secondary },
      error: { main: t.danger },
      success: { main: t.success },
      warning: { main: t.warning },
      info: { main: t.info },
      background: { default: t.background, paper: t.surface },
      text: { primary: t.text, secondary: t.muted },
      divider: definitionTheme?.border || baseTokens.border,
    },
    typography: {
      fontFamily: t.fontFamily,
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
          root: { textTransform: 'none', fontWeight: 600, borderRadius: 8 },
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
              backgroundColor: `${t.primary}18`,
              color: t.primary,
              '& .MuiListItemIcon-root': { color: t.primary },
              '&:hover': { backgroundColor: `${t.primary}28` },
            },
          },
        },
      },
      MuiTextField: { defaultProps: { size: 'small' } },
      MuiChip: { styleOverrides: { root: { borderRadius: 6 } } },
      MuiTableCell: {
        styleOverrides: {
          head: { fontWeight: 600, backgroundColor: t.background },
        },
      },
    },
  });
}
