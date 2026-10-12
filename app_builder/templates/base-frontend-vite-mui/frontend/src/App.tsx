import { useMemo } from 'react';
import { CssBaseline, ThemeProvider } from '@mui/material';
import { Navigate, Route, Routes } from 'react-router-dom';
import AppShell from './components/ui/AppShell';
import PageRenderer from './renderer/PageRenderer';
import { useAppDefinition } from './renderer/useAppDefinition';
import { createAppTheme } from './theme/createAppTheme';

export default function App() {
  const { definition, loading, error } = useAppDefinition();

  const theme = useMemo(
    () => createAppTheme(definition?.theme as Record<string, string> | undefined),
    [definition?.theme],
  );

  if (loading) {
    return null;
  }
  if (error || !definition) {
    return <div style={{ padding: 24 }}>Failed to load app definition.</div>;
  }

  const nav = definition.navigation.map((n) => ({
    label: n.label,
    path: n.path,
    icon: n.icon,
  }));
  const defaultPath = definition.navigation[0]?.path ?? '/';

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Routes>
        <Route element={<AppShell navItems={nav} appName={definition.meta.appName} />}>
          {definition.navigation.map((item) => (
            <Route
              key={item.pageId}
              path={item.path}
              element={<PageRenderer pageId={item.pageId} />}
            />
          ))}
          <Route path="*" element={<Navigate to={defaultPath} replace />} />
        </Route>
      </Routes>
    </ThemeProvider>
  );
}
