import { useMemo } from 'react';
import { CssBaseline, ThemeProvider } from '@mui/material';
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import AppShell from './components/ui/AppShell';
import PageRenderer from './renderer/PageRenderer';
import PredictPage from './pages/PredictPage';
import { AppDefinitionProvider, useAppDefinition } from './renderer/useAppDefinition';
import { createAppTheme } from './theme/createAppTheme';
import AppChatAssistant from './components/chat/AppChatAssistant';

function AppShellLayout() {
  const { definition } = useAppDefinition();
  const location = useLocation();

  if (!definition) return null;

  const nav = definition.navigation.map((n) => ({
    label: n.label,
    path: n.path,
    icon: n.icon,
  }));

  const activeNav = nav.find((n) =>
    n.path === '/' ? location.pathname === '/' : location.pathname.startsWith(n.path),
  );
  const page = definition.pages.find((p) => p.id === activeNav?.pageId);

  const shellVariant =
    definition.theme?.shellVariant === 'brand' ? 'brand' : 'light';

  return (
    <>
      <AppShell
        navItems={nav}
        appName={definition.meta.appName}
        pageTitle={page?.title}
        shellVariant={shellVariant}
      />
      <AppChatAssistant definition={definition} pageId={page?.id} pageTitle={page?.title} />
    </>
  );
}

function AppRoutes() {
  const { definition, loading, error } = useAppDefinition();

  const theme = useMemo(
    () => createAppTheme(definition?.theme as Record<string, string> | undefined),
    [definition?.theme],
  );

  if (loading && !definition) {
    return null;
  }
  if (error || !definition) {
    return <div style={{ padding: 24 }}>Failed to load app definition.</div>;
  }

  const defaultPath = definition.navigation[0]?.path ?? '/';

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Routes>
        <Route element={<AppShellLayout />}>
          {definition.navigation.map((item) => (
            <Route
              key={item.pageId}
              path={item.path}
              element={
                item.pageId === 'predict' ? <PredictPage /> : <PageRenderer pageId={item.pageId} />
              }
            />
          ))}
          <Route path="*" element={<Navigate to={defaultPath} replace />} />
        </Route>
      </Routes>
    </ThemeProvider>
  );
}

export default function App() {
  return (
    <AppDefinitionProvider>
      <AppRoutes />
    </AppDefinitionProvider>
  );
}
