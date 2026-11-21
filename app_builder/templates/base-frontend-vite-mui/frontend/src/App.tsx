import { useMemo } from 'react';
import { CssBaseline, ThemeProvider } from '@mui/material';
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import AppShell from './components/ui/AppShell';
import PageRenderer from './renderer/PageRenderer';
import PredictPage from './pages/PredictPage';
import ObservabilityPage from './pages/ObservabilityPage';
import CreateTicketPage from './pages/CreateTicketPage';
import SimilarTicketsPage from './pages/SimilarTicketsPage';
import { AppDefinitionProvider, useAppDefinition } from './renderer/useAppDefinition';
import { createAppTheme } from './theme/createAppTheme';
import AppChatAssistant from './components/chat/AppChatAssistant';
import {
  isCreateTicketNavItem,
  isObservabilityNavItem,
  isPredictNavItem,
  isSimilarTicketsNavItem,
  resolveActiveNavItem,
} from './renderer/navMatch';

function AppShellLayout() {
  const { definition } = useAppDefinition();
  const location = useLocation();

  const nav = useMemo(() => {
    if (!definition) return [];
    const seen = new Set<string>();
    const items: typeof definition.navigation = [];
    for (const n of definition.navigation) {
      if (seen.has(n.pageId)) continue;
      seen.add(n.pageId);
      items.push(n);
    }
    return items.map((n) => ({
      label: n.label,
      path: n.path,
      icon: n.icon,
      pageId: n.pageId,
    }));
  }, [definition]);

  if (!definition) return null;

  const activeNav = resolveActiveNavItem(nav, location.pathname);
  const page = definition.pages.find((p) => p.id === activeNav?.pageId);

  const shellVariant =
    definition.theme?.shellVariant === 'brand' ? 'brand' : 'light';

  return (
    <>
      <AppShell
        navItems={nav}
        appName={definition.meta.appName}
        pageTitle={activeNav?.label || page?.title}
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

  const routesNav = useMemo(() => {
    const seen = new Set<string>();
    return definition.navigation.filter((n) => {
      if (seen.has(n.pageId)) return false;
      seen.add(n.pageId);
      return true;
    });
  }, [definition.navigation]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Routes>
        <Route element={<AppShellLayout />}>
          {routesNav.map((item) => (
            <Route
              key={item.pageId}
              path={item.path}
              element={
                isPredictNavItem(item) ? (
                  <PredictPage />
                ) : isObservabilityNavItem(item) ? (
                  <ObservabilityPage />
                ) : isCreateTicketNavItem(item) ? (
                  <CreateTicketPage />
                ) : isSimilarTicketsNavItem(item) ? (
                  <SimilarTicketsPage />
                ) : (
                  <PageRenderer pageId={item.pageId} />
                )
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
