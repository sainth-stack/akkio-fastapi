import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { CssBaseline, ThemeProvider } from '@mui/material';
import App from './App';
import { appTheme } from './theme';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 30_000 },
  },
});

// When served under /app/{project_id}/ the static server injects
// window.__AKKIO_BASE_PATH__ = '/app/{project_id}'.
// BrowserRouter needs this as `basename` so route matching works correctly.
declare global {
  interface Window {
    __AKKIO_BASE_PATH__?: string;
    __AKKIO_API_BASE__?: string;
    __AKKIO_ACCESS_TOKEN__?: string;
  }
}
const basename: string = window.__AKKIO_BASE_PATH__ || '/';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={appTheme}>
        <CssBaseline />
        <BrowserRouter basename={basename}>
          <App />
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  </React.StrictMode>,
);
