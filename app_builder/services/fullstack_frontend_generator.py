"""Frontend files for Agentic Builder production MVPs."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app_builder.services.design_system_css import extract_primary_hex, normalize_token_payload

QUALITY_NAV = [
    ("/", "Dashboard"),
    ("/suppliers", "Suppliers"),
    ("/supplier-quality", "Supplier Quality"),
    ("/materials", "Materials"),
    ("/purchase-orders", "Purchase Orders"),
    ("/incoming-lots", "Incoming Lots"),
    ("/inspections", "Incoming Inspection"),
    ("/defects", "Defects"),
    ("/releases", "Release Decisions"),
    ("/capa", "CAPA"),
    ("/ai", "AI Quality Assistant"),
    ("/reports", "Reports"),
]


def _hex(value: Any, fallback: str) -> str:
    if isinstance(value, str) and re.match(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$", value):
        return value
    return fallback


def theme_from_tokens(
    design_tokens: Optional[Dict[str, Any]] = None,
    uiux: str = "",
) -> Dict[str, str]:
    """Map design system / UIUX palette to enterprise automotive MUI tokens."""
    inner = normalize_token_payload(design_tokens)
    colors = inner.get("colors") if isinstance(inner.get("colors"), dict) else {}
    if not isinstance(colors, dict):
        colors = {}
    primary = _hex(colors.get("primary"), extract_primary_hex(uiux, "#0B3D6F"))
    return {
        "primary": primary,
        "primary_dark": _hex(colors.get("primary_dark"), "#082847"),
        "primary_light": _hex(colors.get("primary_light"), "#1E5A8A"),
        "secondary": _hex(colors.get("secondary"), "#1565C0"),
        "accent": _hex(colors.get("accent"), "#00897B"),
        "background": _hex(colors.get("background"), "#EEF2F6"),
        "surface": _hex(colors.get("surface"), "#FFFFFF"),
        "text": _hex(colors.get("text"), "#0F172A"),
        "muted": _hex(colors.get("muted"), "#64748B"),
        "border": _hex(colors.get("border"), "#CBD5E1"),
        "danger": _hex(colors.get("danger") or colors.get("error"), "#C62828"),
        "success": _hex(colors.get("success"), "#2E7D32"),
        "warning": _hex(colors.get("warning"), "#ED6C02"),
        "info": _hex(colors.get("info"), "#0288D1"),
    }


def build_theme_ts(colors: Dict[str, str]) -> str:
    return _theme_ts(colors)


def frontend_files(title: str, colors: Dict[str, str], quality: bool) -> Dict[str, str]:
    nav = QUALITY_NAV if quality else [("/", "Dashboard"), ("/records", "Records"), ("/reports", "Reports"), ("/ai", "AI Assistant")]
    files = {
        "frontend/src/theme.ts": build_theme_ts(colors),
        "frontend/src/App.tsx": _app_tsx(quality),
        "frontend/src/layout/AppShell.tsx": _shell_tsx(title, nav),
        "frontend/src/components/PageHeader.tsx": _page_header_tsx(),
        "frontend/src/pages/DashboardPage.tsx": _dashboard_tsx(),
        "frontend/src/pages/ResourceListPage.tsx": _resource_list_tsx(),
        "frontend/src/pages/SupplierDetailPage.tsx": _supplier_detail_tsx(),
        "frontend/src/pages/LotDetailPage.tsx": _lot_detail_tsx(),
        "frontend/src/pages/InspectionPage.tsx": _inspection_tsx(),
        "frontend/src/pages/CapaPage.tsx": _capa_tsx(),
        "frontend/src/pages/ReportsPage.tsx": _reports_tsx(),
        "frontend/src/pages/AIAssistantPage.tsx": _ai_tsx(),
        "frontend/src/components/KPICard.tsx": _small_component("KPICard"),
        "frontend/src/components/StatusChip.tsx": _small_component("StatusChip"),
        "frontend/src/components/RiskBadge.tsx": _small_component("RiskBadge"),
        "frontend/src/components/SupplierScore.tsx": _small_component("SupplierScore"),
        "frontend/src/components/LotTable.tsx": _small_component("LotTable"),
        "frontend/src/components/InspectionTable.tsx": _small_component("InspectionTable"),
        "frontend/src/components/RiskScoreCard.tsx": _small_component("RiskScoreCard"),
        "frontend/src/components/DecisionPanel.tsx": _small_component("DecisionPanel"),
        "frontend/src/components/TrendChart.tsx": _small_component("TrendChart"),
        "frontend/src/components/DefectChart.tsx": _small_component("DefectChart"),
        "frontend/src/components/AIChatPanel.tsx": _small_component("AIChatPanel"),
        "frontend/src/api/mock.ts": _mock_ts(),
    }
    return files


def _theme_ts(colors: Dict[str, str]) -> str:
    c = colors
    return f"""import {{ alpha, createTheme }} from '@mui/material/styles';

export const tokens = {{
  primary: '{c["primary"]}',
  primaryDark: '{c["primary_dark"]}',
  primaryLight: '{c["primary_light"]}',
  secondary: '{c["secondary"]}',
  accent: '{c["accent"]}',
  background: '{c["background"]}',
  surface: '{c["surface"]}',
  text: '{c["text"]}',
  muted: '{c["muted"]}',
  border: '{c["border"]}',
  danger: '{c["danger"]}',
  success: '{c["success"]}',
  warning: '{c["warning"]}',
  info: '{c["info"]}',
}};

export const appTheme = createTheme({{
  palette: {{
    mode: 'light',
    primary: {{ main: tokens.primary, dark: tokens.primaryDark, light: tokens.primaryLight, contrastText: '#fff' }},
    secondary: {{ main: tokens.secondary }},
    info: {{ main: tokens.info }},
    error: {{ main: tokens.danger }},
    success: {{ main: tokens.success }},
    warning: {{ main: tokens.warning }},
    divider: tokens.border,
    background: {{ default: tokens.background, paper: tokens.surface }},
    text: {{ primary: tokens.text, secondary: tokens.muted }},
  }},
  typography: {{
    fontFamily: '"Inter", "Segoe UI", system-ui, sans-serif',
    h4: {{ fontWeight: 700, letterSpacing: '-0.02em' }},
    h5: {{ fontWeight: 700, letterSpacing: '-0.01em' }},
    h6: {{ fontWeight: 600 }},
    subtitle2: {{ fontWeight: 600, color: tokens.muted, textTransform: 'uppercase', fontSize: '0.7rem', letterSpacing: '0.06em' }},
    button: {{ textTransform: 'none', fontWeight: 600 }},
  }},
  shape: {{ borderRadius: 12 }},
  components: {{
    MuiCssBaseline: {{
      styleOverrides: {{
        body: {{ backgroundColor: tokens.background }},
      }},
    }},
    MuiAppBar: {{
      styleOverrides: {{
        root: {{
          backgroundColor: tokens.surface,
          color: tokens.text,
          borderBottom: `1px solid ${{tokens.border}}`,
        }},
      }},
    }},
    MuiDrawer: {{
      styleOverrides: {{
        paper: {{
          backgroundColor: tokens.primaryDark,
          color: '#E2E8F0',
          borderRight: 'none',
        }},
      }},
    }},
    MuiListItemButton: {{
      styleOverrides: {{
        root: {{
          borderRadius: 10,
          marginBottom: 4,
          '&.Mui-selected': {{
            backgroundColor: alpha(tokens.primaryLight, 0.35),
            color: '#fff',
            '&:hover': {{ backgroundColor: alpha(tokens.primaryLight, 0.45) }},
          }},
          '&:hover': {{ backgroundColor: alpha('#fff', 0.08) }},
        }},
      }},
    }},
    MuiCard: {{
      defaultProps: {{ elevation: 0 }},
      styleOverrides: {{
        root: {{
          border: `1px solid ${{tokens.border}}`,
          boxShadow: '0 1px 3px rgba(15, 23, 42, 0.06)',
        }},
      }},
    }},
    MuiTableHead: {{
      styleOverrides: {{
        root: {{
          backgroundColor: alpha(tokens.primary, 0.06),
          '& .MuiTableCell-head': {{ fontWeight: 700, color: tokens.text }},
        }},
      }},
    }},
    MuiTableCell: {{
      styleOverrides: {{ root: {{ padding: '10px 14px', fontSize: 13, borderColor: tokens.border }} }},
    }},
    MuiTableRow: {{
      styleOverrides: {{ root: {{ '&:hover': {{ backgroundColor: alpha(tokens.primary, 0.04) }} }} }},
    }},
    MuiButton: {{
      styleOverrides: {{
        containedPrimary: {{ boxShadow: 'none', '&:hover': {{ boxShadow: '0 2px 8px rgba(11, 61, 111, 0.25)' }} }},
      }},
    }},
    MuiChip: {{ styleOverrides: {{ root: {{ fontWeight: 600, fontSize: '0.75rem' }} }} }},
  }},
}});

export default appTheme;
"""


def _app_tsx(quality: bool) -> str:
    extra = """
        <Route path="suppliers" element={<ResourceListPage resource="suppliers" title="Suppliers" />} />
        <Route path="suppliers/:id" element={<SupplierDetailPage />} />
        <Route path="supplier-quality" element={<ResourceListPage resource="suppliers" title="Supplier Quality" />} />
        <Route path="materials" element={<ResourceListPage resource="materials" title="Materials" />} />
        <Route path="purchase-orders" element={<ResourceListPage resource="purchase_orders" title="Purchase Orders" />} />
        <Route path="incoming-lots" element={<ResourceListPage resource="incoming_lots" title="Incoming Lots" />} />
        <Route path="incoming-lots/:id" element={<LotDetailPage />} />
        <Route path="inspections" element={<InspectionPage />} />
        <Route path="defects" element={<ResourceListPage resource="defects" title="Defects" />} />
        <Route path="releases" element={<ResourceListPage resource="release_decisions" title="Release Decisions" />} />
        <Route path="capa" element={<CapaPage />} />
        <Route path="ai" element={<AIAssistantPage />} />
        <Route path="reports" element={<ReportsPage />} />""" if quality else """
        <Route path="records" element={<ResourceListPage resource="incoming_lots" title="Records" />} />
        <Route path="ai" element={<AIAssistantPage />} />
        <Route path="reports" element={<ReportsPage />} />"""
    return """import { Navigate, Route, Routes } from 'react-router-dom';
import AppShell from './layout/AppShell';
import DashboardPage from './pages/DashboardPage';
import ResourceListPage from './pages/ResourceListPage';
import SupplierDetailPage from './pages/SupplierDetailPage';
import LotDetailPage from './pages/LotDetailPage';
import InspectionPage from './pages/InspectionPage';
import CapaPage from './pages/CapaPage';
import ReportsPage from './pages/ReportsPage';
import AIAssistantPage from './pages/AIAssistantPage';

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<DashboardPage />} />""" + extra + """
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
"""


def _page_header_tsx() -> str:
    return r"""import { Box, Breadcrumbs, Link, Typography } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';

type Props = { title: string; subtitle?: string; crumbs?: { label: string; to?: string }[] };

export default function PageHeader({ title, subtitle, crumbs }: Props) {
  return (
    <Box sx={{ mb: 3 }}>
      {crumbs && crumbs.length > 0 && (
        <Breadcrumbs sx={{ mb: 1, fontSize: 13 }}>
          {crumbs.map((c) =>
            c.to ? (
              <Link key={c.label} component={RouterLink} to={c.to} underline="hover" color="inherit">
                {c.label}
              </Link>
            ) : (
              <Typography key={c.label} color="text.secondary" variant="body2">{c.label}</Typography>
            ),
          )}
        </Breadcrumbs>
      )}
      <Typography variant="h4" sx={{ mb: subtitle ? 0.5 : 0 }}>{title}</Typography>
      {subtitle && <Typography variant="body2" color="text.secondary">{subtitle}</Typography>}
    </Box>
  );
}
"""


def _shell_tsx(title: str, nav: List[Tuple[str, str]]) -> str:
    nav_js = ",\n  ".join("{ to: '%s', label: '%s' }" % (p, l.replace("'", "\\'")) for p, l in nav)
    safe = title.replace("\\", "\\\\").replace("'", "\\'")
    raw = """import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { AppBar, Box, Drawer, List, ListItemButton, ListItemText, Toolbar, Typography } from '@mui/material';
import { tokens } from '../theme';

const DRAWER_WIDTH = 272;

const NAV = [
  __NAV_ITEMS__,
];

function navSelected(pathname: string, to: string) {
  if (to === '/') return pathname === '/';
  return pathname === to || pathname.startsWith(`${to}/`);
}

export default function AppShell() {
  const location = useLocation();
  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      <Drawer
        variant="permanent"
        sx={{
          width: DRAWER_WIDTH,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { width: DRAWER_WIDTH, boxSizing: 'border-box', pt: 2, px: 1.5 },
        }}
      >
        <Box sx={{ px: 1.5, pb: 2, borderBottom: '1px solid rgba(255,255,255,0.12)', mb: 2 }}>
          <Typography variant="overline" sx={{ color: 'rgba(255,255,255,0.65)', letterSpacing: 1.2 }}>Quality Operations</Typography>
          <Typography variant="subtitle1" sx={{ color: '#fff', fontWeight: 700, lineHeight: 1.3, mt: 0.5 }}>Incoming Material</Typography>
        </Box>
        <List dense disablePadding>
          {NAV.map((item) => (
            <ListItemButton
              key={item.to}
              component={NavLink}
              to={item.to}
              selected={navSelected(location.pathname, item.to)}
            >
              <ListItemText primary={item.label} primaryTypographyProps={{ fontSize: 14, fontWeight: 500 }} />
            </ListItemButton>
          ))}
        </List>
      </Drawer>
      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <AppBar position="sticky" elevation={0} sx={{ zIndex: 1100 }}>
          <Toolbar sx={{ gap: 2, minHeight: 64 }}>
            <Typography variant="h6" sx={{ flex: 1, fontWeight: 700, color: 'text.primary' }}>__APP_TITLE__</Typography>
          </Toolbar>
        </AppBar>
        <Box component="main" sx={{ flexGrow: 1, p: { xs: 2, md: 3 }, maxWidth: 1440, width: '100%', mx: 'auto' }}>
          <Outlet />
        </Box>
        <Box component="footer" sx={{ py: 1.5, px: 3, borderTop: 1, borderColor: 'divider', bgcolor: 'background.paper' }}>
          <Typography variant="caption" color="text.secondary">Automotive supplier quality · Incoming inspection &amp; release · IATF-aligned workflow</Typography>
        </Box>
      </Box>
    </Box>
  );
}
"""
    return raw.replace("__NAV_ITEMS__", nav_js).replace("__APP_TITLE__", safe)


def _auth_ts() -> str:
    return """const VALID_ROLES = ['ADMIN', 'QUALITY_MANAGER', 'INSPECTOR', 'VIEWER'] as const;

export function getToken(): string | null {
  return localStorage.getItem('access_token');
}

export function getRole(): string {
  const raw = localStorage.getItem('role');
  if (!raw || raw === 'undefined' || raw === 'null') return 'VIEWER';
  const up = raw.toUpperCase();
  return (VALID_ROLES as readonly string[]).includes(up) ? up : 'VIEWER';
}

export function setAuth(token: string, role?: string | null) {
  localStorage.setItem('access_token', token);
  const r = (role || 'VIEWER').toString().toUpperCase();
  localStorage.setItem('role', (VALID_ROLES as readonly string[]).includes(r) ? r : 'VIEWER');
}

export function clearAuth() {
  localStorage.removeItem('access_token');
  localStorage.removeItem('role');
}

export function canDecide(): boolean {
  return ['ADMIN', 'QUALITY_MANAGER'].includes(getRole());
}

export function canInspect(): boolean {
  return ['ADMIN', 'QUALITY_MANAGER', 'INSPECTOR'].includes(getRole());
}
"""


def _login_tsx(title: str) -> str:
    safe = title.replace("\\", "\\\\").replace("'", "\\'")
    raw = """import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Alert, Box, Button, Card, CardContent, TextField, Typography } from '@mui/material';
import { apiFetch } from '../api/client';
import { setAuth } from '../auth';
import { tokens } from '../theme';

export default function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState('admin@example.com');
  const [password, setPassword] = useState('admin123');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true); setError('');
    try {
      const res = await apiFetch<{ access_token: string; role: string }>('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) });
      setAuth(res.access_token, res.role);
      navigate('/');
    } catch (err: any) { setError(err?.message || 'Login failed'); }
    finally { setLoading(false); }
  };
  return (
    <Box sx={{ minHeight: '100vh', display: 'grid', placeItems: 'center', p: 2, background: `linear-gradient(145deg, ${tokens.primaryDark} 0%, ${tokens.primary} 45%, ${tokens.secondary} 100%)` }}>
      <Card sx={{ width: 440, maxWidth: '100%', boxShadow: 6 }}>
        <CardContent sx={{ p: 3 }}>
          <Typography variant="overline" color="primary" sx={{ fontWeight: 700 }}>Plant quality portal</Typography>
          <Typography variant="h5" sx={{ mb: 1, fontWeight: 700 }}>__APP_TITLE__</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>Sign in to manage incoming lots, inspections, and release decisions.</Typography>
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          <Box component="form" onSubmit={submit} sx={{ display: 'grid', gap: 2 }}>
            <TextField label="Email" value={email} onChange={(e) => setEmail(e.target.value)} fullWidth />
            <TextField label="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} fullWidth />
            <Button type="submit" variant="contained" disabled={loading}>{loading ? 'Signing in…' : 'Sign in'}</Button>
            <Typography variant="caption" color="text.secondary">Demo: admin@example.com / admin123</Typography>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
"""
    return raw.replace("__APP_TITLE__", safe)


def _dashboard_tsx() -> str:
    return r"""import { useQuery } from '@tanstack/react-query';
import { Box, Card, CardContent, CircularProgress, Typography } from '@mui/material';
import KPICard from '../components/KPICard';
import LotTable from '../components/LotTable';
import TrendChart from '../components/TrendChart';
import DefectChart from '../components/DefectChart';
import PageHeader from '../components/PageHeader';
import { apiFetch } from '../api/client';

export default function DashboardPage() {
  const { data, isLoading, error } = useQuery({ queryKey: ['dashboard'], queryFn: () => apiFetch<any>('/api/dashboard/kpis') });
  if (isLoading) return <Box sx={{ display: 'grid', placeItems: 'center', py: 8 }}><CircularProgress /></Box>;
  if (error) return <Typography color="error">Failed to load dashboard.</Typography>;
  const k = data || {};
  const cards: [string, any, string][] = [
    ['Incoming lots', k.total_lots, 'primary.main'], ['Pending inspections', k.pending_inspections, 'warning.main'],
    ['Released', k.released, 'success.main'], ['Held', k.held, 'warning.dark'], ['Rejected', k.rejected, 'error.main'],
    ['Supplier PPM', k.supplier_ppm, 'info.main'], ['Defect rate', k.defect_rate, 'error.main'],
    ['Release rate', k.release_rate, 'success.main'], ['High-risk lots', k.high_risk_lots, 'error.main'], ['Open CAPA', k.open_capa, 'secondary.main'],
  ];
  return (
    <Box>
      <PageHeader title="Operations dashboard" subtitle="Real-time incoming material quality KPIs and risk signals" />
      <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr 1fr', sm: 'repeat(3, 1fr)', lg: 'repeat(5, 1fr)' } }}>
        {cards.map(([label, value, accent]) => <KPICard key={label} label={label} value={value ?? '—'} accent={accent} />)}
      </Box>
      <Box sx={{ display: 'grid', gap: 2, mt: 3, gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' } }}>
        <Card><CardContent><Typography variant="subtitle2" sx={{ mb: 1 }}>Supplier PPM trend</Typography><TrendChart data={k.ppm_trend || []} /></CardContent></Card>
        <Card><CardContent><Typography variant="subtitle2" sx={{ mb: 1 }}>Defect trend</Typography><TrendChart data={k.defect_trend || []} /></CardContent></Card>
        <Card><CardContent><Typography variant="subtitle2" sx={{ mb: 1 }}>Release / rejection</Typography><TrendChart data={k.release_trend || []} dataKey="released" /></CardContent></Card>
        <Card><CardContent><Typography variant="subtitle2" sx={{ mb: 1 }}>Top defect categories</Typography><DefectChart data={k.top_defects || []} /></CardContent></Card>
      </Box>
      <Card sx={{ mt: 3 }}><CardContent><Typography variant="subtitle2" sx={{ mb: 2 }}>High-risk incoming lots</Typography><LotTable rows={k.high_risk_table || []} /></CardContent></Card>
    </Box>
  );
}
"""


def _resource_list_tsx() -> str:
    return r"""import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Box, Card, CardContent, Table, TableBody, TableCell, TableHead, TablePagination, TableRow, TextField, Typography, TableContainer } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { apiFetch } from '../api/client';
import StatusChip from '../components/StatusChip';
import PageHeader from '../components/PageHeader';

export default function ResourceListPage({ resource, title }: { resource: string; title: string }) {
  const navigate = useNavigate();
  const [q, setQ] = useState('');
  const [page, setPage] = useState(0);
  const path = resource === 'purchase_orders' ? 'purchase-orders' : resource === 'incoming_lots' ? 'incoming-lots' : resource === 'release_decisions' ? 'release-decisions' : resource;
  const { data, isLoading, error } = useQuery({ queryKey: [resource], queryFn: () => apiFetch<any>(`/api/${path}`) });
  const rows = data?.items || data || [];
  const filtered = useMemo(() => {
    const s = q.toLowerCase();
    return (Array.isArray(rows) ? rows : []).filter((r: any) => JSON.stringify(r).toLowerCase().includes(s));
  }, [rows, q]);
  const paged = filtered.slice(page * 10, page * 10 + 10);
  const cols = paged[0] ? Object.keys(paged[0]).filter((k) => k !== 'id').slice(0, 8) : [];
  return (
    <Box>
      <PageHeader title={title} subtitle="Search, filter, and open records" />
      <TextField size="small" placeholder="Search records…" value={q} onChange={(e) => { setQ(e.target.value); setPage(0); }} sx={{ mb: 2, width: 360, bgcolor: 'background.paper' }} />
      <Card><CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}>
        {isLoading && <Typography>Loading…</Typography>}
        {error && <Typography color="error">Failed to load.</Typography>}
        {!isLoading && !filtered.length && <Typography color="text.secondary">No records.</Typography>}
        {!!paged.length && (
          <>
            <TableContainer>
            <Table size="small">
              <TableHead><TableRow>{cols.map((c) => <TableCell key={c}>{c.replace(/_/g, ' ')}</TableCell>)}</TableRow></TableHead>
              <TableBody>
                {paged.map((row: any) => (
                  <TableRow key={row.id} hover onClick={() => {
                    if (resource === 'suppliers') navigate(`/suppliers/${row.id}`);
                    if (resource === 'incoming_lots') navigate(`/incoming-lots/${row.id}`);
                  }}>
                    {cols.map((c) => (
                      <TableCell key={c}>{['status', 'inspection_status', 'release_status', 'result'].includes(c) ? <StatusChip status={String(row[c])} /> : String(row[c] ?? '')}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            </TableContainer>
            <TablePagination component="div" count={filtered.length} page={page} onPageChange={(_, p) => setPage(p)} rowsPerPage={10} rowsPerPageOptions={[10]} />
          </>
        )}
      </CardContent></Card>
    </Box>
  );
}
"""


def _supplier_detail_tsx() -> str:
    return r"""import { useQuery } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import { Box, Card, CardContent, Typography } from '@mui/material';
import { apiFetch } from '../api/client';
import SupplierScore from '../components/SupplierScore';
import TrendChart from '../components/TrendChart';
import StatusChip from '../components/StatusChip';

export default function SupplierDetailPage() {
  const { id } = useParams();
  const { data, isLoading } = useQuery({ queryKey: ['supplier', id], queryFn: () => apiFetch<any>(`/api/suppliers/${id}`) });
  if (isLoading) return <Typography>Loading supplier…</Typography>;
  if (!data) return <Typography>Supplier not found.</Typography>;
  return (
    <Box>
      <Typography variant="h5">{data.name}</Typography>
      <Typography color="text.secondary" sx={{ mb: 2 }}>{data.code} · {data.location} · {data.category}</Typography>
      <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', md: '1fr 1fr 1fr' } }}>
        <Card><CardContent><Typography variant="subtitle2">Quality score</Typography><SupplierScore score={data.quality_score} /></CardContent></Card>
        <Card><CardContent><Typography variant="subtitle2">PPM</Typography><Typography variant="h5">{data.ppm}</Typography></CardContent></Card>
        <Card><CardContent><Typography variant="subtitle2">Status</Typography><StatusChip status={data.status} /></CardContent></Card>
      </Box>
      <Box sx={{ display: 'grid', gap: 2, mt: 2, gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' } }}>
        <Card><CardContent><Typography variant="subtitle2">PPM trend</Typography><TrendChart data={data.ppm_trend || []} /></CardContent></Card>
        <Card><CardContent><Typography variant="subtitle2">Defect trend</Typography><TrendChart data={data.defect_trend || []} /></CardContent></Card>
      </Box>
    </Box>
  );
}
"""


def _lot_detail_tsx() -> str:
    return r"""import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import { Box, Card, CardContent, Typography } from '@mui/material';
import { apiFetch } from '../api/client';
import InspectionTable from '../components/InspectionTable';
import RiskScoreCard from '../components/RiskScoreCard';
import DecisionPanel from '../components/DecisionPanel';
import StatusChip from '../components/StatusChip';

export default function LotDetailPage() {
  const { id } = useParams();
  const qc = useQueryClient();
  const { data, isLoading } = useQuery({ queryKey: ['lot', id], queryFn: () => apiFetch<any>(`/api/incoming-lots/${id}`) });
    const decide = useMutation({
    mutationFn: ({ decision, reason }: { decision: string; reason: string }) => {
      if (decision.includes('CAPA')) {
        return apiFetch('/api/capa', { method: 'POST', body: JSON.stringify({ lot_id: Number(id), title: reason || `CAPA for lot ${id}`, reason }) });
      }
      const path = decision === 'HOLD' ? 'hold' : decision === 'REJECT' ? 'reject' : 'release';
      return apiFetch(`/api/${path}/${id}`, { method: 'POST', body: JSON.stringify({ reason }) });
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['lot', id] }),
  });
  if (isLoading) return <Typography>Loading lot…</Typography>;
  if (!data) return <Typography>Lot not found.</Typography>;
  return (
    <Box>
      <Typography variant="h5">{data.lot_number}</Typography>
      <Box sx={{ display: 'flex', gap: 1, my: 1 }}><StatusChip status={data.inspection_status} /><StatusChip status={data.release_status} /></Box>
      <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', md: '2fr 1fr' } }}>
        <Box>
          <Card sx={{ mb: 2 }}><CardContent>
            <Typography variant="subtitle2">Receiving</Typography>
            <Typography variant="body2">Supplier: {data.supplier} · Material: {data.material} · Qty: {data.quantity} · Received: {data.received_date}</Typography>
          </CardContent></Card>
          <Card><CardContent><Typography variant="subtitle2" sx={{ mb: 1 }}>Inspection results</Typography><InspectionTable rows={data.inspections || []} /></CardContent></Card>
        </Box>
        <Box>
          <RiskScoreCard score={data.risk_score} recommendation={data.recommendation} />
          <Card sx={{ mt: 2 }}><CardContent><Typography variant="subtitle2">AI recommendation</Typography><Typography variant="body2">{data.ai_recommendation}</Typography></CardContent></Card>
          <Box sx={{ mt: 2 }}><DecisionPanel onDecision={(decision, reason) => decide.mutate({ decision, reason })} /></Box>
        </Box>
      </Box>
    </Box>
  );
}
"""


def _inspection_tsx() -> str:
    return r"""import { useState } from 'react';
import { Alert, Box, Button, Card, CardContent, Divider, Stack, TextField, Typography } from '@mui/material';
import InspectionTable from '../components/InspectionTable';
import PageHeader from '../components/PageHeader';
import { apiFetch } from '../api/client';
import { canInspect } from '../auth';

function resultFor(actual: number, lower: number, upper: number) {
  if (actual < lower || actual > upper) return 'FAIL';
  const warn = (upper - lower) * 0.1;
  if (actual <= lower + warn || actual >= upper - warn) return 'WARNING';
  return 'PASS';
}

export default function InspectionPage() {
  const [rows, setRows] = useState<any[]>([
    { parameter: 'Diameter', specification: '280 ± 0.5 mm', lower_limit: 279.5, upper_limit: 280.5, actual_value: 280.2, unit: 'mm', result: 'PASS', inspector: 'J. Patel' },
  ]);
  const [actual, setActual] = useState('280.2');
  const add = async () => {
    const value = Number(actual);
    const row = { ...rows[0], actual_value: value, result: resultFor(value, 279.5, 280.5), inspection_date: new Date().toISOString() };
    setRows([row, ...rows]);
    await apiFetch('/api/inspections', { method: 'POST', body: JSON.stringify({ lot_id: 1, ...row }) });
  };
  return (
    <Box>
      <PageHeader title="Incoming Inspection" subtitle="Record dimensional checks — PASS / WARNING / FAIL from specification limits" />
      <Card sx={{ mb: 3 }}><CardContent>
        <Alert severity="info" sx={{ mb: 2 }}>Diameter spec <strong>280 ± 0.5 mm</strong>. Example: 280.2 → PASS, 281.2 → FAIL.</Alert>
        {canInspect() && (
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5} alignItems={{ sm: 'center' }}>
            <TextField size="small" label="Actual value (mm)" value={actual} onChange={(e) => setActual(e.target.value)} sx={{ minWidth: 200 }} />
            <Button variant="contained" onClick={add}>Record inspection</Button>
          </Stack>
        )}
      </CardContent></Card>
      <Card><CardContent>
        <Typography variant="subtitle2" sx={{ mb: 2 }}>Inspection results</Typography>
        <Divider sx={{ mb: 2 }} />
        <InspectionTable rows={rows} />
      </CardContent></Card>
    </Box>
  );
}
"""


def _capa_tsx() -> str:
    return r"""import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Box, Button, Card, CardContent, TextField, Typography } from '@mui/material';
import { apiFetch } from '../api/client';
import { canDecide } from '../auth';

export default function CapaPage() {
  const qc = useQueryClient();
  const [title, setTitle] = useState('');
  const { data, isLoading } = useQuery({ queryKey: ['capa'], queryFn: () => apiFetch<any>('/api/capa') });
  const create = useMutation({
    mutationFn: () => apiFetch('/api/capa', { method: 'POST', body: JSON.stringify({ title, status: 'OPEN' }) }),
    onSuccess: () => { setTitle(''); qc.invalidateQueries({ queryKey: ['capa'] }); },
  });
  const items = data?.items || data || [];
  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 2 }}>CAPA</Typography>
      {canDecide() && (
        <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
          <TextField size="small" fullWidth label="New CAPA" value={title} onChange={(e) => setTitle(e.target.value)} />
          <Button variant="contained" disabled={!title} onClick={() => create.mutate()}>Create</Button>
        </Box>
      )}
      {isLoading && <Typography>Loading…</Typography>}
      {(items as any[]).map((c) => (
        <Card key={c.id} sx={{ mb: 1 }}><CardContent><Typography>{c.title}</Typography><Typography variant="caption">{c.status}</Typography></CardContent></Card>
      ))}
      {!isLoading && !items.length && <Typography color="text.secondary">No CAPA records.</Typography>}
    </Box>
  );
}
"""


def _reports_tsx() -> str:
    return r"""import { Box, Button, Card, CardContent, Typography } from '@mui/material';
const REPORTS = ['Supplier Quality Report', 'Incoming Inspection Report', 'Rejected Lots Report', 'Defect Analysis', 'CAPA Report'];
function download(name: string) {
  const csv = 'report,generated\n' + name + ',' + new Date().toISOString();
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name.replace(/\s+/g, '-').toLowerCase() + '.csv'; a.click();
  URL.revokeObjectURL(url);
}
export default function ReportsPage() {
  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 2 }}>Reports</Typography>
      {REPORTS.map((r) => (
        <Card key={r} sx={{ mb: 1 }}><CardContent sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography>{r}</Typography>
          <Button onClick={() => download(r)}>Export CSV</Button>
        </CardContent></Card>
      ))}
    </Box>
  );
}
"""


def _ai_tsx() -> str:
    return r"""import { Box, Typography } from '@mui/material';
import AIChatPanel from '../components/AIChatPanel';
export default function AIAssistantPage() {
  return (<Box><Typography variant="h5" sx={{ mb: 2 }}>AI Quality Assistant</Typography><AIChatPanel /></Box>);
}
"""


def _small_component(name: str) -> str:
    parts = {
        "KPICard": """import { Card, CardContent, Typography } from '@mui/material';
export default function KPICard({ label, value, accent }: { label: string; value: string | number; accent?: string }) {
  return (
    <Card sx={{ borderLeft: 4, borderColor: accent || 'primary.main', height: '100%' }}>
      <CardContent sx={{ py: 2, '&:last-child': { pb: 2 } }}>
        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4 }}>{label}</Typography>
        <Typography variant="h5" sx={{ fontWeight: 800, mt: 0.5, color: accent || 'text.primary' }}>{value}</Typography>
      </CardContent>
    </Card>
  );
}
""",
        "StatusChip": """import { Chip } from '@mui/material';
const COLOR: Record<string, 'success' | 'warning' | 'info' | 'error' | 'default'> = {
  RELEASED: 'success', PASS: 'success', ACTIVE: 'success', HOLD: 'warning', WARNING: 'warning',
  PENDING: 'info', REJECTED: 'error', FAIL: 'error', CLOSED: 'default', OPEN: 'info',
};
export default function StatusChip({ status }: { status: string }) {
  const key = (status || '').toUpperCase();
  return <Chip size="small" variant="filled" label={key || 'UNKNOWN'} color={COLOR[key] || 'default'} />;
}
""",
        "RiskBadge": """import { Chip } from '@mui/material';
export default function RiskBadge({ score }: { score: number }) {
  const color = score >= 81 ? 'error' : score >= 61 ? 'warning' : score >= 31 ? 'info' : 'success';
  return <Chip size="small" color={color as any} label={`Risk ${score}`} />;
}
""",
        "SupplierScore": """import { Box, LinearProgress, Typography } from '@mui/material';
export default function SupplierScore({ score }: { score: number }) {
  return (<Box sx={{ minWidth: 120 }}><Typography variant="caption">{score}</Typography><LinearProgress variant="determinate" value={Math.min(100, Number(score) || 0)} /></Box>);
}
""",
        "LotTable": """import { Button, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import StatusChip from './StatusChip';
import RiskBadge from './RiskBadge';
export default function LotTable({ rows }: { rows: any[] }) {
  const navigate = useNavigate();
  if (!rows?.length) return <Typography color="text.secondary" variant="body2">No high-risk lots in queue.</Typography>;
  return (
    <TableContainer><Table size="small"><TableHead><TableRow>{['Lot Number','Supplier','Material','Qty','Risk','Inspection','Release','Action'].map((h) => <TableCell key={h}>{h}</TableCell>)}</TableRow></TableHead>
    <TableBody>{rows.map((r) => (
      <TableRow key={r.id || r.lot_number} hover>
        <TableCell sx={{ fontWeight: 600 }}>{r.lot_number}</TableCell><TableCell>{r.supplier}</TableCell><TableCell>{r.material}</TableCell><TableCell>{r.quantity}</TableCell>
        <TableCell><RiskBadge score={Number(r.risk_score || 0)} /></TableCell>
        <TableCell><StatusChip status={r.inspection_status} /></TableCell>
        <TableCell><StatusChip status={r.release_status} /></TableCell>
        <TableCell><Button size="small" variant="outlined" onClick={() => navigate(`/incoming-lots/${r.id}`)}>Open</Button></TableCell>
      </TableRow>
    ))}</TableBody></Table></TableContainer>
  );
}
""",
        "InspectionTable": """import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material';
import StatusChip from './StatusChip';
export default function InspectionTable({ rows }: { rows: any[] }) {
  if (!rows?.length) return <Typography color="text.secondary" variant="body2">No inspection records yet.</Typography>;
  return (
    <TableContainer><Table size="small"><TableHead><TableRow>{['Parameter','Specification','LSL','USL','Actual','Unit','Result','Inspector'].map((h) => <TableCell key={h}>{h}</TableCell>)}</TableRow></TableHead>
    <TableBody>{rows.map((r, i) => (
      <TableRow key={r.id || i} hover><TableCell sx={{ fontWeight: 600 }}>{r.parameter}</TableCell><TableCell>{r.specification}</TableCell><TableCell>{r.lower_limit}</TableCell><TableCell>{r.upper_limit}</TableCell><TableCell>{r.actual_value}</TableCell><TableCell>{r.unit}</TableCell><TableCell><StatusChip status={r.result} /></TableCell><TableCell>{r.inspector}</TableCell></TableRow>
    ))}</TableBody></Table></TableContainer>
  );
}
""",
        "RiskScoreCard": """import { Card, CardContent, Typography } from '@mui/material';
import RiskBadge from './RiskBadge';
export default function RiskScoreCard({ score, recommendation }: { score: number; recommendation: string }) {
  return (<Card><CardContent><Typography variant="subtitle2">Risk score</Typography><RiskBadge score={Number(score) || 0} /><Typography sx={{ mt: 1 }} variant="body2">{recommendation}</Typography></CardContent></Card>);
}
""",
        "DecisionPanel": """import { Box, Button, TextField } from '@mui/material';
import { useState } from 'react';
import { canDecide } from '../auth';
export default function DecisionPanel({ onDecision }: { onDecision: (decision: string, reason: string) => void }) {
  const [reason, setReason] = useState('');
  if (!canDecide()) return null;
  return (
    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
      <TextField size="small" label="Reason" value={reason} onChange={(e) => setReason(e.target.value)} sx={{ minWidth: 240 }} />
      {['RELEASE', 'HOLD', 'REJECT', 'CREATE CAPA'].map((d) => (
        <Button key={d} variant={d === 'RELEASE' ? 'contained' : 'outlined'} color={d === 'REJECT' ? 'error' : 'primary'} onClick={() => onDecision(d, reason)}>{d}</Button>
      ))}
    </Box>
  );
}
""",
        "TrendChart": """import { useTheme } from '@mui/material/styles';
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from 'recharts';
export default function TrendChart({ data, dataKey = 'value' }: { data: any[]; dataKey?: string }) {
  const theme = useTheme();
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data || []}>
        <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
        <XAxis dataKey="label" tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 12 }} />
        <Tooltip contentStyle={{ borderRadius: 8, border: `1px solid ${theme.palette.divider}` }} />
        <Line type="monotone" dataKey={dataKey} stroke={theme.palette.primary.main} strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}
""",
        "DefectChart": """import { useTheme } from '@mui/material/styles';
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from 'recharts';
export default function DefectChart({ data }: { data: any[] }) {
  const theme = useTheme();
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data || []}>
        <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
        <XAxis dataKey="label" tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 12 }} />
        <Tooltip contentStyle={{ borderRadius: 8, border: `1px solid ${theme.palette.divider}` }} />
        <Bar dataKey="value" fill={theme.palette.error.main} radius={[6, 6, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
""",
        "AIChatPanel": """import { Box, Button, Paper, TextField, Typography } from '@mui/material';
import { useState } from 'react';
import { apiFetch } from '../api/client';
export default function AIChatPanel() {
  const [q, setQ] = useState('Why is this lot on hold?');
  const [a, setA] = useState('');
  const [loading, setLoading] = useState(false);
  const ask = async () => {
    setLoading(true);
    try { const res = await apiFetch<{ answer: string }>('/api/ai/ask', { method: 'POST', body: JSON.stringify({ question: q }) }); setA(res.answer); }
    catch (e: any) { setA(e?.message || 'Unable to answer'); }
    finally { setLoading(false); }
  };
  return (<Paper sx={{ p: 2 }}><Typography variant="h6" sx={{ mb: 2 }}>AI Quality Assistant</Typography><Box sx={{ display: 'flex', gap: 1 }}><TextField fullWidth size="small" value={q} onChange={(e) => setQ(e.target.value)} /><Button variant="contained" onClick={ask} disabled={loading}>{loading ? '…' : 'Ask'}</Button></Box>{a && <Typography sx={{ mt: 2 }} variant="body2">{a}</Typography>}</Paper>);
}
""",
    }
    return parts[name]


def _mock_ts() -> str:
    return r"""type Json = Record<string, unknown>;
const now = () => new Date().toISOString();
const suppliers = [
  { id: 1, code: 'SUP-001', name: 'ABC Auto Components', location: 'Detroit, MI', category: 'Casting', quality_score: 92, ppm: 42, defect_rate: 0.8, rejected_lots: 2, open_capa: 1, status: 'ACTIVE' },
  { id: 2, code: 'SUP-002', name: 'Prime Precision', location: 'Stuttgart, DE', category: 'Machining', quality_score: 88, ppm: 67, defect_rate: 1.1, rejected_lots: 3, open_capa: 2, status: 'ACTIVE' },
  { id: 3, code: 'SUP-003', name: 'XYZ Metals', location: 'Pune, IN', category: 'Forging', quality_score: 81, ppm: 110, defect_rate: 2.4, rejected_lots: 5, open_capa: 3, status: 'ACTIVE' },
  { id: 4, code: 'SUP-004', name: 'Global Bearings', location: 'Osaka, JP', category: 'Bearings', quality_score: 95, ppm: 18, defect_rate: 0.3, rejected_lots: 0, open_capa: 0, status: 'ACTIVE' },
  { id: 5, code: 'SUP-005', name: 'Precision Forge', location: 'Monterrey, MX', category: 'Forging', quality_score: 76, ppm: 160, defect_rate: 3.1, rejected_lots: 7, open_capa: 4, status: 'HOLD' },
];
const materials = [
  { id: 1, code: 'MAT-BD', name: 'Brake Disc', type: 'Casting', specification: '280 ± 0.5 mm', criticality: 'HIGH', inspection_plan: 'CMM + visual', status: 'ACTIVE' },
  { id: 2, code: 'MAT-GB', name: 'Gear Blank', type: 'Forging', specification: 'HRC 28-32', criticality: 'MEDIUM', inspection_plan: 'Hardness + dim', status: 'ACTIVE' },
  { id: 3, code: 'MAT-BR', name: 'Bearing', type: 'Purchased', specification: 'ISO 492 P6', criticality: 'HIGH', inspection_plan: 'Sampling', status: 'ACTIVE' },
  { id: 4, code: 'MAT-SK', name: 'Steering Knuckle', type: 'Casting', specification: 'EN-GJS-500', criticality: 'HIGH', inspection_plan: 'X-ray + dim', status: 'ACTIVE' },
  { id: 5, code: 'MAT-SB', name: 'Suspension Bracket', type: 'Stamping', specification: 'S355', criticality: 'MEDIUM', inspection_plan: 'Visual + weld', status: 'ACTIVE' },
];
const lots = Array.from({ length: 24 }).map((_, i) => ({
  id: i + 1,
  lot_number: `LOT-20260925-${String(i + 1).padStart(3, '0')}`,
  po_number: `PO-100${i}`,
  supplier: suppliers[i % suppliers.length].name,
  material: materials[i % materials.length].name,
  quantity: 80 + i * 10,
  received_date: '2026-09-21',
  inspection_status: i % 5 === 0 ? 'PENDING' : i % 5 === 1 ? 'FAIL' : 'PASS',
  risk_score: 18 + (i * 7) % 80,
  release_status: i % 7 === 0 ? 'HOLD' : i % 7 === 1 ? 'REJECTED' : i % 7 === 2 ? 'PENDING' : 'RELEASED',
}));
const inspections = [
  { id: 1, lot_id: 1, parameter: 'Diameter', specification: '280 ± 0.5 mm', lower_limit: 279.5, upper_limit: 280.5, actual_value: 280.2, unit: 'mm', result: 'PASS', inspector: 'J. Patel', inspection_date: now() },
  { id: 2, lot_id: 1, parameter: 'Diameter', specification: '280 ± 0.5 mm', lower_limit: 279.5, upper_limit: 280.5, actual_value: 281.2, unit: 'mm', result: 'FAIL', inspector: 'J. Patel', inspection_date: now() },
];
const defects = [
  { id: 1, lot_id: 2, category: 'Porosity', description: 'Rim porosity on brake disc', severity: 'MAJOR' },
  { id: 2, lot_id: 5, category: 'Dimensional', description: 'Out of round', severity: 'CRITICAL' },
];
const capa = [
  { id: 1, title: 'Containment for LOT-20260925-001 porosity', status: 'OPEN', supplier: 'ABC Auto Components' },
  { id: 2, title: 'Supplier 8D for dimensional drift', status: 'OPEN', supplier: 'Precision Forge' },
];
const trend = ['W1','W2','W3','W4','W5','W6'].map((label, i) => ({ label, value: 40 + i * 6, released: 20 + i, rejected: 3 }));
const kpis = {
  total_lots: 200, pending_inspections: 18, released: 142, held: 24, rejected: 16,
  supplier_ppm: 64, defect_rate: '1.4%', release_rate: '71%', high_risk_lots: 9, open_capa: 9,
  ppm_trend: trend, defect_trend: trend, release_trend: trend,
  top_defects: [{ label: 'Porosity', value: 42 }, { label: 'Dimensional', value: 31 }, { label: 'Surface', value: 18 }],
  high_risk_table: lots.filter((l) => l.risk_score >= 61).slice(0, 8),
};
function rec(path: string) {
  const clean = path.split('?')[0].replace(/^\/api\//, '/').replace(/^\//, '');
  const parts = clean.split('/').filter(Boolean);
  return { key: parts[0] || '', id: parts[1], rest: parts.slice(2) };
}
function recFor(key: string) {
  if (key === 'suppliers') return suppliers;
  if (key === 'materials') return materials;
  if (key === 'purchase-orders' || key === 'purchase_orders') return lots.map((l, i) => ({ id: i + 1, po_number: l.po_number, supplier: l.supplier, material: l.material, status: 'OPEN' }));
  if (key === 'incoming-lots' || key === 'incoming_lots') return lots;
  if (key === 'inspections') return inspections;
  if (key === 'defects') return defects;
  if (key === 'capa') return capa;
  if (key === 'release-decisions' || key === 'release_decisions') return lots.map((l, i) => ({ id: i + 1, lot: l.lot_number, decision: l.release_status, risk_score: l.risk_score, user: 'quality@example.com', timestamp: now() }));
  return [];
}
export async function mockFetch<T = unknown>(path: string, options: RequestInit = {}): Promise<T> {
  const method = (options.method || 'GET').toUpperCase();
  const { key, id, rest } = rec(path);
  if (key === 'auth' && rest[0] === 'login') return { access_token: 'mock-jwt', token_type: 'bearer', role: 'ADMIN' } as T;
  if (key === 'dashboard') return kpis as T;
  if (key === 'ai') {
    const body = JSON.parse(String(options.body || '{}'));
    const hold = lots.find((l) => l.release_status === 'HOLD');
    return { answer: `Grounded from application data: ${body.question || ''}. Example ${hold?.lot_number} is HOLD with risk ${hold?.risk_score}. Top defects: porosity, dimensional, surface.` } as T;
  }
  const list = recFor(key) as Json[];
  if (method === 'GET' && !id) return { items: list, total: list.length } as T;
  if (method === 'GET' && id) {
    const found = list.find((r) => String(r.id) === String(id)) || list[0] || { id };
    if (key === 'suppliers') return { ...found, ppm_trend: trend, defect_trend: trend } as T;
    if (key === 'incoming-lots') {
      const score = Number((found as any).risk_score || 40);
      const recs = score <= 30 ? 'AUTO RELEASE' : score <= 60 ? 'NORMAL INSPECTION' : score <= 80 ? 'QUALITY REVIEW' : 'HOLD / REJECT';
      return { ...found, inspections, defects, recommendation: recs, ai_recommendation: `Mandatory rules still apply. Suggested: ${recs}.` } as T;
    }
    return found as T;
  }
  if (method === 'POST') {
    let body: Json = {};
    try { body = JSON.parse(String(options.body || '{}')); } catch { body = {}; }
    return { id: Date.now(), ...body, created_at: now(), mocked: true } as T;
  }
  return { ok: true, mocked: true, path, method } as T;
}
"""

