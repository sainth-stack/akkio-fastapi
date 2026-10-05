"""Anomaly Detection app generator for Agentic Builder.

Generates a complete anomaly detection monitoring platform with:
- Dashboard with time-series metric charts and anomaly markers (Recharts)
- Anomaly feed with severity classification and filtering
- Alert threshold configuration per metric
- Analytics reports with temporal and severity breakdown
"""
from __future__ import annotations

from typing import Dict


# ──────────────────────────────────────────────────────────────────────────────
# Domain detection
# ──────────────────────────────────────────────────────────────────────────────

ANOMALY_KEYWORDS = (
    "anomaly detection", "anomaly monitoring", "anomaly alert",
    "outlier detection", "outlier monitoring",
    "threshold alert", "threshold breach",
    "time series anomaly", "metric monitoring",
    "z-score detection", "zscore anomaly",
    "infrastructure monitoring", "observability platform",
    "predictive monitoring", "anomaly detection system",
    "anomaly detection platform", "anomaly detection app",
)

ANOMALY_SUPPORTING = (
    "anomaly", "anomalies", "outlier", "outliers",
    "threshold", "metric", "metrics",
    "alert", "alerts", "monitoring",
    "time series", "time-series",
    "cpu", "memory", "latency", "error rate",
    "detection", "sensor", "iot", "telemetry",
    "z-score", "zscore", "severity",
)


def is_anomaly_detection_domain(requirement: str, prd: str = "", uiux: str = "") -> bool:
    text = "\n".join([requirement or "", prd or "", uiux or ""]).lower()
    strong = sum(1 for k in ANOMALY_KEYWORDS if k in text)
    if strong >= 1:
        return True
    support = sum(1 for k in ANOMALY_SUPPORTING if k in text)
    return support >= 4


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def anomaly_detection_frontend_files(
    title: str,
    colors: Dict[str, str],
    requirement: str = "",
    prd: str = "",
) -> Dict[str, str]:
    """Return frontend file dict for anomaly detection app."""
    primary = colors.get("primary", "#1565C0")
    primary_dark = colors.get("primary_dark", "#0D47A1")
    secondary = colors.get("secondary", "#7B1FA2")
    bg = colors.get("background", "#F5F5F5")
    surface = colors.get("surface", "#FFFFFF")
    text_color = colors.get("text", "#212121")
    muted = colors.get("muted", "#757575")
    safe_title = title.replace("'", "\\'")

    return {
        "frontend/src/pages/DashboardPage.tsx": _dashboard_tsx(primary, primary_dark, bg, surface, text_color, muted, safe_title),
        "frontend/src/pages/AnomalyFeedPage.tsx": _anomaly_feed_tsx(primary, primary_dark, bg, surface, text_color, muted),
        "frontend/src/pages/AlertConfigPage.tsx": _alert_config_tsx(primary, primary_dark, bg, surface, text_color, muted),
        "frontend/src/pages/ReportsPage.tsx": _reports_tsx(primary, primary_dark, secondary, bg, surface, text_color, muted),
        "frontend/src/App.tsx": _app_tsx(),
        "frontend/src/layout/AppLayout.tsx": _app_layout_tsx(primary, primary_dark, bg, surface, text_color, muted, safe_title),
        "frontend/src/pages/LoginPage.tsx": _login_tsx(primary, primary_dark, bg, surface),
        "frontend/src/api/mock.ts": _mock_ts(safe_title),
    }


def anomaly_detection_backend_files(
    title: str,
    requirement: str = "",
    prd: str = "",
) -> Dict[str, str]:
    """Return backend file dict for anomaly detection app."""
    return {
        "backend/main.py": _backend_main_py(),
        "backend/anomaly_engine.py": _anomaly_engine_py(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Frontend helpers
# ──────────────────────────────────────────────────────────────────────────────

def _app_tsx() -> str:
    return """\
import { Navigate, Route, Routes } from 'react-router-dom';
import AppLayout from './layout/AppLayout';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import AnomalyFeedPage from './pages/AnomalyFeedPage';
import AlertConfigPage from './pages/AlertConfigPage';
import ReportsPage from './pages/ReportsPage';
import { isLoggedIn } from './auth';

function PrivateRoute({ children }: { children: JSX.Element }) {
  return isLoggedIn() ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<AppLayout />}>
        <Route path="/" element={<PrivateRoute><DashboardPage /></PrivateRoute>} />
        <Route path="/anomalies" element={<PrivateRoute><AnomalyFeedPage /></PrivateRoute>} />
        <Route path="/alerts" element={<PrivateRoute><AlertConfigPage /></PrivateRoute>} />
        <Route path="/reports" element={<PrivateRoute><ReportsPage /></PrivateRoute>} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
"""


def _app_layout_tsx(primary: str, primary_dark: str, bg: str, surface: str, text_color: str, muted: str, safe_title: str) -> str:
    code = """\
import { AppBar, Box, Chip, Drawer, List, ListItemButton, ListItemIcon, ListItemText, Toolbar, Typography } from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import BugReportIcon from '@mui/icons-material/BugReport';
import NotificationsActiveIcon from '@mui/icons-material/NotificationsActive';
import AssessmentIcon from '@mui/icons-material/Assessment';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import { clearAuth, getUser } from '../auth';

const DRAWER_W = 240;
const NAV = [
  { path: '/', label: 'Dashboard', icon: <DashboardIcon /> },
  { path: '/anomalies', label: 'Anomaly Feed', icon: <BugReportIcon /> },
  { path: '/alerts', label: 'Alert Config', icon: <NotificationsActiveIcon /> },
  { path: '/reports', label: 'Reports', icon: <AssessmentIcon /> },
];

export default function AppLayout() {
  const location = useLocation();
  const navigate = useNavigate();
  const user = getUser();
  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'COLOR_BG' }}>
      <Drawer
        variant="permanent"
        sx={{
          width: DRAWER_W, flexShrink: 0,
          '& .MuiDrawer-paper': { width: DRAWER_W, boxSizing: 'border-box', bgcolor: 'COLOR_PRIMARY', color: '#fff' },
        }}
      >
        <Toolbar sx={{ px: 2, py: 1.5 }}>
          <Typography variant="h6" fontWeight={800} sx={{ color: '#fff', fontSize: '0.88rem', lineHeight: 1.3 }}>
            SAFE_TITLE
          </Typography>
        </Toolbar>
        <List sx={{ px: 1 }}>
          {NAV.map((item) => (
            <ListItemButton
              key={item.path}
              component={Link}
              to={item.path}
              selected={location.pathname === item.path}
              sx={{
                borderRadius: 2, mb: 0.5,
                '&.Mui-selected': { bgcolor: 'rgba(255,255,255,0.18)' },
                '&:hover': { bgcolor: 'rgba(255,255,255,0.1)' },
              }}
            >
              <ListItemIcon sx={{ color: '#fff', minWidth: 36 }}>{item.icon}</ListItemIcon>
              <ListItemText
                primary={item.label}
                primaryTypographyProps={{
                  color: '#fff',
                  fontWeight: location.pathname === item.path ? 700 : 400,
                  fontSize: '0.9rem',
                }}
              />
            </ListItemButton>
          ))}
        </List>
      </Drawer>
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <AppBar
          position="sticky"
          elevation={0}
          sx={{ bgcolor: 'COLOR_SURFACE', color: 'COLOR_TEXT', borderBottom: '1px solid #e0e0e0', zIndex: 1 }}
        >
          <Toolbar sx={{ justifyContent: 'flex-end', gap: 2 }}>
            <Typography variant="body2" sx={{ color: 'COLOR_MUTED' }}>{user?.name || user?.email || 'User'}</Typography>
            <Chip
              label="Logout"
              size="small"
              onClick={() => { clearAuth(); navigate('/login'); }}
              sx={{ cursor: 'pointer' }}
            />
          </Toolbar>
        </AppBar>
        <Box component="main" sx={{ flex: 1, p: 3 }}>
          <Outlet />
        </Box>
      </Box>
    </Box>
  );
}
"""
    return (code
            .replace("COLOR_PRIMARY", primary)
            .replace("COLOR_BG", bg)
            .replace("COLOR_SURFACE", surface)
            .replace("COLOR_TEXT", text_color)
            .replace("COLOR_MUTED", muted)
            .replace("SAFE_TITLE", safe_title))


def _login_tsx(primary: str, primary_dark: str, bg: str, surface: str) -> str:
    gradient = f"linear-gradient(135deg, {primary} 0%, {primary_dark} 100%)"
    code = """\
import { useState } from 'react';
import { Alert, Box, Button, Card, CardContent, TextField, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { apiFetch } from '../api/client';
import { setAuth } from '../auth';

export default function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState('admin@example.com');
  const [password, setPassword] = useState('admin123');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const res = await apiFetch<{ access_token: string; user: any }>('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email, password }),
      });
      setAuth(res.access_token, res.user || { id: 1, name: email, email });
      navigate('/');
    } catch (err: any) {
      setError(err?.message || 'Invalid credentials');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: 'GRADIENT',
        display: 'flex', alignItems: 'center', justifyContent: 'center', p: 2,
      }}
    >
      <Card sx={{ maxWidth: 420, width: '100%', borderRadius: 3, boxShadow: 8 }}>
        <CardContent sx={{ p: 4 }}>
          <Typography variant="h5" fontWeight={800} sx={{ mb: 0.5, color: 'COLOR_PRIMARY' }}>
            Anomaly Detection Platform
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Sign in to monitor your metrics
          </Typography>
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          <Box component="form" onSubmit={submit} sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
              label="Email" type="email" value={email}
              onChange={(e) => setEmail(e.target.value)} required fullWidth size="small"
            />
            <TextField
              label="Password" type="password" value={password}
              onChange={(e) => setPassword(e.target.value)} required fullWidth size="small"
            />
            <Button
              type="submit" variant="contained" size="large"
              disabled={loading} fullWidth
              sx={{ bgcolor: 'COLOR_PRIMARY', '&:hover': { bgcolor: 'COLOR_PRIMARY_DARK' }, mt: 1 }}
            >
              {loading ? 'Signing in…' : 'Sign In'}
            </Button>
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block', textAlign: 'center' }}>
            Demo: admin@example.com / admin123
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
}
"""
    return (code
            .replace("GRADIENT", gradient)
            .replace("COLOR_PRIMARY_DARK", primary_dark)
            .replace("COLOR_PRIMARY", primary))


def _dashboard_tsx(primary: str, primary_dark: str, bg: str, surface: str, text_color: str, muted: str, safe_title: str) -> str:
    code = """\
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Box, Card, CardContent, Typography, Select, MenuItem, FormControl,
  InputLabel, Chip, Stack, CircularProgress, Grid2,
} from '@mui/material';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import TimelineIcon from '@mui/icons-material/Timeline';
import NotificationsActiveIcon from '@mui/icons-material/NotificationsActive';
import SpeedIcon from '@mui/icons-material/Speed';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine, ResponsiveContainer,
} from 'recharts';
import { apiFetch } from '../api/client';

interface MetricPoint {
  timestamp: string;
  value: number;
  is_anomaly: boolean;
  anomaly_score: number;
}

interface MetricData {
  metric_id: string;
  metric_name: string;
  unit: string;
  threshold_high: number;
  threshold_low: number;
  data: MetricPoint[];
}

interface DashboardStats {
  total_anomalies_today: number;
  active_alerts: number;
  metrics_monitored: number;
  avg_anomaly_score: number;
  metrics: MetricData[];
}

const AnomalyDot = (props: any) => {
  const { cx, cy, payload } = props;
  if (!payload?.is_anomaly) return null;
  return <circle cx={cx} cy={cy} r={7} fill="#C62828" stroke="#fff" strokeWidth={2} />;
};

export default function DashboardPage() {
  const [selectedMetricId, setSelectedMetricId] = useState<string>('');

  const { data, isLoading } = useQuery<DashboardStats>({
    queryKey: ['dashboard'],
    queryFn: () => apiFetch<DashboardStats>('/api/dashboard'),
    refetchInterval: 30_000,
  });

  const metrics = data?.metrics ?? [];
  const activeMetric = metrics.find((m) => m.metric_id === selectedMetricId) ?? metrics[0];

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
        <CircularProgress sx={{ color: 'COLOR_PRIMARY' }} />
      </Box>
    );
  }

  const statCards = [
    {
      label: 'Anomalies Today',
      value: data?.total_anomalies_today ?? 0,
      icon: <WarningAmberIcon />,
      color: '#C62828',
      bg: '#FFEBEE',
    },
    {
      label: 'Active Alerts',
      value: data?.active_alerts ?? 0,
      icon: <NotificationsActiveIcon />,
      color: '#E65100',
      bg: '#FFF3E0',
    },
    {
      label: 'Metrics Monitored',
      value: data?.metrics_monitored ?? 0,
      icon: <TimelineIcon />,
      color: 'COLOR_PRIMARY',
      bg: '#E3F2FD',
    },
    {
      label: 'Avg Anomaly Score',
      value: (data?.avg_anomaly_score ?? 0).toFixed(2),
      icon: <SpeedIcon />,
      color: '#2E7D32',
      bg: '#E8F5E9',
    },
  ];

  return (
    <Box sx={{ p: 3, bgcolor: 'COLOR_BG', minHeight: '100vh' }}>
      <Typography variant="h4" fontWeight={800} sx={{ mb: 3, color: 'COLOR_PRIMARY' }}>
        Anomaly Detection Dashboard
      </Typography>

      {/* KPI Summary Cards */}
      <Grid2 container spacing={3} sx={{ mb: 4 }}>
        {statCards.map((card) => (
          <Grid2 key={card.label} size={{ xs: 12, sm: 6, md: 3 }}>
            <Card sx={{ borderRadius: 2, boxShadow: 2, bgcolor: 'COLOR_SURFACE' }}>
              <CardContent>
                <Stack direction="row" alignItems="center" spacing={2}>
                  <Box
                    sx={{
                      width: 48, height: 48, borderRadius: 2,
                      bgcolor: card.bg, display: 'flex',
                      alignItems: 'center', justifyContent: 'center',
                      color: card.color, flexShrink: 0,
                    }}
                  >
                    {card.icon}
                  </Box>
                  <Box>
                    <Typography variant="h4" fontWeight={800} sx={{ color: card.color, lineHeight: 1 }}>
                      {card.value}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {card.label}
                    </Typography>
                  </Box>
                </Stack>
              </CardContent>
            </Card>
          </Grid2>
        ))}
      </Grid2>

      {/* Time-Series Chart with Anomaly Markers */}
      <Card sx={{ borderRadius: 2, boxShadow: 2, bgcolor: 'COLOR_SURFACE' }}>
        <CardContent>
          <Stack
            direction="row" alignItems="center" justifyContent="space-between"
            flexWrap="wrap" gap={2} sx={{ mb: 2 }}
          >
            <Typography variant="h6" fontWeight={700}>
              Metric Time Series — Last 24 Hours
            </Typography>
            <FormControl size="small" sx={{ minWidth: 220 }}>
              <InputLabel>Select Metric</InputLabel>
              <Select
                value={selectedMetricId || metrics[0]?.metric_id || ''}
                label="Select Metric"
                onChange={(e) => setSelectedMetricId(e.target.value)}
              >
                {metrics.map((m) => (
                  <MenuItem key={m.metric_id} value={m.metric_id}>{m.metric_name}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Stack>

          {activeMetric ? (
            <>
              <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mb: 2 }}>
                <Chip
                  size="small"
                  label={`${activeMetric.data.filter((d) => d.is_anomaly).length} anomalies`}
                  sx={{ bgcolor: '#FFEBEE', color: '#C62828', fontWeight: 600 }}
                />
                <Chip size="small" label={`Unit: ${activeMetric.unit}`} variant="outlined" />
                <Chip
                  size="small"
                  label={`High threshold: ${activeMetric.threshold_high}`}
                  sx={{ bgcolor: '#FFF8E1', color: '#E65100' }}
                />
              </Stack>

              <ResponsiveContainer width="100%" height={380}>
                <LineChart data={activeMetric.data} margin={{ top: 12, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(v: string) => {
                      const d = new Date(v);
                      return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
                    }}
                    interval={23}
                    tick={{ fontSize: 11 }}
                  />
                  <YAxis domain={['auto', 'auto']} tick={{ fontSize: 11 }} />
                  <Tooltip
                    labelFormatter={(v: string) => new Date(v).toLocaleString()}
                    formatter={(val: number, name: string) => [
                      `${Number(val).toFixed(2)} ${activeMetric.unit}`,
                      name,
                    ]}
                  />
                  <Legend />
                  <ReferenceLine
                    y={activeMetric.threshold_high}
                    stroke="#C62828"
                    strokeDasharray="6 3"
                    label={{
                      value: `Max (${activeMetric.threshold_high})`,
                      position: 'insideTopRight',
                      fill: '#C62828',
                      fontSize: 11,
                    }}
                  />
                  {activeMetric.threshold_low > 0 && (
                    <ReferenceLine
                      y={activeMetric.threshold_low}
                      stroke="#E65100"
                      strokeDasharray="6 3"
                      label={{
                        value: `Min (${activeMetric.threshold_low})`,
                        position: 'insideBottomRight',
                        fill: '#E65100',
                        fontSize: 11,
                      }}
                    />
                  )}
                  <Line
                    type="monotone"
                    dataKey="value"
                    name={activeMetric.metric_name}
                    stroke="COLOR_PRIMARY"
                    strokeWidth={2}
                    dot={<AnomalyDot />}
                    activeDot={{ r: 6, fill: 'COLOR_PRIMARY' }}
                    connectNulls
                  />
                </LineChart>
              </ResponsiveContainer>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                🔴 Red dots indicate detected anomalies (z-score exceeds detection threshold)
              </Typography>
            </>
          ) : (
            <Box sx={{ py: 6, textAlign: 'center' }}>
              <Typography color="text.secondary">No metric data available.</Typography>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}
"""
    return (code
            .replace("COLOR_PRIMARY", primary)
            .replace("COLOR_BG", bg)
            .replace("COLOR_SURFACE", surface))


def _anomaly_feed_tsx(primary: str, primary_dark: str, bg: str, surface: str, text_color: str, muted: str) -> str:
    code = """\
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Box, Card, CardContent, Typography, TextField, Select, MenuItem,
  FormControl, InputLabel, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Chip, Stack, CircularProgress, InputAdornment,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import { apiFetch } from '../api/client';

interface Anomaly {
  id: string;
  timestamp: string;
  metric_name: string;
  metric_id: string;
  value: number;
  expected_min: number;
  expected_max: number;
  severity: 'Low' | 'Medium' | 'High' | 'Critical';
  status: 'Open' | 'Acknowledged' | 'Resolved';
  anomaly_score: number;
}

const SEVERITY_STYLE: Record<string, { bg: string; color: string }> = {
  Low:      { bg: '#E8F5E9', color: '#2E7D32' },
  Medium:   { bg: '#FFF8E1', color: '#F57F17' },
  High:     { bg: '#FFF3E0', color: '#E65100' },
  Critical: { bg: '#FFEBEE', color: '#B71C1C' },
};

const STATUS_COLOR: Record<string, 'default' | 'warning' | 'success'> = {
  Open:         'warning',
  Acknowledged: 'default',
  Resolved:     'success',
};

export default function AnomalyFeedPage() {
  const [search, setSearch] = useState('');
  const [severityFilter, setSeverityFilter] = useState('All');

  const { data, isLoading } = useQuery({
    queryKey: ['anomalies'],
    queryFn: () => apiFetch<{ items: Anomaly[]; total: number }>('/api/anomalies'),
    refetchInterval: 15_000,
  });

  const anomalies = data?.items ?? [];
  const filtered = anomalies
    .filter((a) => severityFilter === 'All' || a.severity === severityFilter)
    .filter((a) => !search || a.metric_name.toLowerCase().includes(search.toLowerCase()));

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
        <CircularProgress sx={{ color: 'COLOR_PRIMARY' }} />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3, bgcolor: 'COLOR_BG', minHeight: '100vh' }}>
      <Typography variant="h4" fontWeight={800} sx={{ mb: 3, color: 'COLOR_PRIMARY' }}>
        Anomaly Feed
      </Typography>

      {/* Filters Bar */}
      <Card sx={{ borderRadius: 2, boxShadow: 1, bgcolor: 'COLOR_SURFACE', mb: 3 }}>
        <CardContent sx={{ py: 2 }}>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems="center">
            <TextField
              size="small"
              placeholder="Search by metric name…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              sx={{ flex: 1, maxWidth: 360 }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon fontSize="small" />
                  </InputAdornment>
                ),
              }}
            />
            <FormControl size="small" sx={{ minWidth: 160 }}>
              <InputLabel>Severity</InputLabel>
              <Select
                value={severityFilter}
                label="Severity"
                onChange={(e) => setSeverityFilter(e.target.value)}
              >
                <MenuItem value="All">All</MenuItem>
                <MenuItem value="Critical">Critical</MenuItem>
                <MenuItem value="High">High</MenuItem>
                <MenuItem value="Medium">Medium</MenuItem>
                <MenuItem value="Low">Low</MenuItem>
              </Select>
            </FormControl>
            <Typography variant="body2" color="text.secondary">
              {filtered.length} of {anomalies.length} anomalies
            </Typography>
          </Stack>
        </CardContent>
      </Card>

      {/* Anomaly Table */}
      <Card sx={{ borderRadius: 2, boxShadow: 2, bgcolor: 'COLOR_SURFACE' }}>
        <TableContainer>
          <Table size="small">
            <TableHead sx={{ bgcolor: 'COLOR_PRIMARY' }}>
              <TableRow>
                {['Timestamp', 'Metric', 'Value', 'Expected Range', 'Score', 'Severity', 'Status'].map((h) => (
                  <TableCell key={h} sx={{ color: '#fff', fontWeight: 700 }}>{h}</TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {filtered.map((row) => {
                const sev = SEVERITY_STYLE[row.severity] ?? { bg: '#f5f5f5', color: '#424242' };
                return (
                  <TableRow key={row.id} hover sx={{ '&:last-child td': { border: 0 } }}>
                    <TableCell sx={{ fontSize: '0.78rem', whiteSpace: 'nowrap' }}>
                      {new Date(row.timestamp).toLocaleString()}
                    </TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>{row.metric_name}</TableCell>
                    <TableCell sx={{ fontFamily: 'monospace', fontWeight: 700 }}>
                      {Number(row.value).toFixed(2)}
                    </TableCell>
                    <TableCell sx={{ fontSize: '0.78rem', color: 'COLOR_MUTED' }}>
                      {Number(row.expected_min).toFixed(1)} – {Number(row.expected_max).toFixed(1)}
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {Number(row.anomaly_score).toFixed(2)}
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={row.severity}
                        size="small"
                        sx={{ bgcolor: sev.bg, color: sev.color, fontWeight: 700, fontSize: '0.72rem' }}
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={row.status}
                        size="small"
                        color={STATUS_COLOR[row.status] ?? 'default'}
                      />
                    </TableCell>
                  </TableRow>
                );
              })}
              {filtered.length === 0 && (
                <TableRow>
                  <TableCell colSpan={7} align="center" sx={{ py: 4 }}>
                    <Typography color="text.secondary">
                      No anomalies match the current filters.
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Card>
    </Box>
  );
}
"""
    return (code
            .replace("COLOR_PRIMARY", primary)
            .replace("COLOR_BG", bg)
            .replace("COLOR_SURFACE", surface)
            .replace("COLOR_MUTED", muted))


def _alert_config_tsx(primary: str, primary_dark: str, bg: str, surface: str, text_color: str, muted: str) -> str:
    code = """\
import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Box, Card, CardContent, Typography, Button, Switch, FormControlLabel,
  Dialog, DialogTitle, DialogContent, DialogActions, TextField, Stack,
  CircularProgress, Grid2, Divider, Chip, IconButton,
} from '@mui/material';
import EditIcon from '@mui/icons-material/Edit';
import { apiFetch } from '../api/client';

interface MetricConfig {
  metric_id: string;
  metric_name: string;
  unit: string;
  threshold_min: number;
  threshold_max: number;
  zscore_threshold: number;
  enabled: boolean;
  last_triggered?: string;
}

interface ThresholdDraft {
  threshold_min: number;
  threshold_max: number;
  zscore_threshold: number;
}

export default function AlertConfigPage() {
  const qc = useQueryClient();
  const [editing, setEditing] = useState<MetricConfig | null>(null);
  const [draft, setDraft] = useState<ThresholdDraft>({
    threshold_min: 0,
    threshold_max: 100,
    zscore_threshold: 3.0,
  });

  const { data, isLoading } = useQuery({
    queryKey: ['thresholds'],
    queryFn: () => apiFetch<{ items: MetricConfig[] }>('/api/thresholds'),
  });

  const updateMutation = useMutation({
    mutationFn: (payload: Partial<MetricConfig> & { metric_id: string }) =>
      apiFetch(`/api/thresholds/${payload.metric_id}`, {
        method: 'PUT',
        body: JSON.stringify(payload),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['thresholds'] }),
  });

  const metrics = data?.items ?? [];

  const handleToggle = (m: MetricConfig) => {
    updateMutation.mutate({ ...m, enabled: !m.enabled });
  };

  const handleEdit = (m: MetricConfig) => {
    setDraft({
      threshold_min: m.threshold_min,
      threshold_max: m.threshold_max,
      zscore_threshold: m.zscore_threshold,
    });
    setEditing(m);
  };

  const handleSave = () => {
    if (!editing) return;
    updateMutation.mutate({ ...editing, ...draft });
    setEditing(null);
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
        <CircularProgress sx={{ color: 'COLOR_PRIMARY' }} />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3, bgcolor: 'COLOR_BG', minHeight: '100vh' }}>
      <Typography variant="h4" fontWeight={800} sx={{ mb: 1, color: 'COLOR_PRIMARY' }}>
        Alert Configuration
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Configure detection thresholds and z-score sensitivity for each monitored metric.
      </Typography>

      <Grid2 container spacing={3}>
        {metrics.map((m) => (
          <Grid2 key={m.metric_id} size={{ xs: 12, sm: 6, md: 4 }}>
            <Card
              sx={{
                borderRadius: 2, boxShadow: 2, bgcolor: 'COLOR_SURFACE',
                opacity: m.enabled ? 1 : 0.65,
                transition: 'opacity 0.2s',
              }}
            >
              <CardContent>
                <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
                  <Typography variant="subtitle1" fontWeight={700}>{m.metric_name}</Typography>
                  <Stack direction="row" alignItems="center" spacing={0.5}>
                    <Chip label={m.unit} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} />
                    <IconButton size="small" onClick={() => handleEdit(m)} sx={{ color: 'COLOR_PRIMARY' }}>
                      <EditIcon fontSize="small" />
                    </IconButton>
                  </Stack>
                </Stack>
                <Divider sx={{ mb: 1.5 }} />
                <Stack spacing={0.75}>
                  <Stack direction="row" justifyContent="space-between">
                    <Typography variant="caption" color="text.secondary">Min Threshold</Typography>
                    <Typography variant="caption" fontWeight={600} fontFamily="monospace">
                      {m.threshold_min} {m.unit}
                    </Typography>
                  </Stack>
                  <Stack direction="row" justifyContent="space-between">
                    <Typography variant="caption" color="text.secondary">Max Threshold</Typography>
                    <Typography variant="caption" fontWeight={600} fontFamily="monospace" sx={{ color: '#C62828' }}>
                      {m.threshold_max} {m.unit}
                    </Typography>
                  </Stack>
                  <Stack direction="row" justifyContent="space-between">
                    <Typography variant="caption" color="text.secondary">Z-Score Threshold</Typography>
                    <Typography variant="caption" fontWeight={600} fontFamily="monospace">
                      {m.zscore_threshold}
                    </Typography>
                  </Stack>
                  {m.last_triggered && (
                    <Stack direction="row" justifyContent="space-between">
                      <Typography variant="caption" color="text.secondary">Last Triggered</Typography>
                      <Typography variant="caption" sx={{ color: '#E65100' }}>
                        {new Date(m.last_triggered).toLocaleDateString()}
                      </Typography>
                    </Stack>
                  )}
                </Stack>
                <Divider sx={{ mt: 1.5, mb: 1 }} />
                <FormControlLabel
                  control={
                    <Switch
                      checked={m.enabled}
                      onChange={() => handleToggle(m)}
                      size="small"
                      sx={{
                        '& .MuiSwitch-switchBase.Mui-checked': { color: 'COLOR_PRIMARY' },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': { bgcolor: 'COLOR_PRIMARY' },
                      }}
                    />
                  }
                  label={
                    <Typography variant="caption">
                      {m.enabled ? 'Monitoring Enabled' : 'Disabled'}
                    </Typography>
                  }
                />
              </CardContent>
            </Card>
          </Grid2>
        ))}
      </Grid2>

      {/* Edit Threshold Dialog */}
      <Dialog open={Boolean(editing)} onClose={() => setEditing(null)} maxWidth="xs" fullWidth>
        <DialogTitle fontWeight={700}>
          Edit Threshold — {editing?.metric_name}
        </DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField
              label={`Min Threshold (${editing?.unit ?? ''})`}
              type="number"
              value={draft.threshold_min}
              onChange={(e) => setDraft({ ...draft, threshold_min: Number(e.target.value) })}
              fullWidth size="small"
            />
            <TextField
              label={`Max Threshold (${editing?.unit ?? ''})`}
              type="number"
              value={draft.threshold_max}
              onChange={(e) => setDraft({ ...draft, threshold_max: Number(e.target.value) })}
              fullWidth size="small"
            />
            <TextField
              label="Z-Score Threshold"
              type="number"
              inputProps={{ step: 0.1, min: 1, max: 10 }}
              value={draft.zscore_threshold}
              onChange={(e) => setDraft({ ...draft, zscore_threshold: Number(e.target.value) })}
              fullWidth size="small"
              helperText="Lower value = more sensitive. Typical range: 2.5 – 4.0"
            />
          </Stack>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setEditing(null)} color="inherit">Cancel</Button>
          <Button
            onClick={handleSave}
            variant="contained"
            sx={{ bgcolor: 'COLOR_PRIMARY', '&:hover': { bgcolor: 'COLOR_PRIMARY_DARK' } }}
          >
            Save Changes
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
"""
    return (code
            .replace("COLOR_PRIMARY_DARK", primary_dark)
            .replace("COLOR_PRIMARY", primary)
            .replace("COLOR_BG", bg)
            .replace("COLOR_SURFACE", surface))


def _reports_tsx(primary: str, primary_dark: str, secondary: str, bg: str, surface: str, text_color: str, muted: str) -> str:
    code = """\
import { useQuery } from '@tanstack/react-query';
import {
  Box, Card, CardContent, Typography, Stack, CircularProgress, Grid2,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip,
} from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, PieChart, Pie, Cell,
} from 'recharts';
import { apiFetch } from '../api/client';

interface DailyCount { date: string; count: number; }
interface SeverityCount { name: string; value: number; }
interface TopMetric { metric_name: string; count: number; severity: string; }

interface ReportsSummary {
  total_anomalies: number;
  anomaly_rate: number;
  mttd_minutes: number;
  daily_counts: DailyCount[];
  by_severity: SeverityCount[];
  top_metrics: TopMetric[];
}

const SEV_COLORS: Record<string, string> = {
  Low: '#81C784',
  Medium: '#FFB74D',
  High: '#FF8A65',
  Critical: '#E57373',
};
const PIE_PALETTE = ['#81C784', '#FFB74D', '#FF8A65', '#E57373'];

const RADIAN = Math.PI / 180;
const PieLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }: any) => {
  const r = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + r * Math.cos(-midAngle * RADIAN);
  const y = cy + r * Math.sin(-midAngle * RADIAN);
  if (percent <= 0.05) return null;
  return (
    <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" fontSize={12} fontWeight={700}>
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
};

export default function ReportsPage() {
  const { data, isLoading } = useQuery<ReportsSummary>({
    queryKey: ['reports/summary'],
    queryFn: () => apiFetch<ReportsSummary>('/api/reports/summary'),
  });

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
        <CircularProgress sx={{ color: 'COLOR_PRIMARY' }} />
      </Box>
    );
  }

  const summaryCards = [
    { label: 'Total Anomalies (30d)', value: data?.total_anomalies ?? 0, color: '#C62828' },
    { label: 'Anomaly Rate', value: `${((data?.anomaly_rate ?? 0) * 100).toFixed(3)}%`, color: 'COLOR_PRIMARY' },
    { label: 'Avg MTTD', value: `${(data?.mttd_minutes ?? 0).toFixed(1)} min`, color: '#2E7D32' },
  ];

  return (
    <Box sx={{ p: 3, bgcolor: 'COLOR_BG', minHeight: '100vh' }}>
      <Typography variant="h4" fontWeight={800} sx={{ mb: 3, color: 'COLOR_PRIMARY' }}>
        Anomaly Reports
      </Typography>

      {/* Summary KPIs */}
      <Grid2 container spacing={3} sx={{ mb: 4 }}>
        {summaryCards.map((c) => (
          <Grid2 key={c.label} size={{ xs: 12, sm: 4 }}>
            <Card sx={{ borderRadius: 2, boxShadow: 2, bgcolor: 'COLOR_SURFACE' }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h3" fontWeight={900} sx={{ color: c.color }}>
                  {c.value}
                </Typography>
                <Typography variant="body2" color="text.secondary">{c.label}</Typography>
              </CardContent>
            </Card>
          </Grid2>
        ))}
      </Grid2>

      <Grid2 container spacing={3} sx={{ mb: 4 }}>
        {/* Bar Chart — Anomalies per Day */}
        <Grid2 size={{ xs: 12, md: 8 }}>
          <Card sx={{ borderRadius: 2, boxShadow: 2, bgcolor: 'COLOR_SURFACE' }}>
            <CardContent>
              <Typography variant="h6" fontWeight={700} sx={{ mb: 2 }}>
                Anomalies per Day — Last 30 Days
              </Typography>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart
                  data={data?.daily_counts ?? []}
                  margin={{ top: 5, right: 20, left: 0, bottom: 35 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f5f5f5" />
                  <XAxis dataKey="date" angle={-40} textAnchor="end" tick={{ fontSize: 10 }} interval={4} />
                  <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" name="Anomalies" fill="COLOR_PRIMARY" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid2>

        {/* Pie Chart — By Severity */}
        <Grid2 size={{ xs: 12, md: 4 }}>
          <Card sx={{ borderRadius: 2, boxShadow: 2, bgcolor: 'COLOR_SURFACE' }}>
            <CardContent>
              <Typography variant="h6" fontWeight={700} sx={{ mb: 2 }}>
                By Severity
              </Typography>
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={data?.by_severity ?? []}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    labelLine={false}
                    label={PieLabel}
                  >
                    {(data?.by_severity ?? []).map((_: any, idx: number) => (
                      <Cell key={`cell-${idx}`} fill={PIE_PALETTE[idx % PIE_PALETTE.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid2>
      </Grid2>

      {/* Top Anomalous Metrics Table */}
      <Card sx={{ borderRadius: 2, boxShadow: 2, bgcolor: 'COLOR_SURFACE' }}>
        <CardContent>
          <Typography variant="h6" fontWeight={700} sx={{ mb: 2 }}>
            Top Anomalous Metrics
          </Typography>
          <TableContainer>
            <Table size="small">
              <TableHead sx={{ bgcolor: 'COLOR_PRIMARY' }}>
                <TableRow>
                  {['Rank', 'Metric Name', 'Anomaly Count', 'Worst Severity'].map((h) => (
                    <TableCell key={h} sx={{ color: '#fff', fontWeight: 700 }}>{h}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {(data?.top_metrics ?? []).map((row, i) => {
                  const sevColor = SEV_COLORS[row.severity] ?? '#e0e0e0';
                  return (
                    <TableRow key={row.metric_name} hover>
                      <TableCell sx={{ fontWeight: 700, color: 'COLOR_MUTED' }}>#{i + 1}</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>{row.metric_name}</TableCell>
                      <TableCell sx={{ fontFamily: 'monospace', fontWeight: 700 }}>{row.count}</TableCell>
                      <TableCell>
                        <Chip
                          label={row.severity}
                          size="small"
                          sx={{ bgcolor: `${sevColor}30`, color: sevColor, fontWeight: 700, fontSize: '0.72rem' }}
                        />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
}
"""
    return (code
            .replace("COLOR_PRIMARY", primary)
            .replace("COLOR_BG", bg)
            .replace("COLOR_SURFACE", surface)
            .replace("COLOR_MUTED", muted))


def _mock_ts(safe_title: str) -> str:
    return """\
// Auto-generated mock API — Anomaly Detection Platform
// All API calls fall back here when the backend is unreachable.

const USERS = [
  { id: 1, email: 'admin@example.com', password: 'admin123', name: 'Admin User', role: 'ADMIN' },
  { id: 2, email: 'analyst@example.com', password: 'analyst123', name: 'Data Analyst', role: 'ANALYST' },
];

const METRICS_CONFIG = [
  { metric_id: 'cpu_usage', metric_name: 'CPU Usage', unit: '%', threshold_min: 0, threshold_max: 85, zscore_threshold: 3.0, enabled: true, last_triggered: '2026-10-04T14:22:00Z' },
  { metric_id: 'memory_usage', metric_name: 'Memory Usage', unit: '%', threshold_min: 0, threshold_max: 90, zscore_threshold: 3.0, enabled: true, last_triggered: '2026-10-05T08:14:00Z' },
  { metric_id: 'request_latency', metric_name: 'Request Latency', unit: 'ms', threshold_min: 0, threshold_max: 2000, zscore_threshold: 2.5, enabled: true, last_triggered: '2026-10-05T10:47:00Z' },
  { metric_id: 'error_rate', metric_name: 'Error Rate', unit: '%', threshold_min: 0, threshold_max: 5, zscore_threshold: 2.5, enabled: true, last_triggered: '2026-10-03T22:11:00Z' },
  { metric_id: 'disk_io', metric_name: 'Disk I/O', unit: 'MB/s', threshold_min: 0, threshold_max: 500, zscore_threshold: 3.0, enabled: true },
  { metric_id: 'network_in', metric_name: 'Network In', unit: 'Mbps', threshold_min: 0, threshold_max: 1000, zscore_threshold: 3.0, enabled: false },
];

let _metricsConfig = METRICS_CONFIG.map((m) => ({ ...m }));

// Generate deterministic time-series data for a metric
function genTimeSeries(metricId: string, hours = 24): any[] {
  const now = Date.now();
  const interval = 5 * 60 * 1000;
  const n = Math.floor((hours * 60) / 5);
  const cfgs: Record<string, [number, number, number]> = {
    cpu_usage:       [45, 10, 93],
    memory_usage:    [65,  8, 94],
    request_latency: [250, 50, 3400],
    error_rate:      [0.5, 0.3, 15],
    disk_io:         [120, 30, 640],
    network_in:      [350, 80, 1350],
  };
  const [base, std, spike] = cfgs[metricId] ?? [50, 10, 100];
  // Use a simple pseudo-random seeded sequence for reproducibility
  let seed = metricId.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  const rand = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
  const anomalySet = new Set<number>();
  for (let k = 0; k < 4; k++) anomalySet.add(Math.floor(rand() * (n - 20)) + 10);
  return Array.from({ length: n }, (_, i) => {
    const ts = new Date(now - (n - i) * interval).toISOString();
    const isAnomaly = anomalySet.has(i);
    const val = isAnomaly
      ? spike * (0.9 + rand() * 0.25)
      : Math.max(0, base + (rand() - 0.5) * 2.5 * std);
    return {
      timestamp: ts,
      value: Math.round(val * 100) / 100,
      is_anomaly: isAnomaly,
      anomaly_score: isAnomaly ? 3.5 + rand() * 2.5 : rand() * 0.7,
    };
  });
}

const ANOMALIES = [
  { id: 'a1',  timestamp: '2026-10-05T18:22:00Z', metric_name: 'CPU Usage',        metric_id: 'cpu_usage',        value: 97.3,   expected_min: 20,   expected_max: 85,   severity: 'Critical', status: 'Open',         anomaly_score: 5.12 },
  { id: 'a2',  timestamp: '2026-10-05T15:47:00Z', metric_name: 'Request Latency',  metric_id: 'request_latency',  value: 4280,   expected_min: 100,  expected_max: 2000, severity: 'High',     status: 'Acknowledged', anomaly_score: 4.03 },
  { id: 'a3',  timestamp: '2026-10-05T12:14:00Z', metric_name: 'Error Rate',       metric_id: 'error_rate',       value: 18.5,   expected_min: 0,    expected_max: 5,    severity: 'Critical', status: 'Open',         anomaly_score: 6.77 },
  { id: 'a4',  timestamp: '2026-10-05T09:33:00Z', metric_name: 'Memory Usage',     metric_id: 'memory_usage',     value: 96.1,   expected_min: 40,   expected_max: 90,   severity: 'High',     status: 'Open',         anomaly_score: 3.98 },
  { id: 'a5',  timestamp: '2026-10-04T23:11:00Z', metric_name: 'Disk I/O',         metric_id: 'disk_io',          value: 712,    expected_min: 0,    expected_max: 500,  severity: 'Medium',   status: 'Resolved',     anomaly_score: 3.14 },
  { id: 'a6',  timestamp: '2026-10-04T19:05:00Z', metric_name: 'CPU Usage',        metric_id: 'cpu_usage',        value: 91.8,   expected_min: 20,   expected_max: 85,   severity: 'High',     status: 'Resolved',     anomaly_score: 3.67 },
  { id: 'a7',  timestamp: '2026-10-04T14:44:00Z', metric_name: 'Request Latency',  metric_id: 'request_latency',  value: 3100,   expected_min: 100,  expected_max: 2000, severity: 'High',     status: 'Resolved',     anomaly_score: 3.51 },
  { id: 'a8',  timestamp: '2026-10-04T10:21:00Z', metric_name: 'Network In',       metric_id: 'network_in',       value: 1450,   expected_min: 0,    expected_max: 1000, severity: 'Medium',   status: 'Resolved',     anomaly_score: 2.88 },
  { id: 'a9',  timestamp: '2026-10-03T22:58:00Z', metric_name: 'Error Rate',       metric_id: 'error_rate',       value: 8.2,    expected_min: 0,    expected_max: 5,    severity: 'High',     status: 'Resolved',     anomaly_score: 3.45 },
  { id: 'a10', timestamp: '2026-10-03T16:37:00Z', metric_name: 'Memory Usage',     metric_id: 'memory_usage',     value: 92.4,   expected_min: 40,   expected_max: 90,   severity: 'Medium',   status: 'Resolved',     anomaly_score: 2.71 },
  { id: 'a11', timestamp: '2026-10-03T11:19:00Z', metric_name: 'CPU Usage',        metric_id: 'cpu_usage',        value: 88.7,   expected_min: 20,   expected_max: 85,   severity: 'Medium',   status: 'Resolved',     anomaly_score: 2.53 },
  { id: 'a12', timestamp: '2026-10-02T08:44:00Z', metric_name: 'Disk I/O',         metric_id: 'disk_io',          value: 620,    expected_min: 0,    expected_max: 500,  severity: 'Low',      status: 'Resolved',     anomaly_score: 2.21 },
];

const ALERTS = [
  { id: 'al1', metric_name: 'CPU Usage',       threshold: 85,   current_value: 97.3,  severity: 'Critical', triggered_at: '2026-10-05T18:22:00Z', status: 'Active' },
  { id: 'al2', metric_name: 'Error Rate',      threshold: 5,    current_value: 18.5,  severity: 'Critical', triggered_at: '2026-10-05T12:14:00Z', status: 'Active' },
  { id: 'al3', metric_name: 'Memory Usage',    threshold: 90,   current_value: 96.1,  severity: 'High',     triggered_at: '2026-10-05T09:33:00Z', status: 'Active' },
  { id: 'al4', metric_name: 'Request Latency', threshold: 2000, current_value: 4280,  severity: 'High',     triggered_at: '2026-10-05T15:47:00Z', status: 'Acknowledged' },
];

function genDailyCounts(): any[] {
  const now = new Date();
  let seed = 42;
  const rand = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
  return Array.from({ length: 30 }, (_, i) => {
    const d = new Date(now);
    d.setDate(d.getDate() - (29 - i));
    return {
      date: d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      count: Math.floor(rand() * 8) + 1,
    };
  });
}

const DAILY_COUNTS = genDailyCounts();

function ok<T>(data: T): Promise<T> { return Promise.resolve(data); }
function err(msg: string): Promise<never> { return Promise.reject(new Error(msg)); }

export async function mockFetch<T = unknown>(path: string, options: RequestInit = {}): Promise<T> {
  const method = (options.method || 'GET').toUpperCase();
  const clean = path.replace(/^[/]api[/]/, '').replace(/[?].*$/, '');
  const parts = clean.split('/').filter(Boolean);

  // ── Auth ──────────────────────────────────────────────────────────────────
  if (clean === 'auth/login' && method === 'POST') {
    const b = JSON.parse(String(options.body || '{}'));
    const u = USERS.find((u) => u.email === b.email && u.password === b.password);
    if (!u) return err('Invalid credentials') as any;
    return ok({ access_token: 'mock-jwt-' + u.id, user: { id: u.id, name: u.name, email: u.email, role: u.role } }) as T;
  }
  if (clean === 'auth/register' && method === 'POST') {
    const b = JSON.parse(String(options.body || '{}'));
    USERS.push({ id: USERS.length + 1, email: b.email, password: b.password, name: b.name || b.email, role: 'ANALYST' });
    return ok({ ok: true }) as T;
  }

  // ── Dashboard ─────────────────────────────────────────────────────────────
  if (clean === 'dashboard' && method === 'GET') {
    const enabled = _metricsConfig.filter((m) => m.enabled);
    const metricsWithData = enabled.map((m) => ({
      ...m,
      threshold_low: m.threshold_min,
      threshold_high: m.threshold_max,
      data: genTimeSeries(m.metric_id),
    }));
    const today = new Date().toISOString().slice(0, 10);
    const todayCount = ANOMALIES.filter((a) => a.timestamp.startsWith(today)).length || 3;
    const avgScore = ANOMALIES.reduce((s, a) => s + a.anomaly_score, 0) / ANOMALIES.length;
    return ok({
      total_anomalies_today: todayCount,
      active_alerts: ALERTS.filter((a) => a.status === 'Active').length,
      metrics_monitored: enabled.length,
      avg_anomaly_score: Math.round(avgScore * 100) / 100,
      metrics: metricsWithData,
    }) as T;
  }

  // ── Anomalies list ────────────────────────────────────────────────────────
  if (clean === 'anomalies' && method === 'GET') {
    return ok({ items: ANOMALIES, total: ANOMALIES.length }) as T;
  }

  // ── Alerts list ───────────────────────────────────────────────────────────
  if (clean === 'alerts' && method === 'GET') {
    return ok({ items: ALERTS, total: ALERTS.length }) as T;
  }

  // ── Thresholds list ───────────────────────────────────────────────────────
  if (clean === 'thresholds' && method === 'GET') {
    return ok({ items: _metricsConfig, total: _metricsConfig.length }) as T;
  }

  // ── Update threshold ──────────────────────────────────────────────────────
  if (parts[0] === 'thresholds' && parts.length === 2 && method === 'PUT') {
    const metricId = parts[1];
    const b = JSON.parse(String(options.body || '{}'));
    _metricsConfig = _metricsConfig.map((m) => (m.metric_id === metricId ? { ...m, ...b } : m));
    return ok(_metricsConfig.find((m) => m.metric_id === metricId)) as T;
  }

  // ── Reports summary ───────────────────────────────────────────────────────
  if (clean === 'reports/summary' && method === 'GET') {
    const bySeverity = ['Low', 'Medium', 'High', 'Critical']
      .map((sev) => ({ name: sev, value: ANOMALIES.filter((a) => a.severity === sev).length }))
      .filter((s) => s.value > 0);
    const topMetrics = [
      { metric_name: 'CPU Usage',        count: 4, severity: 'Critical' },
      { metric_name: 'Error Rate',       count: 3, severity: 'Critical' },
      { metric_name: 'Request Latency',  count: 3, severity: 'High' },
      { metric_name: 'Memory Usage',     count: 3, severity: 'High' },
      { metric_name: 'Disk I/O',         count: 2, severity: 'Medium' },
    ];
    return ok({
      total_anomalies: ANOMALIES.length,
      anomaly_rate: 0.0031,
      mttd_minutes: 4.7,
      daily_counts: DAILY_COUNTS,
      by_severity: bySeverity,
      top_metrics: topMetrics,
    }) as T;
  }

  // ── Metrics config (alias) ────────────────────────────────────────────────
  if (clean === 'metrics' && method === 'GET') {
    return ok({ items: _metricsConfig, total: _metricsConfig.length }) as T;
  }

  return ok({ ok: true, mocked: true, path, method }) as T;
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ──────────────────────────────────────────────────────────────────────────────

def _backend_main_py() -> str:
    return '''\
"""
FastAPI Anomaly Detection API.

Endpoints:
  POST /api/auth/login            — login, returns JWT
  POST /api/auth/register         — register new user
  GET  /api/dashboard             — summary stats + metric time-series
  GET  /api/anomalies             — list detected anomalies (filterable)
  GET  /api/alerts                — list active alerts
  GET  /api/thresholds            — list metric threshold configs
  PUT  /api/thresholds/{id}       — update threshold config
  GET  /api/reports/summary       — 30-day analytics summary
"""
from __future__ import annotations

import datetime
import random
import statistics
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from anomaly_engine import detect_anomalies

app = FastAPI(title="Anomaly Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# In-memory store (replace with DB in prod)
# ─────────────────────────────────────────

_USERS: List[Dict[str, Any]] = [
    {"id": 1, "email": "admin@example.com",   "password": "admin123",   "name": "Admin User",   "role": "ADMIN"},
    {"id": 2, "email": "analyst@example.com", "password": "analyst123", "name": "Data Analyst", "role": "ANALYST"},
]

_METRICS_CONFIG: List[Dict[str, Any]] = [
    {"metric_id": "cpu_usage",       "metric_name": "CPU Usage",       "unit": "%",    "threshold_min": 0.0,  "threshold_max": 85.0,   "zscore_threshold": 3.0, "enabled": True,  "last_triggered": "2026-10-05T08:22:00Z"},
    {"metric_id": "memory_usage",    "metric_name": "Memory Usage",    "unit": "%",    "threshold_min": 0.0,  "threshold_max": 90.0,   "zscore_threshold": 3.0, "enabled": True,  "last_triggered": "2026-10-05T08:14:00Z"},
    {"metric_id": "request_latency", "metric_name": "Request Latency", "unit": "ms",   "threshold_min": 0.0,  "threshold_max": 2000.0, "zscore_threshold": 2.5, "enabled": True,  "last_triggered": "2026-10-05T10:47:00Z"},
    {"metric_id": "error_rate",      "metric_name": "Error Rate",      "unit": "%",    "threshold_min": 0.0,  "threshold_max": 5.0,    "zscore_threshold": 2.5, "enabled": True,  "last_triggered": "2026-10-04T22:11:00Z"},
    {"metric_id": "disk_io",         "metric_name": "Disk I/O",        "unit": "MB/s", "threshold_min": 0.0,  "threshold_max": 500.0,  "zscore_threshold": 3.0, "enabled": True},
    {"metric_id": "network_in",      "metric_name": "Network In",      "unit": "Mbps", "threshold_min": 0.0,  "threshold_max": 1000.0, "zscore_threshold": 3.0, "enabled": False},
]

_ANOMALIES: List[Dict[str, Any]] = [
    {"id": "a1",  "timestamp": "2026-10-05T18:22:00Z", "metric_name": "CPU Usage",       "metric_id": "cpu_usage",       "value": 97.3,  "expected_min": 20.0, "expected_max": 85.0,   "severity": "Critical", "status": "Open",         "anomaly_score": 5.12},
    {"id": "a2",  "timestamp": "2026-10-05T15:47:00Z", "metric_name": "Request Latency", "metric_id": "request_latency", "value": 4280.0,"expected_min": 100.0,"expected_max": 2000.0, "severity": "High",     "status": "Acknowledged", "anomaly_score": 4.03},
    {"id": "a3",  "timestamp": "2026-10-05T12:14:00Z", "metric_name": "Error Rate",      "metric_id": "error_rate",      "value": 18.5,  "expected_min": 0.0,  "expected_max": 5.0,    "severity": "Critical", "status": "Open",         "anomaly_score": 6.77},
    {"id": "a4",  "timestamp": "2026-10-05T09:33:00Z", "metric_name": "Memory Usage",    "metric_id": "memory_usage",    "value": 96.1,  "expected_min": 40.0, "expected_max": 90.0,   "severity": "High",     "status": "Open",         "anomaly_score": 3.98},
    {"id": "a5",  "timestamp": "2026-10-04T23:11:00Z", "metric_name": "Disk I/O",        "metric_id": "disk_io",         "value": 712.0, "expected_min": 0.0,  "expected_max": 500.0,  "severity": "Medium",   "status": "Resolved",     "anomaly_score": 3.14},
    {"id": "a6",  "timestamp": "2026-10-04T19:05:00Z", "metric_name": "CPU Usage",       "metric_id": "cpu_usage",       "value": 91.8,  "expected_min": 20.0, "expected_max": 85.0,   "severity": "High",     "status": "Resolved",     "anomaly_score": 3.67},
    {"id": "a7",  "timestamp": "2026-10-04T14:44:00Z", "metric_name": "Request Latency", "metric_id": "request_latency", "value": 3100.0,"expected_min": 100.0,"expected_max": 2000.0, "severity": "High",     "status": "Resolved",     "anomaly_score": 3.51},
    {"id": "a8",  "timestamp": "2026-10-04T10:21:00Z", "metric_name": "Network In",      "metric_id": "network_in",      "value": 1450.0,"expected_min": 0.0,  "expected_max": 1000.0, "severity": "Medium",   "status": "Resolved",     "anomaly_score": 2.88},
    {"id": "a9",  "timestamp": "2026-10-03T22:58:00Z", "metric_name": "Error Rate",      "metric_id": "error_rate",      "value": 8.2,   "expected_min": 0.0,  "expected_max": 5.0,    "severity": "High",     "status": "Resolved",     "anomaly_score": 3.45},
    {"id": "a10", "timestamp": "2026-10-03T16:37:00Z", "metric_name": "Memory Usage",    "metric_id": "memory_usage",    "value": 92.4,  "expected_min": 40.0, "expected_max": 90.0,   "severity": "Medium",   "status": "Resolved",     "anomaly_score": 2.71},
    {"id": "a11", "timestamp": "2026-10-03T11:19:00Z", "metric_name": "CPU Usage",       "metric_id": "cpu_usage",       "value": 88.7,  "expected_min": 20.0, "expected_max": 85.0,   "severity": "Medium",   "status": "Resolved",     "anomaly_score": 2.53},
    {"id": "a12", "timestamp": "2026-10-02T08:44:00Z", "metric_name": "Disk I/O",        "metric_id": "disk_io",         "value": 620.0, "expected_min": 0.0,  "expected_max": 500.0,  "severity": "Low",      "status": "Resolved",     "anomaly_score": 2.21},
]

_ALERTS: List[Dict[str, Any]] = [
    {"id": "al1", "metric_name": "CPU Usage",       "threshold": 85.0,   "current_value": 97.3,  "severity": "Critical", "triggered_at": "2026-10-05T18:22:00Z", "status": "Active"},
    {"id": "al2", "metric_name": "Error Rate",      "threshold": 5.0,    "current_value": 18.5,  "severity": "Critical", "triggered_at": "2026-10-05T12:14:00Z", "status": "Active"},
    {"id": "al3", "metric_name": "Memory Usage",    "threshold": 90.0,   "current_value": 96.1,  "severity": "High",     "triggered_at": "2026-10-05T09:33:00Z", "status": "Active"},
    {"id": "al4", "metric_name": "Request Latency", "threshold": 2000.0, "current_value": 4280.0,"severity": "High",     "triggered_at": "2026-10-05T15:47:00Z", "status": "Acknowledged"},
]

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

_BASE_PARAMS: Dict[str, tuple] = {
    "cpu_usage":       (45.0, 10.0,  93.0),
    "memory_usage":    (65.0,  8.0,  94.0),
    "request_latency": (250.0, 50.0, 3400.0),
    "error_rate":      (0.5,   0.3,  15.0),
    "disk_io":         (120.0, 30.0, 640.0),
    "network_in":      (350.0, 80.0, 1350.0),
}


def _generate_time_series(metric_id: str, cfg: Dict[str, Any], hours: int = 24) -> List[Dict[str, Any]]:
    now = datetime.datetime.utcnow()
    interval_min = 5
    n = (hours * 60) // interval_min
    base, std, spike = _BASE_PARAMS.get(metric_id, (50.0, 10.0, 100.0))
    rng = random.Random(sum(ord(c) for c in metric_id))  # deterministic seed per metric
    anomaly_indices = set(rng.sample(range(10, n - 10), min(4, max(1, (n - 20) // 10))))

    raw: List[float] = []
    for i in range(n):
        raw.append(spike * (0.9 + rng.random() * 0.25) if i in anomaly_indices
                   else max(0.0, base + rng.gauss(0, std)))

    anomaly_pts = detect_anomalies(raw, threshold=cfg.get("zscore_threshold", 3.0))
    points = []
    for i, (val, ap) in enumerate(zip(raw, anomaly_pts)):
        ts = now - datetime.timedelta(minutes=(n - i) * interval_min)
        points.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "value": round(val, 2),
            "is_anomaly": ap.is_anomaly,
            "anomaly_score": round(ap.score, 3),
        })
    return points


# ─────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class ThresholdUpdate(BaseModel):
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    zscore_threshold: Optional[float] = None
    enabled: Optional[bool] = None


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = next((u for u in _USERS if u["email"] == req.email and u["password"] == req.password), None)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "access_token": f"dev-token-{user['id']}",
        "user": {"id": user["id"], "name": user["name"], "email": user["email"], "role": user["role"]},
    }


@app.post("/api/auth/register")
def register(req: RegisterRequest):
    if any(u["email"] == req.email for u in _USERS):
        raise HTTPException(status_code=409, detail="Email already registered")
    new_user = {
        "id": len(_USERS) + 1, "email": req.email, "password": req.password,
        "name": req.name or req.email, "role": "ANALYST",
    }
    _USERS.append(new_user)
    return {"ok": True}


@app.get("/api/dashboard")
def dashboard():
    enabled = [m for m in _METRICS_CONFIG if m["enabled"]]
    metrics_with_data = []
    for cfg in enabled:
        series = _generate_time_series(cfg["metric_id"], cfg)
        metrics_with_data.append({
            **cfg,
            "threshold_low": cfg["threshold_min"],
            "threshold_high": cfg["threshold_max"],
            "data": series,
        })
    today = datetime.date.today().isoformat()
    today_count = sum(1 for a in _ANOMALIES if a["timestamp"].startswith(today)) or 3
    scores = [a["anomaly_score"] for a in _ANOMALIES]
    avg_score = round(statistics.mean(scores), 2) if scores else 0.0
    return {
        "total_anomalies_today": today_count,
        "active_alerts": sum(1 for a in _ALERTS if a["status"] == "Active"),
        "metrics_monitored": len(enabled),
        "avg_anomaly_score": avg_score,
        "metrics": metrics_with_data,
    }


@app.get("/api/anomalies")
def list_anomalies(severity: Optional[str] = None, status: Optional[str] = None, limit: int = 100):
    items = list(_ANOMALIES)
    if severity:
        items = [a for a in items if a["severity"].lower() == severity.lower()]
    if status:
        items = [a for a in items if a["status"].lower() == status.lower()]
    return {"items": items[:limit], "total": len(items)}


@app.get("/api/alerts")
def list_alerts():
    return {"items": _ALERTS, "total": len(_ALERTS)}


@app.get("/api/thresholds")
def list_thresholds():
    return {"items": _METRICS_CONFIG, "total": len(_METRICS_CONFIG)}


@app.put("/api/thresholds/{metric_id}")
def update_threshold(metric_id: str, update: ThresholdUpdate):
    cfg = next((m for m in _METRICS_CONFIG if m["metric_id"] == metric_id), None)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Metric '{metric_id}' not found")
    if update.threshold_min is not None:
        cfg["threshold_min"] = update.threshold_min
    if update.threshold_max is not None:
        cfg["threshold_max"] = update.threshold_max
    if update.zscore_threshold is not None:
        cfg["zscore_threshold"] = update.zscore_threshold
    if update.enabled is not None:
        cfg["enabled"] = update.enabled
    return cfg


@app.get("/api/reports/summary")
def reports_summary():
    today = datetime.date.today()
    rng = random.Random(42)
    daily = []
    for i in range(30):
        d = today - datetime.timedelta(days=29 - i)
        count = sum(1 for a in _ANOMALIES if a["timestamp"][:10] == d.isoformat())
        if count == 0:
            count = rng.randint(0, 6)
        daily.append({"date": d.strftime("%b %d"), "count": count})

    sev_map: Dict[str, int] = {}
    for a in _ANOMALIES:
        sev_map[a["severity"]] = sev_map.get(a["severity"], 0) + 1
    by_severity = [{"name": k, "value": v} for k, v in sev_map.items()]

    metric_agg: Dict[str, Dict[str, Any]] = {}
    sev_order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    for a in _ANOMALIES:
        mn = a["metric_name"]
        if mn not in metric_agg:
            metric_agg[mn] = {"metric_name": mn, "count": 0, "severity": "Low"}
        metric_agg[mn]["count"] += 1
        if sev_order.get(a["severity"], 0) > sev_order.get(metric_agg[mn]["severity"], 0):
            metric_agg[mn]["severity"] = a["severity"]
    top_metrics = sorted(metric_agg.values(), key=lambda x: x["count"], reverse=True)[:5]

    total = len(_ANOMALIES)
    rate = round(total / (30 * 24 * 60 / 5), 6)
    return {
        "total_anomalies": total,
        "anomaly_rate": rate,
        "mttd_minutes": round(4.2 + rng.random() * 2, 1),
        "daily_counts": daily,
        "by_severity": by_severity,
        "top_metrics": top_metrics,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''


def _anomaly_engine_py() -> str:
    return '''\
"""Z-score based anomaly detection engine.

Usage:
    from anomaly_engine import detect_anomalies, AnomalyPoint

    points = detect_anomalies(values, threshold=3.0)
    for p in points:
        if p.is_anomaly:
            print(f"Anomaly at index {p.index}: value={p.value}, z-score={p.score:.2f}")
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AnomalyPoint:
    """Result for a single time-series data point."""
    index: int
    value: float
    score: float        # absolute z-score
    is_anomaly: bool


def detect_anomalies(
    values: List[float],
    threshold: float = 3.0,
    window: Optional[int] = None,
) -> List[AnomalyPoint]:
    """
    Detect anomalies in a time-series using the z-score method.

    Args:
        values:    Ordered list of numeric metric readings.
        threshold: Z-score above which a point is flagged (default 3.0).
        window:    Optional rolling window size for local mean/std.
                   When None, uses global mean/std across all values.

    Returns:
        List[AnomalyPoint] — one per input value, with ``is_anomaly`` flag.
    """
    n = len(values)
    if n == 0:
        return []
    if n < 3:
        return [AnomalyPoint(index=i, value=v, score=0.0, is_anomaly=False)
                for i, v in enumerate(values)]

    result: List[AnomalyPoint] = []

    if window and window > 0:
        # Rolling-window z-score — better for non-stationary signals
        for i, v in enumerate(values):
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            segment = values[lo:hi]
            if len(segment) < 3:
                result.append(AnomalyPoint(index=i, value=v, score=0.0, is_anomaly=False))
                continue
            mean = statistics.mean(segment)
            try:
                std = statistics.stdev(segment)
            except statistics.StatisticsError:
                std = 0.0
            if std < 1e-10:
                result.append(AnomalyPoint(index=i, value=v, score=0.0, is_anomaly=False))
                continue
            score = abs(v - mean) / std
            result.append(AnomalyPoint(
                index=i, value=v, score=round(score, 4), is_anomaly=score >= threshold,
            ))
    else:
        # Global z-score — simple and effective for stationary signals
        mean = statistics.mean(values)
        try:
            std = statistics.stdev(values)
        except statistics.StatisticsError:
            std = 0.0

        for i, v in enumerate(values):
            if std < 1e-10:
                result.append(AnomalyPoint(index=i, value=v, score=0.0, is_anomaly=False))
            else:
                score = abs(v - mean) / std
                result.append(AnomalyPoint(
                    index=i, value=v, score=round(score, 4), is_anomaly=score >= threshold,
                ))

    return result
'''
