"""Config-driven anomaly detection frontend + shared data bundle for mock API and backend."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app_builder.services.anomaly_bundle_from_plan import (
    apply_plan_to_bundle,
    merge_theme_colors,
)


def _kpi_visuals(colors: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    primary = colors.get("primary", "#1565C0")
    return {
        "error": {"iconBg": "#FFEBEE", "iconColor": colors.get("danger", "#C62828")},
        "warning": {"iconBg": "#FFF3E0", "iconColor": colors.get("warning", "#E65100")},
        "primary": {"iconBg": "#E3F2FD", "iconColor": primary},
        "success": {"iconBg": "#E8F5E9", "iconColor": colors.get("success", "#2E7D32")},
    }


def build_data_bundle(
    title: str,
    colors: Dict[str, str],
    requirement: str = "",
    prd: str = "",
    uiux: str = "",
    architecture: Optional[Dict[str, Any]] = None,
    design_tokens: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Single source of truth for UI config and demo data (customized from PRD / UI/UX / architecture)."""
    colors = merge_theme_colors(colors, design_tokens, uiux)
    primary = colors.get("primary", "#1565C0")
    app_name = (title or "Anomaly Detection Platform").strip()
    header = colors.get("header_title") or "Operations Center"

    navigation: List[Dict[str, str]] = [
        {"path": "/dashboard", "label": "Dashboard", "icon": "Dashboard", "page": "dashboard"},
        {"path": "/anomalies", "label": "Anomaly Feed", "icon": "BugReport", "page": "anomalies"},
        {"path": "/alerts", "label": "Alert Config", "icon": "NotificationsActive", "page": "alerts"},
        {"path": "/reports", "label": "Reports", "icon": "Assessment", "page": "reports"},
    ]

    metrics_config = [
        {
            "metric_id": "cpu_usage",
            "metric_name": "CPU Usage",
            "unit": "%",
            "threshold_min": 0,
            "threshold_max": 85,
            "zscore_threshold": 3.0,
            "enabled": True,
            "last_triggered": "2026-10-04T14:22:00Z",
        },
        {
            "metric_id": "memory_usage",
            "metric_name": "Memory Usage",
            "unit": "%",
            "threshold_min": 0,
            "threshold_max": 90,
            "zscore_threshold": 3.0,
            "enabled": True,
            "last_triggered": "2026-10-05T08:14:00Z",
        },
        {
            "metric_id": "request_latency",
            "metric_name": "Request Latency",
            "unit": "ms",
            "threshold_min": 0,
            "threshold_max": 2000,
            "zscore_threshold": 2.5,
            "enabled": True,
            "last_triggered": "2026-10-05T10:47:00Z",
        },
        {
            "metric_id": "error_rate",
            "metric_name": "Error Rate",
            "unit": "%",
            "threshold_min": 0,
            "threshold_max": 5,
            "zscore_threshold": 2.5,
            "enabled": True,
            "last_triggered": "2026-10-03T22:11:00Z",
        },
        {
            "metric_id": "disk_io",
            "metric_name": "Disk I/O",
            "unit": "MB/s",
            "threshold_min": 0,
            "threshold_max": 500,
            "zscore_threshold": 3.0,
            "enabled": True,
        },
        {
            "metric_id": "network_in",
            "metric_name": "Network In",
            "unit": "Mbps",
            "threshold_min": 0,
            "threshold_max": 1000,
            "zscore_threshold": 3.0,
            "enabled": False,
        },
    ]

    anomalies = [
        {"id": "a1", "timestamp": "2026-10-05T18:22:00Z", "metric_name": "CPU Usage", "metric_id": "cpu_usage", "value": 97.3, "expected_min": 20, "expected_max": 85, "severity": "Critical", "status": "Open", "anomaly_score": 5.12},
        {"id": "a2", "timestamp": "2026-10-05T15:47:00Z", "metric_name": "Request Latency", "metric_id": "request_latency", "value": 4280, "expected_min": 100, "expected_max": 2000, "severity": "High", "status": "Acknowledged", "anomaly_score": 4.03},
        {"id": "a3", "timestamp": "2026-10-05T12:14:00Z", "metric_name": "Error Rate", "metric_id": "error_rate", "value": 18.5, "expected_min": 0, "expected_max": 5, "severity": "Critical", "status": "Open", "anomaly_score": 6.77},
        {"id": "a4", "timestamp": "2026-10-05T09:33:00Z", "metric_name": "Memory Usage", "metric_id": "memory_usage", "value": 96.1, "expected_min": 40, "expected_max": 90, "severity": "High", "status": "Open", "anomaly_score": 3.98},
        {"id": "a5", "timestamp": "2026-10-04T23:11:00Z", "metric_name": "Disk I/O", "metric_id": "disk_io", "value": 712, "expected_min": 0, "expected_max": 500, "severity": "Medium", "status": "Resolved", "anomaly_score": 3.14},
        {"id": "a6", "timestamp": "2026-10-04T19:05:00Z", "metric_name": "CPU Usage", "metric_id": "cpu_usage", "value": 91.8, "expected_min": 20, "expected_max": 85, "severity": "High", "status": "Resolved", "anomaly_score": 3.67},
        {"id": "a7", "timestamp": "2026-10-04T14:44:00Z", "metric_name": "Request Latency", "metric_id": "request_latency", "value": 3100, "expected_min": 100, "expected_max": 2000, "severity": "High", "status": "Resolved", "anomaly_score": 3.51},
        {"id": "a8", "timestamp": "2026-10-04T10:21:00Z", "metric_name": "Network In", "metric_id": "network_in", "value": 1450, "expected_min": 0, "expected_max": 1000, "severity": "Medium", "status": "Resolved", "anomaly_score": 2.88},
        {"id": "a9", "timestamp": "2026-10-03T22:58:00Z", "metric_name": "Error Rate", "metric_id": "error_rate", "value": 8.2, "expected_min": 0, "expected_max": 5, "severity": "High", "status": "Resolved", "anomaly_score": 3.45},
        {"id": "a10", "timestamp": "2026-10-03T16:37:00Z", "metric_name": "Memory Usage", "metric_id": "memory_usage", "value": 92.4, "expected_min": 40, "expected_max": 90, "severity": "Medium", "status": "Resolved", "anomaly_score": 2.71},
        {"id": "a11", "timestamp": "2026-10-03T11:19:00Z", "metric_name": "CPU Usage", "metric_id": "cpu_usage", "value": 88.7, "expected_min": 20, "expected_max": 85, "severity": "Medium", "status": "Resolved", "anomaly_score": 2.53},
        {"id": "a12", "timestamp": "2026-10-02T08:44:00Z", "metric_name": "Disk I/O", "metric_id": "disk_io", "value": 620, "expected_min": 0, "expected_max": 500, "severity": "Low", "status": "Resolved", "anomaly_score": 2.21},
    ]

    alerts = [
        {"id": "al1", "metric_name": "CPU Usage", "threshold": 85, "current_value": 97.3, "severity": "Critical", "triggered_at": "2026-10-05T18:22:00Z", "status": "Active"},
        {"id": "al2", "metric_name": "Error Rate", "threshold": 5, "current_value": 18.5, "severity": "Critical", "triggered_at": "2026-10-05T12:14:00Z", "status": "Active"},
        {"id": "al3", "metric_name": "Memory Usage", "threshold": 90, "current_value": 96.1, "severity": "High", "triggered_at": "2026-10-05T09:33:00Z", "status": "Active"},
        {"id": "al4", "metric_name": "Request Latency", "threshold": 2000, "current_value": 4280, "severity": "High", "triggered_at": "2026-10-05T15:47:00Z", "status": "Acknowledged"},
    ]

    bundle: Dict[str, Any] = {
        "branding": {
            "appName": app_name,
            "headerTitle": header,
            "loginTitle": app_name,
            "loginSubtitle": "Sign in to monitor metrics and respond to anomalies",
            "loginDemoHint": "Demo: admin@example.com / admin123",
        },
        "theme": {
            "primary": primary,
            "primaryDark": colors.get("primary_dark", "#0D47A1"),
            "secondary": colors.get("secondary", "#7B1FA2"),
            "background": colors.get("background", "#F5F5F5"),
            "surface": colors.get("surface", "#FFFFFF"),
            "text": colors.get("text", "#212121"),
            "muted": colors.get("muted", "#757575"),
            "anomalyMarker": colors.get("anomaly_marker", "#C62828"),
            "drawerWidth": 260,
        },
        "navigation": navigation,
        "pages": {
            "dashboard": {
                "title": "Anomaly Detection Dashboard",
                "chartTitle": "Metric time series — last 24 hours",
                "chartHint": "Red markers indicate detected anomalies (z-score above threshold)",
                "metricSelectLabel": "Select metric",
                "kpis": [],  # filled below with theme-aware icon colors
            },
            "anomalies": {
                "title": "Anomaly feed",
                "searchPlaceholder": "Search by metric name…",
                "severityFilterLabel": "Severity",
                "severityOptions": ["All", "Critical", "High", "Medium", "Low"],
                "columns": ["Timestamp", "Metric", "Value", "Expected range", "Score", "Severity", "Status"],
            },
            "alerts": {
                "title": "Alert configuration",
                "subtitle": "Configure detection thresholds and z-score sensitivity for each monitored metric.",
                "zScoreHelper": "Lower value = more sensitive. Typical range: 2.5 – 4.0",
            },
            "reports": {
                "title": "Anomaly reports",
                "dailyChartTitle": "Anomalies per day — last 30 days",
                "severityChartTitle": "By severity",
                "topMetricsTitle": "Top anomalous metrics",
                "summaryCards": [
                    {"key": "total_anomalies", "label": "Total anomalies (30d)"},
                    {"key": "anomaly_rate", "label": "Anomaly rate", "format": "percent"},
                    {"key": "mttd_minutes", "label": "Avg MTTD", "format": "minutes"},
                ],
                "tableColumns": ["Rank", "Metric", "Count", "Severity"],
            },
        },
        "severityStyles": {
            "Low": {"bg": "#E8F5E9", "color": "#2E7D32"},
            "Medium": {"bg": "#FFF8E1", "color": "#F57F17"},
            "High": {"bg": "#FFF3E0", "color": "#E65100"},
            "Critical": {"bg": "#FFEBEE", "color": "#B71C1C"},
        },
        "statusChipColor": {
            "Open": "warning",
            "Acknowledged": "default",
            "Resolved": "success",
        },
        "piePalette": ["#81C784", "#FFB74D", "#FF8A65", "#E57373"],
        "users": [
            {"id": 1, "email": "admin@example.com", "password": "admin123", "name": "Admin User", "role": "ADMIN"},
            {"id": 2, "email": "analyst@example.com", "password": "analyst123", "name": "Data Analyst", "role": "ANALYST"},
        ],
        "metricsConfig": metrics_config,
        "anomalies": anomalies,
        "alerts": alerts,
        "reportsSummary": {
            "anomaly_rate": 0.0031,
            "mttd_minutes": 4.7,
            "top_metrics": [
                {"metric_name": "CPU Usage", "count": 4, "severity": "Critical"},
                {"metric_name": "Error Rate", "count": 3, "severity": "Critical"},
                {"metric_name": "Request Latency", "count": 3, "severity": "High"},
                {"metric_name": "Memory Usage", "count": 3, "severity": "High"},
                {"metric_name": "Disk I/O", "count": 2, "severity": "Medium"},
            ],
        },
        "timeSeriesParams": {
            "cpu_usage": [45, 10, 93],
            "memory_usage": [65, 8, 94],
            "request_latency": [250, 50, 3400],
            "error_rate": [0.5, 0.3, 15],
            "disk_io": [120, 30, 640],
            "network_in": [350, 80, 1350],
        },
        "polling": {
            "dashboardMs": 30000,
            "anomaliesMs": 15000,
        },
    }

    kpi_defs = [
        ("total_anomalies_today", "Anomalies today", "WarningAmber", "error"),
        ("active_alerts", "Active alerts", "NotificationsActive", "warning"),
        ("metrics_monitored", "Metrics monitored", "Timeline", "primary"),
        ("avg_anomaly_score", "Avg anomaly score", "Speed", "success"),
    ]
    visuals = _kpi_visuals(colors)
    bundle["pages"]["dashboard"]["kpis"] = [
        {
            "key": key,
            "label": label,
            "icon": icon,
            "tone": tone,
            "iconBg": visuals[tone]["iconBg"],
            "iconColor": visuals[tone]["iconColor"],
        }
        for key, label, icon, tone in kpi_defs
    ]

    return apply_plan_to_bundle(
        bundle,
        requirement=requirement,
        prd=prd,
        uiux=uiux,
        architecture=architecture,
        title=title,
    )


def _ts_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def build_anomaly_frontend_files(
    title: str,
    colors: Dict[str, str],
    requirement: str = "",
    prd: str = "",
    uiux: str = "",
    architecture: Optional[Dict[str, Any]] = None,
    design_tokens: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    bundle = build_data_bundle(
        title,
        colors,
        requirement=requirement,
        prd=prd,
        uiux=uiux,
        architecture=architecture,
        design_tokens=design_tokens,
    )
    ui_config = {
        k: bundle[k]
        for k in (
            "branding",
            "theme",
            "navigation",
            "pages",
            "severityStyles",
            "statusChipColor",
            "piePalette",
            "polling",
        )
    }
    mock_payload = {
        k: bundle[k]
        for k in (
            "users",
            "metricsConfig",
            "anomalies",
            "alerts",
            "reportsSummary",
            "timeSeriesParams",
        )
    }

    files: Dict[str, str] = {
        "frontend/src/config/appConfig.ts": f"""// Generated from PRD + UI/UX + design tokens — edit to rebrand or restructure navigation.
export const appConfig = {_ts_json(ui_config)} as const;

export type AppConfig = typeof appConfig;
export type NavItem = AppConfig['navigation'][number];
export type PageKey = NavItem['page'];
""",
        "frontend/src/data/mockData.ts": f"""// Generated demo dataset — drives mock API and can mirror backend seed data.
export const mockData = {_ts_json(mock_payload)} as const;

export type MockData = typeof mockData;
""",
    }

    static_files = _static_frontend_sources()
    files.update(static_files)
    return files


def build_anomaly_backend_files(
    title: str,
    colors: Dict[str, str],
    requirement: str = "",
    prd: str = "",
    uiux: str = "",
    architecture: Optional[Dict[str, Any]] = None,
    design_tokens: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    bundle = build_data_bundle(
        title,
        colors,
        requirement=requirement,
        prd=prd,
        uiux=uiux,
        architecture=architecture,
        design_tokens=design_tokens,
    )
    return {
        "backend/app_data.json": json.dumps(bundle, indent=2),
        "backend/main.py": _backend_main_py(),
        "backend/anomaly_engine.py": _anomaly_engine_py(),
    }


def _static_frontend_sources() -> Dict[str, str]:
    """Stable TS/TSX sources — all branding and content come from appConfig + mockData."""
    return {
        "frontend/src/theme/index.ts": _theme_index(),
        "frontend/src/utils/navIcons.tsx": _nav_icons(),
        "frontend/src/layout/AppLayout.tsx": _app_layout(),
        "frontend/src/App.tsx": _app_tsx(),
        "frontend/src/pages/DashboardPage.tsx": _dashboard_page(),
        "frontend/src/pages/AnomalyFeedPage.tsx": _anomaly_feed_page(),
        "frontend/src/pages/AlertConfigPage.tsx": _alert_config_page(),
        "frontend/src/pages/ReportsPage.tsx": _reports_page(),
        "frontend/src/api/mock.ts": _mock_ts(),
    }


def _theme_index() -> str:
    return """\
import { createTheme } from '@mui/material/styles';
import { appConfig } from '../config/appConfig';

const t = appConfig.theme;

export const appTheme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: t.primary, dark: t.primaryDark },
    secondary: { main: t.secondary },
    background: { default: t.background, paper: t.surface },
    text: { primary: t.text, secondary: t.muted },
    error: { main: t.anomalyMarker },
  },
  shape: { borderRadius: 10 },
  typography: {
    fontFamily: '\"Inter\", \"Roboto\", \"Helvetica\", \"Arial\", sans-serif',
    h4: { fontWeight: 800 },
    h6: { fontWeight: 700 },
  },
  components: {
    MuiCard: { styleOverrides: { root: { borderRadius: 12 } } },
    MuiDrawer: {
      styleOverrides: {
        paper: { borderRight: 'none', boxShadow: '2px 0 12px rgba(0,0,0,0.06)' },
      },
    },
  },
});
"""


def _nav_icons() -> str:
    return """\
import DashboardIcon from '@mui/icons-material/Dashboard';
import BugReportIcon from '@mui/icons-material/BugReport';
import NotificationsActiveIcon from '@mui/icons-material/NotificationsActive';
import AssessmentIcon from '@mui/icons-material/Assessment';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import TimelineIcon from '@mui/icons-material/Timeline';
import SpeedIcon from '@mui/icons-material/Speed';
import type { SvgIconComponent } from '@mui/icons-material';

const MAP: Record<string, SvgIconComponent> = {
  Dashboard: DashboardIcon,
  BugReport: BugReportIcon,
  NotificationsActive: NotificationsActiveIcon,
  Assessment: AssessmentIcon,
  WarningAmber: WarningAmberIcon,
  Timeline: TimelineIcon,
  Speed: SpeedIcon,
};

export function resolveNavIcon(name: string): SvgIconComponent {
  return MAP[name] ?? DashboardIcon;
}
"""


def _app_layout() -> str:
    return """\
import {
  AppBar, Box, Drawer, List, ListItemButton, ListItemIcon, ListItemText, Toolbar, Typography,
} from '@mui/material';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { appConfig } from '../config/appConfig';
import { resolveNavIcon } from '../utils/navIcons';

export default function AppLayout() {
  const location = useLocation();
  const { branding, theme, navigation } = appConfig;
  const drawerW = theme.drawerWidth;

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      <Drawer
        variant="permanent"
        sx={{
          width: drawerW,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerW,
            boxSizing: 'border-box',
            bgcolor: theme.primary,
            color: '#fff',
          },
        }}
      >
        <Toolbar sx={{ px: 2, py: 1.5 }}>
          <Typography variant="h6" fontWeight={800} sx={{ color: '#fff', fontSize: '0.9rem', lineHeight: 1.35 }}>
            {branding.appName}
          </Typography>
        </Toolbar>
        <List sx={{ px: 1 }}>
          {navigation.map((item) => {
            const Icon = resolveNavIcon(item.icon);
            const selected = location.pathname === item.path;
            return (
              <ListItemButton
                key={item.path}
                component={Link}
                to={item.path}
                selected={selected}
                sx={{
                  borderRadius: 2,
                  mb: 0.5,
                  '&.Mui-selected': { bgcolor: 'rgba(255,255,255,0.18)' },
                  '&:hover': { bgcolor: 'rgba(255,255,255,0.1)' },
                }}
              >
                <ListItemIcon sx={{ color: '#fff', minWidth: 36 }}>
                  <Icon fontSize="small" />
                </ListItemIcon>
                <ListItemText
                  primary={item.label}
                  primaryTypographyProps={{
                    color: '#fff',
                    fontWeight: selected ? 700 : 400,
                    fontSize: '0.9rem',
                  }}
                />
              </ListItemButton>
            );
          })}
        </List>
      </Drawer>
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <AppBar
          position="sticky"
          elevation={0}
          sx={{
            bgcolor: theme.surface,
            color: theme.text,
            borderBottom: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Toolbar>
            <Typography variant="h6" fontWeight={700} sx={{ flex: 1 }}>
              {branding.headerTitle}
            </Typography>
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


def _app_tsx() -> str:
    return """\
import type { ComponentType } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import AppLayout from './layout/AppLayout';
import DashboardPage from './pages/DashboardPage';
import AnomalyFeedPage from './pages/AnomalyFeedPage';
import AlertConfigPage from './pages/AlertConfigPage';
import ReportsPage from './pages/ReportsPage';
import { appConfig } from './config/appConfig';
import type { PageKey } from './config/appConfig';

const PAGE_COMPONENTS: Record<PageKey, ComponentType> = {
  dashboard: DashboardPage,
  anomalies: AnomalyFeedPage,
  alerts: AlertConfigPage,
  reports: ReportsPage,
};

const defaultPath = appConfig.navigation[0]?.path ?? '/dashboard';

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route path="/" element={<Navigate to={defaultPath} replace />} />
        {appConfig.navigation.map((item) => {
          const Page = PAGE_COMPONENTS[item.page];
          if (!Page) return null;
          return <Route key={item.path} path={item.path} element={<Page />} />;
        })}
      </Route>
      <Route path="*" element={<Navigate to={defaultPath} replace />} />
    </Routes>
  );
}
"""


def _dashboard_page() -> str:
    return """\
import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Box, Card, CardContent, Typography, Select, MenuItem, FormControl,
  InputLabel, Chip, Stack, CircularProgress, Grid2,
} from '@mui/material';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine, ResponsiveContainer,
} from 'recharts';
import { apiFetch } from '../api/client';
import { appConfig } from '../config/appConfig';
import { resolveNavIcon } from '../utils/navIcons';

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

function AnomalyDot(props: any) {
  const { cx, cy, payload } = props;
  if (!payload?.is_anomaly) return null;
  return (
    <circle
      cx={cx}
      cy={cy}
      r={7}
      fill={appConfig.theme.anomalyMarker}
      stroke="#fff"
      strokeWidth={2}
    />
  );
}

export default function DashboardPage() {
  const copy = appConfig.pages.dashboard;
  const [selectedMetricId, setSelectedMetricId] = useState('');

  const { data, isLoading } = useQuery<DashboardStats>({
    queryKey: ['dashboard'],
    queryFn: () => apiFetch<DashboardStats>('/api/dashboard'),
    refetchInterval: appConfig.polling.dashboardMs,
  });

  const metrics = data?.metrics ?? [];
  const activeMetric = metrics.find((m) => m.metric_id === selectedMetricId) ?? metrics[0];

  const kpiValues = useMemo(() => ({
    total_anomalies_today: data?.total_anomalies_today ?? 0,
    active_alerts: data?.active_alerts ?? 0,
    metrics_monitored: data?.metrics_monitored ?? 0,
    avg_anomaly_score: (data?.avg_anomaly_score ?? 0).toFixed(2),
  }), [data]);

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
        <CircularProgress color="primary" />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" color="primary" sx={{ mb: copy.subtitle ? 1 : 3 }}>
        {copy.title}
      </Typography>
      {'subtitle' in copy && copy.subtitle ? (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>{copy.subtitle}</Typography>
      ) : null}

      <Grid2 container spacing={3} sx={{ mb: 4 }}>
        {copy.kpis.map((kpi) => {
          const Icon = resolveNavIcon(kpi.icon);
          const value = kpiValues[kpi.key as keyof typeof kpiValues];
          return (
            <Grid2 key={kpi.key} size={{ xs: 12, sm: 6, md: 3 }}>
              <Card>
                <CardContent>
                  <Stack direction="row" alignItems="center" spacing={2}>
                    <Box
                      sx={{
                        width: 48,
                        height: 48,
                        borderRadius: 2,
                        bgcolor: kpi.iconBg,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: kpi.iconColor,
                      }}
                    >
                      <Icon />
                    </Box>
                    <Box>
                      <Typography variant="h4" fontWeight={800} sx={{ color: kpi.iconColor, lineHeight: 1 }}>
                        {value}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">{kpi.label}</Typography>
                    </Box>
                  </Stack>
                </CardContent>
              </Card>
            </Grid2>
          );
        })}
      </Grid2>

      <Card>
        <CardContent>
          <Stack direction="row" alignItems="center" justifyContent="space-between" flexWrap="wrap" gap={2} sx={{ mb: 2 }}>
            <Typography variant="h6">{copy.chartTitle}</Typography>
            <FormControl size="small" sx={{ minWidth: 220 }}>
              <InputLabel>{copy.metricSelectLabel}</InputLabel>
              <Select
                value={selectedMetricId || metrics[0]?.metric_id || ''}
                label={copy.metricSelectLabel}
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
                  color="error"
                  variant="outlined"
                />
                <Chip size="small" label={`Unit: ${activeMetric.unit}`} variant="outlined" />
                <Chip size="small" label={`High: ${activeMetric.threshold_high}`} color="warning" variant="outlined" />
              </Stack>
              <ResponsiveContainer width="100%" height={380}>
                <LineChart data={activeMetric.data} margin={{ top: 12, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(v: string) =>
                      new Date(v).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
                    }
                    interval={23}
                    tick={{ fontSize: 11 }}
                  />
                  <YAxis domain={['auto', 'auto']} tick={{ fontSize: 11 }} />
                  <Tooltip
                    labelFormatter={(v: string) => new Date(v).toLocaleString()}
                    formatter={(val: number) => [`${Number(val).toFixed(2)} ${activeMetric.unit}`, activeMetric.metric_name]}
                  />
                  <Legend />
                  <ReferenceLine y={activeMetric.threshold_high} stroke={appConfig.theme.anomalyMarker} strokeDasharray="6 3" />
                  {activeMetric.threshold_low > 0 && (
                    <ReferenceLine y={activeMetric.threshold_low} stroke="#E65100" strokeDasharray="6 3" />
                  )}
                  <Line
                    type="monotone"
                    dataKey="value"
                    name={activeMetric.metric_name}
                    stroke={appConfig.theme.primary}
                    strokeWidth={2}
                    dot={<AnomalyDot />}
                    connectNulls
                  />
                </LineChart>
              </ResponsiveContainer>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                {copy.chartHint}
              </Typography>
            </>
          ) : (
            <Typography color="text.secondary" align="center" sx={{ py: 6 }}>No metric data available.</Typography>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}
"""


def _anomaly_feed_page() -> str:
    return """\
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Box, Card, CardContent, Typography, TextField, Select, MenuItem,
  FormControl, InputLabel, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Chip, Stack, CircularProgress, InputAdornment,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import { apiFetch } from '../api/client';
import { appConfig } from '../config/appConfig';

interface Anomaly {
  id: string;
  timestamp: string;
  metric_name: string;
  value: number;
  expected_min: number;
  expected_max: number;
  severity: string;
  status: string;
  anomaly_score: number;
}

export default function AnomalyFeedPage() {
  const copy = appConfig.pages.anomalies;
  const [search, setSearch] = useState('');
  const [severityFilter, setSeverityFilter] = useState('All');

  const { data, isLoading } = useQuery({
    queryKey: ['anomalies'],
    queryFn: () => apiFetch<{ items: Anomaly[]; total: number }>('/api/anomalies'),
    refetchInterval: appConfig.polling.anomaliesMs,
  });

  const anomalies = data?.items ?? [];
  const filtered = anomalies
    .filter((a) => severityFilter === 'All' || a.severity === severityFilter)
    .filter((a) => !search || a.metric_name.toLowerCase().includes(search.toLowerCase()));

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
        <CircularProgress color="primary" />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" color="primary" sx={{ mb: 3 }}>{copy.title}</Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent sx={{ py: 2 }}>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems="center">
            <TextField
              size="small"
              placeholder={copy.searchPlaceholder}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              sx={{ flex: 1, maxWidth: 360 }}
              InputProps={{
                startAdornment: <InputAdornment position="start"><SearchIcon fontSize="small" /></InputAdornment>,
              }}
            />
            <FormControl size="small" sx={{ minWidth: 160 }}>
              <InputLabel>{copy.severityFilterLabel}</InputLabel>
              <Select value={severityFilter} label={copy.severityFilterLabel} onChange={(e) => setSeverityFilter(e.target.value)}>
                {copy.severityOptions.map((opt) => (
                  <MenuItem key={opt} value={opt}>{opt}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <Typography variant="body2" color="text.secondary">
              {filtered.length} of {anomalies.length}
            </Typography>
          </Stack>
        </CardContent>
      </Card>

      <Card>
        <TableContainer>
          <Table size="small">
            <TableHead sx={{ bgcolor: 'primary.main' }}>
              <TableRow>
                {copy.columns.map((h) => (
                  <TableCell key={h} sx={{ color: '#fff', fontWeight: 700 }}>{h}</TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {filtered.map((row) => {
                const sev = appConfig.severityStyles[row.severity as keyof typeof appConfig.severityStyles]
                  ?? { bg: '#f5f5f5', color: '#424242' };
                const statusColor = appConfig.statusChipColor[row.status as keyof typeof appConfig.statusChipColor] ?? 'default';
                return (
                  <TableRow key={row.id} hover>
                    <TableCell sx={{ fontSize: '0.78rem', whiteSpace: 'nowrap' }}>
                      {new Date(row.timestamp).toLocaleString()}
                    </TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>{row.metric_name}</TableCell>
                    <TableCell sx={{ fontFamily: 'monospace', fontWeight: 700 }}>{Number(row.value).toFixed(2)}</TableCell>
                    <TableCell sx={{ fontSize: '0.78rem', color: 'text.secondary' }}>
                      {Number(row.expected_min).toFixed(1)} – {Number(row.expected_max).toFixed(1)}
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>{Number(row.anomaly_score).toFixed(2)}</TableCell>
                    <TableCell>
                      <Chip label={row.severity} size="small" sx={{ bgcolor: sev.bg, color: sev.color, fontWeight: 700 }} />
                    </TableCell>
                    <TableCell>
                      <Chip label={row.status} size="small" color={statusColor as 'default' | 'warning' | 'success'} />
                    </TableCell>
                  </TableRow>
                );
              })}
              {filtered.length === 0 && (
                <TableRow>
                  <TableCell colSpan={copy.columns.length} align="center" sx={{ py: 4 }}>
                    <Typography color="text.secondary">No anomalies match the current filters.</Typography>
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


def _alert_config_page() -> str:
    return """\
import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Box, Card, CardContent, Typography, Button, Switch, FormControlLabel,
  Dialog, DialogTitle, DialogContent, DialogActions, TextField, Stack,
  CircularProgress, Grid2, Divider, Chip, IconButton,
} from '@mui/material';
import EditIcon from '@mui/icons-material/Edit';
import { apiFetch } from '../api/client';
import { appConfig } from '../config/appConfig';

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

export default function AlertConfigPage() {
  const copy = appConfig.pages.alerts;
  const qc = useQueryClient();
  const [editing, setEditing] = useState<MetricConfig | null>(null);
  const [draft, setDraft] = useState({ threshold_min: 0, threshold_max: 100, zscore_threshold: 3.0 });

  const { data, isLoading } = useQuery({
    queryKey: ['thresholds'],
    queryFn: () => apiFetch<{ items: MetricConfig[] }>('/api/thresholds'),
  });

  const updateMutation = useMutation({
    mutationFn: (payload: Partial<MetricConfig> & { metric_id: string }) =>
      apiFetch(`/api/thresholds/${payload.metric_id}`, { method: 'PUT', body: JSON.stringify(payload) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['thresholds'] }),
  });

  const metrics = data?.items ?? [];

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
        <CircularProgress color="primary" />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" color="primary" sx={{ mb: 1 }}>{copy.title}</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>{copy.subtitle}</Typography>

      <Grid2 container spacing={3}>
        {metrics.map((m) => (
          <Grid2 key={m.metric_id} size={{ xs: 12, sm: 6, md: 4 }}>
            <Card sx={{ opacity: m.enabled ? 1 : 0.65 }}>
              <CardContent>
                <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
                  <Typography variant="subtitle1" fontWeight={700}>{m.metric_name}</Typography>
                  <Stack direction="row" alignItems="center" spacing={0.5}>
                    <Chip label={m.unit} size="small" variant="outlined" />
                    <IconButton size="small" color="primary" onClick={() => {
                      setDraft({
                        threshold_min: m.threshold_min,
                        threshold_max: m.threshold_max,
                        zscore_threshold: m.zscore_threshold,
                      });
                      setEditing(m);
                    }}>
                      <EditIcon fontSize="small" />
                    </IconButton>
                  </Stack>
                </Stack>
                <Divider sx={{ mb: 1.5 }} />
                <Stack spacing={0.75}>
                  <Stack direction="row" justifyContent="space-between">
                    <Typography variant="caption" color="text.secondary">Min</Typography>
                    <Typography variant="caption" fontWeight={600} fontFamily="monospace">{m.threshold_min} {m.unit}</Typography>
                  </Stack>
                  <Stack direction="row" justifyContent="space-between">
                    <Typography variant="caption" color="text.secondary">Max</Typography>
                    <Typography variant="caption" fontWeight={600} fontFamily="monospace" color="error.main">
                      {m.threshold_max} {m.unit}
                    </Typography>
                  </Stack>
                  <Stack direction="row" justifyContent="space-between">
                    <Typography variant="caption" color="text.secondary">Z-score</Typography>
                    <Typography variant="caption" fontWeight={600} fontFamily="monospace">{m.zscore_threshold}</Typography>
                  </Stack>
                </Stack>
                <Divider sx={{ mt: 1.5, mb: 1 }} />
                <FormControlLabel
                  control={<Switch checked={m.enabled} onChange={() => updateMutation.mutate({ ...m, enabled: !m.enabled })} size="small" />}
                  label={<Typography variant="caption">{m.enabled ? 'Monitoring enabled' : 'Disabled'}</Typography>}
                />
              </CardContent>
            </Card>
          </Grid2>
        ))}
      </Grid2>

      <Dialog open={Boolean(editing)} onClose={() => setEditing(null)} maxWidth="xs" fullWidth>
        <DialogTitle fontWeight={700}>Edit — {editing?.metric_name}</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField label={`Min (${editing?.unit})`} type="number" value={draft.threshold_min}
              onChange={(e) => setDraft({ ...draft, threshold_min: Number(e.target.value) })} fullWidth size="small" />
            <TextField label={`Max (${editing?.unit})`} type="number" value={draft.threshold_max}
              onChange={(e) => setDraft({ ...draft, threshold_max: Number(e.target.value) })} fullWidth size="small" />
            <TextField label="Z-score threshold" type="number" value={draft.zscore_threshold}
              inputProps={{ step: 0.1, min: 1, max: 10 }}
              onChange={(e) => setDraft({ ...draft, zscore_threshold: Number(e.target.value) })}
              fullWidth size="small" helperText={copy.zScoreHelper} />
          </Stack>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setEditing(null)} color="inherit">Cancel</Button>
          <Button variant="contained" onClick={() => {
            if (editing) updateMutation.mutate({ ...editing, ...draft });
            setEditing(null);
          }}>Save</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
"""


def _reports_page() -> str:
    return """\
import { useQuery } from '@tanstack/react-query';
import {
  Box, Card, CardContent, Typography, CircularProgress, Grid2,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip,
} from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from 'recharts';
import { apiFetch } from '../api/client';
import { appConfig } from '../config/appConfig';

interface ReportsSummary {
  total_anomalies: number;
  anomaly_rate: number;
  mttd_minutes: number;
  daily_counts: { date: string; count: number }[];
  by_severity: { name: string; value: number }[];
  top_metrics: { metric_name: string; count: number; severity: string }[];
}

function formatSummaryValue(key: string, format?: string, data?: ReportsSummary): string | number {
  if (!data) return '—';
  const raw = data[key as keyof ReportsSummary];
  if (format === 'percent' && typeof raw === 'number') return `${(raw * 100).toFixed(3)}%`;
  if (format === 'minutes' && typeof raw === 'number') return `${raw.toFixed(1)} min`;
  return typeof raw === 'number' ? raw : String(raw);
}

export default function ReportsPage() {
  const copy = appConfig.pages.reports;

  const { data, isLoading } = useQuery<ReportsSummary>({
    queryKey: ['reports/summary'],
    queryFn: () => apiFetch<ReportsSummary>('/api/reports/summary'),
  });

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
        <CircularProgress color="primary" />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" color="primary" sx={{ mb: 3 }}>{copy.title}</Typography>

      <Grid2 container spacing={3} sx={{ mb: 4 }}>
        {copy.summaryCards.map((card) => (
          <Grid2 key={card.key} size={{ xs: 12, sm: 4 }}>
            <Card>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h3" fontWeight={900} color="primary">
                  {formatSummaryValue(card.key, card.format, data)}
                </Typography>
                <Typography variant="body2" color="text.secondary">{card.label}</Typography>
              </CardContent>
            </Card>
          </Grid2>
        ))}
      </Grid2>

      <Grid2 container spacing={3} sx={{ mb: 4 }}>
        <Grid2 size={{ xs: 12, md: 8 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>{copy.dailyChartTitle}</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={data?.daily_counts ?? []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" angle={-40} textAnchor="end" tick={{ fontSize: 10 }} interval={4} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill={appConfig.theme.primary} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid2>
        <Grid2 size={{ xs: 12, md: 4 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>{copy.severityChartTitle}</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie data={data?.by_severity ?? []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label>
                    {(data?.by_severity ?? []).map((_, idx) => (
                      <Cell key={idx} fill={appConfig.piePalette[idx % appConfig.piePalette.length]} />
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

      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>{copy.topMetricsTitle}</Typography>
          <TableContainer>
            <Table size="small">
              <TableHead sx={{ bgcolor: 'primary.main' }}>
                <TableRow>
                  {(copy.tableColumns ?? []).map((h) => (
                    <TableCell key={h} sx={{ color: '#fff', fontWeight: 700 }}>{h}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {(data?.top_metrics ?? []).map((row, i) => {
                  const sev = appConfig.severityStyles[row.severity as keyof typeof appConfig.severityStyles];
                  return (
                    <TableRow key={row.metric_name} hover>
                      <TableCell sx={{ fontWeight: 700, color: 'text.secondary' }}>#{i + 1}</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>{row.metric_name}</TableCell>
                      <TableCell sx={{ fontFamily: 'monospace', fontWeight: 700 }}>{row.count}</TableCell>
                      <TableCell>
                        <Chip label={row.severity} size="small" sx={{ bgcolor: sev?.bg, color: sev?.color, fontWeight: 700 }} />
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


def _mock_ts() -> str:
    return """\
import { mockData } from '../data/mockData';

type MetricConfig = (typeof mockData.metricsConfig)[number];

let metricsState: MetricConfig[] = mockData.metricsConfig.map((m) => ({ ...m }));

function genTimeSeries(metricId: string, hours = 24) {
  const now = Date.now();
  const interval = 5 * 60 * 1000;
  const n = Math.floor((hours * 60) / 5);
  const params = mockData.timeSeriesParams[metricId as keyof typeof mockData.timeSeriesParams] ?? [50, 10, 100];
  const [base, std, spike] = params;
  let seed = metricId.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 0xffffffff; };
  const anomalySet = new Set<number>();
  for (let k = 0; k < 4; k++) anomalySet.add(Math.floor(rand() * (n - 20)) + 10);
  return Array.from({ length: n }, (_, i) => {
    const ts = new Date(now - (n - i) * interval).toISOString();
    const isAnomaly = anomalySet.has(i);
    const val = isAnomaly ? spike * (0.9 + rand() * 0.25) : Math.max(0, base + (rand() - 0.5) * 2.5 * std);
    return {
      timestamp: ts,
      value: Math.round(val * 100) / 100,
      is_anomaly: isAnomaly,
      anomaly_score: isAnomaly ? 3.5 + rand() * 2.5 : rand() * 0.7,
    };
  });
}

function genDailyCounts() {
  const now = new Date();
  let seed = 42;
  const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 0xffffffff; };
  return Array.from({ length: 30 }, (_, i) => {
    const d = new Date(now);
    d.setDate(d.getDate() - (29 - i));
    return { date: d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }), count: Math.floor(rand() * 8) + 1 };
  });
}

function ok<T>(data: T): Promise<T> { return Promise.resolve(data); }
function err(msg: string): Promise<never> { return Promise.reject(new Error(msg)); }

export async function mockFetch<T = unknown>(path: string, options: RequestInit = {}): Promise<T> {
  const method = (options.method || 'GET').toUpperCase();
  const clean = path.replace(/^[/]api[/]/, '').replace(/[?].*$/, '');
  const parts = clean.split('/').filter(Boolean);

  if (clean === 'auth/login' && method === 'POST') {
    const b = JSON.parse(String(options.body || '{}'));
    const u = mockData.users.find((x) => x.email === b.email && x.password === b.password);
    if (!u) return err('Invalid credentials') as any;
    return ok({ access_token: 'mock-jwt-' + u.id, user: { id: u.id, name: u.name, email: u.email, role: u.role } }) as T;
  }

  if (clean === 'dashboard' && method === 'GET') {
    const enabled = metricsState.filter((m) => m.enabled);
    const metrics = enabled.map((m) => ({
      ...m,
      threshold_low: m.threshold_min,
      threshold_high: m.threshold_max,
      data: genTimeSeries(m.metric_id),
    }));
    const today = new Date().toISOString().slice(0, 10);
    const todayCount = mockData.anomalies.filter((a) => a.timestamp.startsWith(today)).length || 3;
    const avgScore = mockData.anomalies.reduce((s, a) => s + a.anomaly_score, 0) / mockData.anomalies.length;
    return ok({
      total_anomalies_today: todayCount,
      active_alerts: mockData.alerts.filter((a) => a.status === 'Active').length,
      metrics_monitored: enabled.length,
      avg_anomaly_score: Math.round(avgScore * 100) / 100,
      metrics,
    }) as T;
  }

  if (clean === 'anomalies' && method === 'GET') {
    return ok({ items: mockData.anomalies, total: mockData.anomalies.length }) as T;
  }

  if (clean === 'alerts' && method === 'GET') {
    return ok({ items: mockData.alerts, total: mockData.alerts.length }) as T;
  }

  if (clean === 'thresholds' && method === 'GET') {
    return ok({ items: metricsState, total: metricsState.length }) as T;
  }

  if (parts[0] === 'thresholds' && parts.length === 2 && method === 'PUT') {
    const metricId = parts[1];
    const b = JSON.parse(String(options.body || '{}'));
    metricsState = metricsState.map((m) => (m.metric_id === metricId ? { ...m, ...b } : m));
    return ok(metricsState.find((m) => m.metric_id === metricId)) as T;
  }

  if (clean === 'reports/summary' && method === 'GET') {
    const bySeverity = ['Low', 'Medium', 'High', 'Critical']
      .map((sev) => ({ name: sev, value: mockData.anomalies.filter((a) => a.severity === sev).length }))
      .filter((s) => s.value > 0);
    return ok({
      total_anomalies: mockData.anomalies.length,
      anomaly_rate: mockData.reportsSummary.anomaly_rate,
      mttd_minutes: mockData.reportsSummary.mttd_minutes,
      daily_counts: genDailyCounts(),
      by_severity: bySeverity,
      top_metrics: mockData.reportsSummary.top_metrics,
    }) as T;
  }

  return ok({ ok: true, mocked: true, path, method }) as T;
}
"""


def _backend_main_py() -> str:
    return '''\
"""FastAPI Anomaly Detection API — data loaded from app_data.json (same bundle as frontend mock)."""
from __future__ import annotations

import datetime
import json
import os
import random
import statistics
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from anomaly_engine import detect_anomalies

_DATA_PATH = os.path.join(os.path.dirname(__file__), "app_data.json")

def _load_bundle() -> Dict[str, Any]:
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)

_BUNDLE = _load_bundle()
_USERS: List[Dict[str, Any]] = list(_BUNDLE.get("users", []))
_METRICS_CONFIG: List[Dict[str, Any]] = [dict(m) for m in _BUNDLE.get("metricsConfig", [])]
_ANOMALIES: List[Dict[str, Any]] = list(_BUNDLE.get("anomalies", []))
_ALERTS: List[Dict[str, Any]] = list(_BUNDLE.get("alerts", []))
_REPORTS = _BUNDLE.get("reportsSummary", {})
_TS_PARAMS: Dict[str, List[float]] = _BUNDLE.get("timeSeriesParams", {})

app = FastAPI(title=_BUNDLE.get("branding", {}).get("appName", "Anomaly API"), version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _generate_time_series(metric_id: str, cfg: Dict[str, Any], hours: int = 24) -> List[Dict[str, Any]]:
    now = datetime.datetime.utcnow()
    interval_min = 5
    n = (hours * 60) // interval_min
    params = _TS_PARAMS.get(metric_id, [50.0, 10.0, 100.0])
    base, std, spike = float(params[0]), float(params[1]), float(params[2])
    rng = random.Random(sum(ord(c) for c in metric_id))
    anomaly_indices = set(rng.sample(range(10, n - 10), min(4, max(1, (n - 20) // 10))))
    raw: List[float] = []
    for i in range(n):
        raw.append(
            spike * (0.9 + rng.random() * 0.25) if i in anomaly_indices
            else max(0.0, base + rng.gauss(0, std))
        )
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


class LoginRequest(BaseModel):
    email: str
    password: str


class ThresholdUpdate(BaseModel):
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    zscore_threshold: Optional[float] = None
    enabled: Optional[bool] = None


@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = next((u for u in _USERS if u["email"] == req.email and u["password"] == req.password), None)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "access_token": f"dev-token-{user['id']}",
        "user": {"id": user["id"], "name": user["name"], "email": user["email"], "role": user["role"]},
    }


@app.get("/api/dashboard")
def dashboard():
    enabled = [m for m in _METRICS_CONFIG if m.get("enabled")]
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
def list_anomalies():
    return {"items": _ANOMALIES, "total": len(_ANOMALIES)}


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
    return {
        "total_anomalies": len(_ANOMALIES),
        "anomaly_rate": _REPORTS.get("anomaly_rate", 0.003),
        "mttd_minutes": _REPORTS.get("mttd_minutes", 4.5),
        "daily_counts": daily,
        "by_severity": by_severity,
        "top_metrics": _REPORTS.get("top_metrics", []),
    }
'''


def _anomaly_engine_py() -> str:
    return '''\
"""Z-score based anomaly detection engine."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AnomalyPoint:
    index: int
    value: float
    score: float
    is_anomaly: bool


def detect_anomalies(
    values: List[float],
    threshold: float = 3.0,
    window: Optional[int] = None,
) -> List[AnomalyPoint]:
    n = len(values)
    if n == 0:
        return []
    if n < 3:
        return [AnomalyPoint(index=i, value=v, score=0.0, is_anomaly=False) for i, v in enumerate(values)]

    result: List[AnomalyPoint] = []
    if window and window > 0:
        for i, v in enumerate(values):
            lo, hi = max(0, i - window), min(n, i + window + 1)
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
            result.append(AnomalyPoint(index=i, value=v, score=round(score, 4), is_anomaly=score >= threshold))
    else:
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
                result.append(AnomalyPoint(index=i, value=v, score=round(score, 4), is_anomaly=score >= threshold))
    return result
'''
