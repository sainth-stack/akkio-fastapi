"""JSON-only app generation: template + app-definition.json (no LLM TSX)."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, Optional

from app_builder.schemas.app_definition import AppDefinition
from app_builder.services.app_definition_compiler import compile_app_definition

logger = logging.getLogger("app_builder")

TEMPLATE_NAME = "base-frontend-vite-mui"
_TEMPLATES_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "templates",
)
_TEMPLATE_DIR = os.path.join(_TEMPLATES_BASE, TEMPLATE_NAME)


def load_template_frontend_files() -> Dict[str, str]:
    if not os.path.isdir(_TEMPLATE_DIR):
        logger.error("[json_app] template missing: %s", _TEMPLATE_DIR)
        return {}
    files: Dict[str, str] = {}
    exclude = {"__pycache__", "node_modules", ".git", "dist", "build"}
    for root, dirs, filenames in os.walk(_TEMPLATE_DIR):
        dirs[:] = [d for d in dirs if d not in exclude]
        rel_root = os.path.relpath(root, _TEMPLATE_DIR)
        for fn in filenames:
            if fn.endswith(".pyc") or fn == "base.json":
                continue
            rel = os.path.join(rel_root, fn) if rel_root != "." else fn
            rel = rel.replace("\\", "/")
            if not rel.startswith("frontend/"):
                continue
            try:
                with open(os.path.join(root, fn), encoding="utf-8") as fh:
                    files[rel] = fh.read()
            except OSError:
                pass
    logger.info("[json_app] loaded template | files=%d", len(files))
    return files


def _tokens_ts_from_theme(theme: Any) -> str:
    t = theme.model_dump() if hasattr(theme, "model_dump") else theme
    palette = t.get("chartPalette") or []
    font = (t.get("fontFamily") or "Inter").replace("'", "\\'")
    return f"""/**
 * Design tokens — generated from AppDefinition.theme (edit app-definition.json to restyle).
 */
export const tokens = {{
  primary:        '{t.get("primary", "#1976d2")}',
  primaryDark:    '{t.get("primaryDark", "#1565c0")}',
  primaryLight:   '{t.get("primaryLight", "#42a5f5")}',
  secondary:      '{t.get("secondary", "#9c27b0")}',
  accent:         '#00bcd4',
  background:     '{t.get("background", "#f5f7fb")}',
  surface:        '{t.get("surface", "#ffffff")}',
  text:           '{t.get("text", "#0f172a")}',
  muted:          '{t.get("muted", "#64748b")}',
  border:         '#e2e8f0',
  danger:         '{t.get("danger", "#d32f2f")}',
  success:        '{t.get("success", "#2e7d32")}',
  warning:        '{t.get("warning", "#ed6c02")}',
  info:           '{t.get("info", "#0288d1")}',
  fontFamily:     "'{font}', system-ui, -apple-system, sans-serif",
  chartPalette:   {json.dumps(palette)},
}} as const;

export type Tokens = typeof tokens;
"""


def generate_json_app_files(
    *,
    requirement: str = "",
    prd: str = "",
    uiux: str = "",
    plan_json: Optional[Dict[str, Any]] = None,
    app_name: Optional[str] = None,
    app_definition: Optional[AppDefinition] = None,
) -> Dict[str, str]:
    """Return full frontend file map: frozen template + app-definition.json + tokens."""
    definition = app_definition or compile_app_definition(
        requirement=requirement,
        prd=prd,
        uiux=uiux,
        plan_json=plan_json,
        app_name=app_name,
    )

    files = load_template_frontend_files()
    files.update(_json_renderer_sources())

    files["frontend/src/config/app-definition.json"] = json.dumps(
        definition.to_json_dict(), indent=2, ensure_ascii=False
    )
    files["frontend/src/theme/tokens.ts"] = _tokens_ts_from_theme(definition.theme)

    if "frontend/tsconfig.json" in files:
        try:
            tc = json.loads(files["frontend/tsconfig.json"])
            tc.setdefault("compilerOptions", {})["resolveJsonModule"] = True
            files["frontend/tsconfig.json"] = json.dumps(tc, indent=2) + "\n"
        except json.JSONDecodeError:
            pass

    # Remove demo-only page from route (renderer handles all routes)
    files.pop("frontend/src/pages/DemoPage.tsx", None)

    logger.info(
        "[json_app] definition | pages=%d nav=%d mock_keys=%d",
        len(definition.pages),
        len(definition.navigation),
        len(definition.mock),
    )
    return files


async def run_json_app_codegen(
    *,
    requirement: str,
    plan_json: Optional[Dict[str, Any]],
    project_name: str,
    on_event: Optional[Callable] = None,
    prd: str = "",
    uiux: str = "",
) -> Dict[str, str]:
    """Async entry for codegen WebSocket — emits progress, returns files."""
    async def emit(payload: Dict[str, Any]) -> None:
        if on_event:
            await on_event(payload)

    await emit({
        "event": "agent_start",
        "agent": "app_definition_agent",
        "message": "Compiling app definition from your plan…",
    })

    prd_text = prd
    uiux_text = uiux
    if plan_json:
        if isinstance(plan_json.get("prd"), dict):
            prd_text = prd_text or json.dumps(plan_json["prd"])
        elif isinstance(plan_json.get("prd"), str):
            prd_text = prd_text or plan_json["prd"]
        if isinstance(plan_json.get("uiux"), dict):
            uiux_text = uiux_text or json.dumps(plan_json["uiux"])
        elif isinstance(plan_json.get("uiux"), str):
            uiux_text = uiux_text or plan_json["uiux"]

    definition = compile_app_definition(
        requirement=requirement,
        prd=prd_text,
        uiux=uiux_text,
        plan_json=plan_json,
    )

    await emit({
        "event": "agent_complete",
        "agent": "app_definition_agent",
        "message": f"App definition ready — {len(definition.pages)} pages, {len(definition.navigation)} nav items",
    })
    await emit({
        "event": "app_definition_json",
        "data": definition.to_json_dict(),
    })

    await emit({
        "event": "agent_start",
        "agent": "template_agent",
        "message": f"Loading template {TEMPLATE_NAME} and binding JSON to UI…",
    })

    files = generate_json_app_files(
        requirement=requirement,
        prd=prd_text,
        uiux=uiux_text,
        plan_json=plan_json,
        app_definition=definition,
    )

    await emit({
        "event": "agent_complete",
        "agent": "template_agent",
        "message": f"Template ready ({len(files)} files, 1 app-definition.json)",
    })

    return files


def _json_renderer_sources() -> Dict[str, str]:
    """Frozen JSON renderer sources (same for every app)."""
    return {
        "frontend/src/App.tsx": _APP_TSX,
        "frontend/src/renderer/PageRenderer.tsx": _PAGE_RENDERER,
        "frontend/src/renderer/useAppDefinition.ts": _USE_DEFINITION,
        "frontend/src/renderer/BlockRenderer.tsx": _BLOCK_RENDERER,
    }


_APP_TSX = """\
import { Navigate, Route, Routes } from 'react-router-dom';
import AppShell from './components/ui/AppShell';
import PageRenderer from './renderer/PageRenderer';
import { useAppDefinition } from './renderer/useAppDefinition';

export default function App() {
  const { definition, loading, error } = useAppDefinition();

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
  );
}
"""

_USE_DEFINITION = """\
import { useMemo } from 'react';
import definitionJson from '../config/app-definition.json';

export interface AppDefinition {
  version: number;
  meta: { appName: string; description: string; domain: string };
  theme: Record<string, unknown>;
  navigation: Array<{ label: string; path: string; icon: string; pageId: string }>;
  pages: Array<{ id: string; title: string; subtitle?: string; blocks: Array<Record<string, unknown>> }>;
  mock: Record<string, unknown>;
}

export function useAppDefinition() {
  const definition = useMemo(() => definitionJson as AppDefinition, []);
  return { definition, loading: false, error: null as string | null };
}

export function getMockSlice(definition: AppDefinition, key: string): unknown {
  return definition.mock?.[key];
}
"""

_PAGE_RENDERER = """\
import { Box, Typography } from '@mui/material';
import BlockRenderer from './BlockRenderer';
import { useAppDefinition } from './useAppDefinition';

export default function PageRenderer({ pageId }: { pageId: string }) {
  const { definition } = useAppDefinition();
  const page = definition.pages.find((p) => p.id === pageId);

  if (!page) {
    return <Typography color="error">Unknown page: {pageId}</Typography>;
  }

  return (
    <Box>
      <Typography variant="h4" fontWeight={700} gutterBottom>
        {page.title}
      </Typography>
      {page.subtitle ? (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          {page.subtitle}
        </Typography>
      ) : null}
      {page.blocks.map((block, i) => (
        <BlockRenderer key={`${pageId}-${i}`} block={block} definition={definition} />
      ))}
    </Box>
  );
}
"""

_BLOCK_RENDERER = """\
import { Box, Stack, Typography } from '@mui/material';
import {
  BarChart,
  DataTable,
  LineChart,
  PageHeader,
  PieChart,
  StatCard,
} from '../components/ui';
import type { AppDefinition } from './useAppDefinition';
import { getMockSlice } from './useAppDefinition';
import { tokens } from '../theme/tokens';

type Block = Record<string, unknown>;

function kpiValues(definition: AppDefinition, block: Block): Record<string, string | number> {
  const keys = (block.items as Array<{ valueKey: string }> | undefined)?.map((i) => i.valueKey) ?? [];
  const merged: Record<string, string | number> = {};
  for (const [k, v] of Object.entries(definition.mock || {})) {
    if (k.startsWith('kpis_') && v && typeof v === 'object' && !Array.isArray(v)) {
      Object.assign(merged, v as Record<string, string | number>);
    }
  }
  const out: Record<string, string | number> = {};
  for (const k of keys) {
    out[k] = merged[k] ?? '—';
  }
  return out;
}

export default function BlockRenderer({
  block,
  definition,
}: {
  block: Block;
  definition: AppDefinition;
}) {
  const type = block.type as string;

  if (type === 'kpiRow') {
    const items = (block.items as Array<{ label: string; valueKey: string; color?: string }>) || [];
    const values = kpiValues(definition, block);
    return (
      <Stack direction="row" flexWrap="wrap" gap={2} sx={{ mb: 3 }}>
        {items.map((item) => (
          <Box key={item.valueKey} sx={{ flex: '1 1 220px', minWidth: 200, maxWidth: 320 }}>
            <StatCard
              label={item.label}
              value={values[item.valueKey] ?? '—'}
              color={(item.color as 'primary') || 'primary'}
            />
          </Box>
        ))}
      </Stack>
    );
  }

  if (type === 'lineChart') {
    const data = (getMockSlice(definition, block.dataKey as string) as Record<string, unknown>[]) || [];
    const series = (block.series as Array<{ dataKey: string; name?: string; color?: string }>) || [];
    return (
      <Box sx={{ mb: 3 }}>
        {block.title ? (
          <Typography variant="h6" fontWeight={600} sx={{ mb: 1 }}>{block.title as string}</Typography>
        ) : null}
        <LineChart
          data={data}
          series={series.map((s, i) => ({
            dataKey: s.dataKey,
            name: s.name || s.dataKey,
            color: s.color || tokens.chartPalette[i % tokens.chartPalette.length],
          }))}
          xAxisKey={(block.xAxisKey as string) || 'name'}
          height={(block.height as number) || 280}
        />
      </Box>
    );
  }

  if (type === 'barChart') {
    const data = (getMockSlice(definition, block.dataKey as string) as Record<string, unknown>[]) || [];
    const series = (block.series as Array<{ dataKey: string; name?: string; color?: string }>) || [];
    return (
      <Box sx={{ mb: 3 }}>
        {block.title ? (
          <Typography variant="h6" fontWeight={600} sx={{ mb: 1 }}>{block.title as string}</Typography>
        ) : null}
        <BarChart
          data={data}
          series={series.map((s, i) => ({
            dataKey: s.dataKey,
            name: s.name || s.dataKey,
            color: s.color || tokens.chartPalette[i % tokens.chartPalette.length],
          }))}
          xAxisKey={(block.xAxisKey as string) || 'name'}
          height={(block.height as number) || 280}
        />
      </Box>
    );
  }

  if (type === 'pieChart') {
    const data = (getMockSlice(definition, block.dataKey as string) as Array<{ name: string; value: number }>) || [];
    return (
      <Box sx={{ mb: 3 }}>
        {block.title ? (
          <Typography variant="h6" fontWeight={600} sx={{ mb: 1 }}>{block.title as string}</Typography>
        ) : null}
        <PieChart data={data} height={(block.height as number) || 280} />
      </Box>
    );
  }

  if (type === 'dataTable') {
    const rows = (getMockSlice(definition, block.dataKey as string) as Record<string, unknown>[]) || [];
    const columns = (block.columns as Array<{ field: string; header: string; align?: string }>) || [];
    return (
      <Box sx={{ mb: 3 }}>
        {block.title ? <PageHeader title={block.title as string} /> : null}
        <DataTable
          columns={columns.map((c) => ({
            field: c.field as keyof Record<string, unknown>,
            header: c.header,
            align: c.align as 'left' | 'center' | 'right' | undefined,
          }))}
          rows={rows}
          rowKey="id"
          searchable={Boolean(block.searchable)}
          emptyMessage={(block.emptyMessage as string) || 'No records found'}
        />
      </Box>
    );
  }

  if (type === 'text') {
    return (
      <Typography variant="body1" sx={{ mb: 2 }}>{block.content as string}</Typography>
    );
  }

  return null;
}
"""
