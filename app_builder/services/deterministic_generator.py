"""
Deterministic generator — produces App.tsx, router, nav config, and page stubs
from a Blueprint dict.  Zero LLM calls; pure string construction.

Blueprint schema (Python dict):
    app_name:         str  — e.g. "CRM Dashboard"
    description:      str  — one-paragraph description
    pages:            list[PageSpec]
    primary_color:    str  — hex e.g. "#1976d2"
    background_color: str  — hex e.g. "#f5f7fb"
    style:            str  — "minimal" | "bold" | "dark" | "corporate"
    domain:           str  — classified domain
    font_family:      str  — e.g. "Inter"

PageSpec:
    name:        str  — component name e.g. "DashboardPage"
    path:        str  — router path   e.g. "/"
    nav_label:   str  — sidebar label
    icon:        str  — MUI icon name e.g. "Dashboard"
    description: str  — short description (used in page placeholder)
"""
from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# App.tsx — BrowserRouter-level routes wired to AppShell
# ---------------------------------------------------------------------------

def _build_app_tsx(pages: List[Dict[str, Any]], app_name: str) -> str:
    imports = []
    route_lines = []
    for page in pages:
        name = page["name"]
        path = page["path"]
        imports.append(f"import {name} from './pages/{name}';")
        route_lines.append(f"        <Route path=\"{path}\" element={{<{name} />}} />")

    nav_items_json = json.dumps(
        [{"label": p["nav_label"], "path": p["path"], "icon": p["icon"]} for p in pages],
        indent=2,
    )
    imports_str = "\n".join(imports)
    routes_str = "\n".join(route_lines)

    return textwrap.dedent(f"""\
        import {{ Routes, Route }} from 'react-router-dom';
        import AppShell from './components/ui/AppShell';
        {imports_str}

        const NAV_ITEMS = {nav_items_json};

        export default function App() {{
          return (
            <Routes>
              <Route element={{<AppShell navItems={{NAV_ITEMS}} appName={{"{app_name}"}} />}}>
        {routes_str}
              </Route>
            </Routes>
          );
        }}
        """)


# ---------------------------------------------------------------------------
# theme/tokens.ts — design tokens from blueprint
# ---------------------------------------------------------------------------

def _build_tokens_ts(blueprint: Dict[str, Any]) -> str:
    primary = blueprint.get("primary_color", "#1976d2")
    bg = blueprint.get("background_color", "#f5f7fb")
    font = blueprint.get("font_family", "Inter")

    return textwrap.dedent(f"""\
        /**
         * Design tokens derived from the app blueprint.
         * Edit these to restyle the entire application.
         */
        export const tokens = {{
          primary:        '{primary}',
          primaryDark:    '{_darken(primary)}',
          primaryLight:   '{_lighten(primary)}',
          secondary:      '#9c27b0',
          accent:         '#00bcd4',
          background:     '{bg}',
          surface:        '#ffffff',
          text:           '#0f172a',
          muted:          '#64748b',
          border:         '#e2e8f0',
          danger:         '#d32f2f',
          success:        '#2e7d32',
          warning:        '#ed6c02',
          info:           '#0288d1',
          fontFamily:     "'{font}', system-ui, -apple-system, sans-serif",
        }} as const;

        export type Tokens = typeof tokens;
        """)


def _darken(hex_color: str) -> str:
    """Naive hex darkener — shifts each channel down by 15."""
    try:
        h = hex_color.lstrip("#")
        if len(h) != 6:
            return hex_color
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r, g, b = max(0, r - 30), max(0, g - 30), max(0, b - 30)
        return f"#{r:02x}{g:02x}{b:02x}"
    except ValueError:
        return hex_color


def _lighten(hex_color: str) -> str:
    """Naive hex lightener — shifts each channel up by 40."""
    try:
        h = hex_color.lstrip("#")
        if len(h) != 6:
            return hex_color
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r, g, b = min(255, r + 40), min(255, g + 40), min(255, b + 40)
        return f"#{r:02x}{g:02x}{b:02x}"
    except ValueError:
        return hex_color


# ---------------------------------------------------------------------------
# Page stubs — empty but type-correct TypeScript components
# ---------------------------------------------------------------------------

def _build_page_stub(page: Dict[str, Any]) -> str:
    name = page["name"]
    label = page.get("nav_label", name.replace("Page", ""))
    description = page.get("description", f"{label} management")
    icon = page.get("icon", "Article")

    return textwrap.dedent(f"""\
        import {{ Box, Typography }} from '@mui/material';
        import {{ {icon} as {icon}Icon }} from '@mui/icons-material';
        import {{ PageHeader, EmptyState }} from '../components/ui';

        export default function {name}() {{
          return (
            <Box>
              <PageHeader title="{label}" subtitle="{description}" />
              <EmptyState
                icon={{<{icon}Icon sx={{{{ fontSize: 64, color: 'text.disabled' }}}} />}}
                title="Coming Soon"
                description="This section is under construction. Check back soon!"
              />
            </Box>
          );
        }}
        """)


# ---------------------------------------------------------------------------
# mock/index.ts — typed mock data barrel
# ---------------------------------------------------------------------------

def _build_mock_index(blueprint: Dict[str, Any]) -> str:
    domain = blueprint.get("domain", "generic")
    app_name = blueprint.get("app_name", "App")
    pages = blueprint.get("pages", [])

    entity_names = []
    for p in pages:
        n = p.get("name", "").replace("Page", "")
        if n and n not in ("Dashboard", "Reports", "Settings", "Main"):
            entity_names.append(n)

    if not entity_names:
        entity_names = ["Item"]

    entries = []
    for entity in entity_names[:3]:
        singular = entity.rstrip("s")
        lower = entity.lower()
        entries.append(f"""\
export const mock{entity}: {singular}[] = [
  {{ id: '1', name: 'Sample {singular} A', status: 'active', createdAt: '2026-01-10' }},
  {{ id: '2', name: 'Sample {singular} B', status: 'inactive', createdAt: '2026-02-15' }},
  {{ id: '3', name: 'Sample {singular} C', status: 'active', createdAt: '2026-03-20' }},
];

export interface {singular} {{
  id: string;
  name: string;
  status: string;
  createdAt: string;
  [key: string]: unknown;
}}""")

    entries_str = "\n\n".join(entries)
    return f"""\
/**
 * Mock data for {app_name} ({domain} domain).
 * Replace with real API calls when your backend is ready.
 */

{entries_str}

export const mockKpis = [
  {{ label: 'Total Items', value: '1,234', trend: {{ value: 12, label: 'vs last month' }} }},
  {{ label: 'Active', value: '987', trend: {{ value: 5, label: 'vs last month' }} }},
  {{ label: 'Pending', value: '247', trend: {{ value: -3, label: 'vs last month' }} }},
  {{ label: 'Completed', value: '456', trend: {{ value: 8, label: 'vs last month' }} }},
];

export const mockChartData = Array.from({{ length: 12 }}, (_, i) => ({{
  name: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][i],
  value: Math.floor(Math.random() * 400 + 100),
  previous: Math.floor(Math.random() * 400 + 80),
}}));
"""


# ---------------------------------------------------------------------------
# package.json — pinned versions, frontend-only (no proxy)
# ---------------------------------------------------------------------------

PINNED_PACKAGE_JSON: Dict[str, Any] = {
    "name": "frontend",
    "version": "1.0.0",
    "private": True,
    "type": "module",
    "scripts": {
        "dev": "vite",
        "build": "vite build",
        "preview": "vite preview",
        "typecheck": "tsc --noEmit",
    },
    "dependencies": {
        "@emotion/react": "11.13.5",
        "@emotion/styled": "11.13.5",
        "@mui/icons-material": "6.1.9",
        "@mui/material": "6.1.9",
        "@tanstack/react-query": "5.62.7",
        "react": "18.3.1",
        "react-dom": "18.3.1",
        "react-router-dom": "6.28.0",
        "recharts": "2.15.0",
    },
    "devDependencies": {
        "@types/react": "18.3.12",
        "@types/react-dom": "18.3.1",
        "@vitejs/plugin-react": "4.3.4",
        "typescript": "5.6.3",
        "vite": "5.4.11",
    },
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_from_blueprint(blueprint: Dict[str, Any]) -> Dict[str, str]:
    """
    Deterministically generate project files from a blueprint dict.

    Returns a Dict[str, str] with file paths (relative to project root) as keys.
    Only writes files that are NOT frozen in the template (App.tsx, pages/*, mock/*,
    theme/tokens.ts, package.json) — template infrastructure files are untouched.

    Args:
        blueprint: Dict matching the Blueprint schema described in this module's
                   docstring.  All keys are optional; safe defaults are used.
    """
    pages: List[Dict[str, Any]] = blueprint.get("pages") or []
    app_name: str = blueprint.get("app_name") or "Application"

    files: Dict[str, str] = {}

    # 1. App.tsx — route tree
    if pages:
        files["frontend/src/App.tsx"] = _build_app_tsx(pages, app_name)

    # 2. Design tokens
    files["frontend/src/theme/tokens.ts"] = _build_tokens_ts(blueprint)

    # 3. Page stubs (only for pages that don't already exist in template)
    for page in pages:
        path = f"frontend/src/pages/{page['name']}.tsx"
        files[path] = _build_page_stub(page)

    # 4. Mock data
    files["frontend/src/mock/index.ts"] = _build_mock_index(blueprint)

    # 5. package.json — update app name
    pkg = dict(PINNED_PACKAGE_JSON)
    safe_name = app_name.lower().replace(" ", "-").replace("_", "-")[:50]
    pkg["name"] = safe_name or "frontend"
    files["frontend/package.json"] = json.dumps(pkg, indent=2) + "\n"

    return files
