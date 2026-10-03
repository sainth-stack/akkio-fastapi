"""
AI-driven fullstack generator — uses LLM to generate custom pages/colors from user prompt.
Works like Lovable: reads PRD + architecture → generates production MUI TypeScript code.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("app_builder")


# ──────────────────────────────────────────────────────────────────────────────
# 1. DESIGN EXTRACTION — LLM reads prompt and returns colors + style guide
# ──────────────────────────────────────────────────────────────────────────────

DESIGN_EXTRACTION_PROMPT = '''You are a senior product designer with expertise in brand identity, color theory, and design systems.

Read the user's app requirement and extract a complete, cohesive design system that matches their intent.

User Requirement:
{requirement}

PRD:
{prd}

UIUX Notes (if any):
{uiux}

Instructions:
- Use your knowledge of real brands, design systems, and color psychology
- If they reference a brand ("like Myntra", "Zomato-style") → use that brand's actual colors
- If they describe a mood ("luxury", "playful", "minimal", "dark") → choose colors that express it
- If they name colors or hex codes → use them directly
- If they describe an industry ("healthcare", "fintech", "edtech") → use industry-standard palettes
- Generate a harmonious, complete color system — not just primary color

Return a JSON object with EXACTLY this structure (no extra keys, no markdown):
{{
  "primary": "#HEX",
  "primary_dark": "#HEX",
  "primary_light": "#HEX",
  "secondary": "#HEX",
  "accent": "#HEX",
  "background": "#HEX",
  "surface": "#HEX",
  "text": "#HEX",
  "muted": "#HEX",
  "border": "#HEX",
  "danger": "#HEX",
  "success": "#HEX",
  "warning": "#HEX",
  "info": "#HEX",
  "style": "minimal|bold|dark|corporate|playful|luxury",
  "font_pair": "e.g. Inter + Playfair Display"
}}

ALWAYS return valid 6-digit hex codes (#RRGGBB). Return ONLY the JSON object.'''


PAGE_GENERATION_PROMPT = '''You are an expert React/TypeScript developer. Generate a production-quality page component.

App Title: {title}
Page: {page_name}
Description: {page_description}

Design Tokens (use these EXACT colors via theme or inline):
Primary: {primary}
Secondary: {secondary}
Accent: {accent}
Background: {background}
Surface: {surface}
Text: {text}

Architecture Context:
{architecture_context}

Mock API endpoint: {api_endpoint}

Rules:
1. Use Material UI (MUI) v6 components ONLY — Box, Card, Grid, Typography, Button, TextField, Table, Chip, Avatar, IconButton, etc.
2. Import useQuery from @tanstack/react-query for data fetching
3. Import apiFetch from '../api/client' for API calls
4. Use the theme colors via sx={{}} props with hex values directly (e.g., sx={{ color: '{primary}', bgcolor: '{background}' }})
5. Include loading states (CircularProgress) and error states
6. Make it visually impressive — use cards, proper spacing (p/m: 2-4), shadows (elevation: 3), rounded corners (borderRadius: 2)
7. For lists: use MUI Table or Grid of Cards
8. For forms: use TextField + Button in a Card
9. For dashboards: use KPI cards (4 stats at top) + charts area
10. Import icons from @mui/icons-material as needed
11. Export default function {page_name}()

Return ONLY the TypeScript React component code. No markdown, no explanation.
Start with imports, end with export default.'''


BACKEND_GENERATION_PROMPT = '''You are a senior Python/FastAPI developer. Generate production backend code.

App Title: {title}
PRD: {prd}

Database tables from architecture:
{tables_json}

Generate a complete FastAPI routes file with:
1. SQLAlchemy ORM models matching the tables
2. Pydantic schemas for request/response
3. CRUD routes: GET /api/items, POST /api/items, GET /api/items/{{id}}, PUT /api/items/{{id}}, DELETE /api/items/{{id}}
4. Authentication guard using: from auth import get_current_user
5. Proper error handling with HTTPException
6. Use database.py (already exists): from database import Base, SessionLocal, engine, get_db

Return the complete routes.py file content. Python 3.9 compatible. No markdown fences.'''


def _extract_json_from_llm(text: str) -> Optional[dict]:
    """Extract JSON object from LLM response, handling markdown fences."""
    text = text.strip()
    # Remove markdown fences
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.M)
    text = re.sub(r'```\s*$', '', text, flags=re.M)
    # Find first { ... }
    m = re.search(r'\{[\s\S]*\}', text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
            return json.loads(repair_json(m.group(0)))
        except Exception:
            return None


def _extract_code_from_llm(text: str) -> str:
    """Extract code from LLM response, removing markdown fences."""
    text = text.strip()
    # Remove ```tsx or ```typescript or ```python fences
    text = re.sub(r'^```(?:tsx?|typescript|python|jsx?|py)?\s*', '', text, flags=re.M)
    text = re.sub(r'```\s*$', '', text, flags=re.M)
    return text.strip()


async def extract_design_from_prompt(
    requirement: str,
    prd: str,
    uiux: str,
    llm=None,
) -> Dict[str, str]:
    """Use LLM to extract a full color/style design system from the user's prompt."""
    if llm is None:
        try:
            from llm_helper import get_llm_for_user
            llm = get_llm_for_user(None, temperature=0.1)
        except Exception:
            return {}

    prompt = DESIGN_EXTRACTION_PROMPT.format(
        requirement=(requirement or "")[:2000],
        prd=(prd or "")[:3000],
        uiux=(uiux or "")[:1000],
    )
    try:
        resp = await asyncio.to_thread(llm.invoke, prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        data = _extract_json_from_llm(text)
        if data and "primary" in data:
            # Validate all values are hex strings
            hex_keys = ["primary", "primary_dark", "primary_light", "secondary", "accent",
                        "background", "surface", "text", "muted", "border",
                        "danger", "success", "warning", "info"]
            defaults = {
                "primary": "#1565C0", "primary_dark": "#0D47A1", "primary_light": "#1E88E5",
                "secondary": "#7B1FA2", "accent": "#00897B", "background": "#F5F5F5",
                "surface": "#FFFFFF", "text": "#212121", "muted": "#757575",
                "border": "#E0E0E0", "danger": "#C62828", "success": "#2E7D32",
                "warning": "#F57C00", "info": "#0288D1",
            }
            result = {}
            for k in hex_keys:
                val = data.get(k, defaults.get(k, "#000000"))
                if isinstance(val, str) and re.match(r'^#[0-9a-fA-F]{3,6}$', val.strip()):
                    result[k] = val.strip()
                else:
                    result[k] = defaults.get(k, "#000000")
            result["style"] = str(data.get("style", "corporate"))
            result["font_pair"] = str(data.get("font_pair", ""))
            logger.info("[ai-design] Extracted colors: primary=%s style=%s", result["primary"], result["style"])
            return result
    except Exception as e:
        logger.warning("[ai-design] Design extraction failed: %s", e)
    return {}


async def generate_page_with_ai(
    page_name: str,
    page_description: str,
    title: str,
    colors: Dict[str, str],
    architecture: Dict[str, Any],
    llm=None,
) -> Optional[str]:
    """Use LLM to generate a custom MUI TypeScript page component."""
    if llm is None:
        try:
            from llm_helper import get_llm_for_user
            llm = get_llm_for_user(None, temperature=0.15)
        except Exception:
            return None

    # Build architecture context for this page
    tables = (architecture.get("database_schema") or {}).get("tables") or []
    table_names = [t.get("table_name") or t.get("name") or "" for t in tables if isinstance(t, dict)]
    # Guess API endpoint from page name
    api_map = {
        "dashboard": "/api/dashboard", "home": "/api/stats",
        "list": "/api/items", "products": "/api/products",
        "orders": "/api/orders", "users": "/api/users",
        "suppliers": "/api/suppliers", "reports": "/api/reports",
    }
    page_lower = page_name.lower()
    api_endpoint = next((v for k, v in api_map.items() if k in page_lower), f"/api/{page_lower.replace('page', '').lower()}")

    arch_context = f"Tables: {', '.join(table_names[:8])}" if table_names else "No specific DB tables defined."
    screens = (architecture.get("screens") or [])
    for s in screens:
        if isinstance(s, dict) and (s.get("name") or "").lower() in page_lower:
            if s.get("description"):
                arch_context += f"\nScreen spec: {s['description']}"
            if s.get("components"):
                arch_context += f"\nComponents: {', '.join(s['components'][:5])}"
            break

    prompt = PAGE_GENERATION_PROMPT.format(
        title=title,
        page_name=page_name,
        page_description=page_description,
        primary=colors.get("primary", "#1565C0"),
        secondary=colors.get("secondary", "#7B1FA2"),
        accent=colors.get("accent", "#00897B"),
        background=colors.get("background", "#F5F5F5"),
        surface=colors.get("surface", "#FFFFFF"),
        text=colors.get("text", "#212121"),
        architecture_context=arch_context,
        api_endpoint=api_endpoint,
    )

    try:
        resp = await asyncio.to_thread(llm.invoke, prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        code = _extract_code_from_llm(text)
        if code and len(code) > 200 and "export default" in code:
            logger.info("[ai-page] Generated %s (%d chars)", page_name, len(code))
            return code
    except Exception as e:
        logger.warning("[ai-page] Failed to generate %s: %s", page_name, e)
    return None


async def generate_backend_with_ai(
    title: str,
    prd: str,
    architecture: Dict[str, Any],
    llm=None,
) -> Optional[str]:
    """Use LLM to generate a custom FastAPI routes.py based on PRD + architecture."""
    if llm is None:
        try:
            from llm_helper import get_llm_for_user
            llm = get_llm_for_user(None, temperature=0.1)
        except Exception:
            return None

    tables = (architecture.get("database_schema") or {}).get("tables") or []
    tables_json = json.dumps(tables[:10], indent=2)

    prompt = BACKEND_GENERATION_PROMPT.format(
        title=title,
        prd=(prd or "")[:3000],
        tables_json=tables_json,
    )

    try:
        resp = await asyncio.to_thread(llm.invoke, prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        code = _extract_code_from_llm(text)
        if code and len(code) > 300 and ("@router" in code or "APIRouter" in code):
            logger.info("[ai-backend] Generated routes.py (%d chars)", len(code))
            return code
    except Exception as e:
        logger.warning("[ai-backend] routes.py generation failed: %s", e)
    return None


def _extract_pages_from_architecture(
    architecture: Dict[str, Any],
    prd: str,
    requirement: str,
) -> List[Tuple[str, str]]:
    """
    Extract page names + descriptions from architecture.screens or PRD headings.
    Returns list of (page_name, description).
    """
    pages: List[Tuple[str, str]] = []

    # From architecture.screens
    screens = architecture.get("screens") or []
    for s in screens:
        if isinstance(s, dict):
            name = s.get("name") or s.get("title") or ""
            desc = s.get("description") or s.get("purpose") or name
            if name and len(name) < 60:
                page_name = _to_page_component_name(name)
                pages.append((page_name, str(desc)[:400]))
        elif isinstance(s, str) and s.strip():
            page_name = _to_page_component_name(s)
            pages.append((page_name, s))

    # From PRD headings if no screens found
    if not pages:
        # Match "## Page Name" or "### Screen: ..." or "1. Login Page"
        for m in re.finditer(
            r'(?m)^(?:#{2,3}\s+|(?:\d+\.\s+))((?:[\w\s&/]+?)\s*(?:Page|Screen|View|Dashboard|Panel))',
            prd or "",
            re.I,
        ):
            name = m.group(1).strip()
            if 3 < len(name) < 60:
                page_name = _to_page_component_name(name)
                pages.append((page_name, name))

    # Deduplicate and limit
    seen = set()
    result = []
    for name, desc in pages:
        if name.lower() not in seen and name:
            seen.add(name.lower())
            result.append((name, desc))

    return result[:12]  # max 12 pages


def _to_page_component_name(name: str) -> str:
    """Convert 'Login Page' → 'LoginPage', 'Product List' → 'ProductListPage'."""
    # Remove special chars, title-case each word
    name = re.sub(r'[^a-zA-Z0-9 ]', ' ', name)
    parts = [p.capitalize() for p in name.split()]
    result = "".join(parts)
    if not result.endswith("Page") and not result.endswith("View"):
        result += "Page"
    return result


def _build_app_tsx_for_pages(pages: List[Tuple[str, str]], has_auth: bool = True) -> str:
    """Generate App.tsx routing for a list of pages."""
    imports = []
    routes = []

    # Auth always first
    if has_auth:
        imports.append("import LoginPage from './pages/LoginPage';")

    for page_name, _ in pages:
        if page_name == "LoginPage":
            continue
        imports.append(f"import {page_name} from './pages/{page_name}';")
        route_path = _page_name_to_path(page_name)
        if page_name == "DashboardPage" or page_name == "HomePage":
            routes.insert(0, f'        <Route path="{route_path}" element={{<PrivateRoute><{page_name} /></PrivateRoute>}} />')
        else:
            routes.append(f'        <Route path="{route_path}" element={{<PrivateRoute><{page_name} /></PrivateRoute>}} />')

    imports_str = "\n".join(imports)
    routes_str = "\n".join(routes)

    return f"""import {{ Navigate, Route, Routes }} from 'react-router-dom';
import AppLayout from './layout/AppLayout';
{imports_str}
import {{ isLoggedIn }} from './auth';

function PrivateRoute({{ children }}: {{ children: JSX.Element }}) {{
  return isLoggedIn() ? children : <Navigate to="/login" replace />;
}}

export default function App() {{
  return (
    <Routes>
      <Route path="/login" element={{<LoginPage />}} />
      <Route element={{<AppLayout />}}>
{routes_str}
      </Route>
      <Route path="*" element={{<Navigate to="/" replace />}} />
    </Routes>
  );
}}
"""


def _page_name_to_path(page_name: str) -> str:
    """LoginPage → /login, ProductListPage → /products, DashboardPage → /."""
    clean = re.sub(r'Page$|View$', '', page_name)
    # Convert CamelCase to kebab-case
    path = re.sub(r'(?<!^)(?=[A-Z])', '-', clean).lower()
    if path in ("dashboard", "home", "index"):
        return "/"
    return f"/{path}"


def _build_app_layout_tsx(title: str, pages: List[Tuple[str, str]], colors: Dict[str, str]) -> str:
    """Generate a sidebar layout using the extracted colors."""
    primary = colors.get("primary", "#1565C0")
    surface = colors.get("surface", "#FFFFFF")
    text_color = colors.get("text", "#212121")
    muted = colors.get("muted", "#757575")
    bg = colors.get("background", "#F5F5F5")
    safe_title = title.replace("'", "\\'").replace('"', '\\"')

    nav_items = []
    for page_name, _ in pages:
        if "login" in page_name.lower():
            continue
        path = _page_name_to_path(page_name)
        label = re.sub(r'Page$|View$', '', page_name)
        label = re.sub(r'(?<!^)(?=[A-Z])', ' ', label).strip()
        nav_items.append(f"  {{ path: '{path}', label: '{label}' }},")

    nav_str = "\n".join(nav_items)

    # Use string template to avoid brace escaping issues
    tpl = (
        "import { AppBar, Box, Drawer, List, ListItemButton, ListItemText, Toolbar, Typography, Chip } from '@mui/material';\n"
        "import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';\n"
        "import { clearAuth, getUser } from '../auth';\n\n"
        "const DRAWER = 240;\n"
        "const NAV = [\n__NAV__\n];\n\n"
        "export default function AppLayout() {\n"
        "  const location = useLocation();\n"
        "  const navigate = useNavigate();\n"
        "  const user = getUser();\n"
        "  return (\n"
        "    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: '__BG__' }}>\n"
        "      <Drawer variant=\"permanent\" sx={{ width: DRAWER, flexShrink: 0, '& .MuiDrawer-paper': { width: DRAWER, boxSizing: 'border-box', bgcolor: '__PRIMARY__', color: '#fff' } }}>\n"
        "        <Toolbar sx={{ px: 2, py: 2 }}>\n"
        "          <Typography variant=\"h6\" fontWeight={800} sx={{ color: '#fff', fontSize: '0.95rem', lineHeight: 1.3 }}>\n"
        "            __TITLE__\n"
        "          </Typography>\n"
        "        </Toolbar>\n"
        "        <List sx={{ px: 1 }}>\n"
        "          {NAV.map((item) => (\n"
        "            <ListItemButton key={item.path} component={Link} to={item.path}\n"
        "              selected={location.pathname === item.path}\n"
        "              sx={{ borderRadius: 2, mb: 0.5, '&.Mui-selected': { bgcolor: 'rgba(255,255,255,0.18)' }, '&:hover': { bgcolor: 'rgba(255,255,255,0.1)' } }}>\n"
        "              <ListItemText primary={item.label} primaryTypographyProps={{ color: '#fff', fontWeight: location.pathname === item.path ? 700 : 400, fontSize: '0.9rem' }} />\n"
        "            </ListItemButton>\n"
        "          ))}\n"
        "        </List>\n"
        "      </Drawer>\n"
        "      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>\n"
        "        <AppBar position=\"sticky\" elevation={0} sx={{ bgcolor: '__SURFACE__', color: '__TEXT__', borderBottom: '1px solid #e0e0e0', zIndex: 1 }}>\n"
        "          <Toolbar sx={{ justifyContent: 'flex-end', gap: 2 }}>\n"
        "            <Typography variant=\"body2\" color=\"__MUTED__\">{user?.name || user?.email || 'User'}</Typography>\n"
        "            <Chip label=\"Logout\" size=\"small\" onClick={() => { clearAuth(); navigate('/login'); }} sx={{ cursor: 'pointer' }} />\n"
        "          </Toolbar>\n"
        "        </AppBar>\n"
        "        <Box component=\"main\" sx={{ flex: 1, p: 3 }}>\n"
        "          <Outlet />\n"
        "        </Box>\n"
        "      </Box>\n"
        "    </Box>\n"
        "  );\n"
        "}\n"
    )
    return (tpl
            .replace("__NAV__", nav_str)
            .replace("__PRIMARY__", primary)
            .replace("__BG__", bg)
            .replace("__SURFACE__", surface)
            .replace("__TEXT__", text_color)
            .replace("__MUTED__", muted)
            .replace("__TITLE__", safe_title))


def _build_login_page_tsx(title: str, colors: Dict[str, str]) -> str:
    """Generate a beautiful login page using extracted colors."""
    primary = colors.get("primary", "#1565C0")
    primary_dark = colors.get("primary_dark", primary)
    bg = colors.get("background", "#F5F5F5")
    safe_title = title.replace("'", "\\'").replace('"', '\\"')
    gradient = f"linear-gradient(135deg, {primary} 0%, {primary_dark} 100%)"

    tpl = (
        "import { useState } from 'react';\n"
        "import { Alert, Box, Button, Card, CardContent, TextField, Typography } from '@mui/material';\n"
        "import { useNavigate } from 'react-router-dom';\n"
        "import { apiFetch } from '../api/client';\n"
        "import { setAuth } from '../auth';\n\n"
        "export default function LoginPage() {\n"
        "  const navigate = useNavigate();\n"
        "  const [email, setEmail] = useState('admin@example.com');\n"
        "  const [password, setPassword] = useState('admin123');\n"
        "  const [error, setError] = useState('');\n"
        "  const [loading, setLoading] = useState(false);\n\n"
        "  const submit = async (e: React.FormEvent) => {\n"
        "    e.preventDefault(); setError(''); setLoading(true);\n"
        "    try {\n"
        "      const res = await apiFetch<{ access_token: string; user: any }>('/api/auth/login', {\n"
        "        method: 'POST', body: JSON.stringify({ email, password }),\n"
        "      });\n"
        "      setAuth(res.access_token, res.user || { id: 1, name: email, email });\n"
        "      navigate('/');\n"
        "    } catch (err: any) { setError(err?.message || 'Invalid credentials'); }\n"
        "    finally { setLoading(false); }\n"
        "  };\n\n"
        "  return (\n"
        "    <Box sx={{ minHeight: '100vh', background: '__GRADIENT__', display: 'flex', alignItems: 'center', justifyContent: 'center', p: 2 }}>\n"
        "      <Card sx={{ maxWidth: 400, width: '100%', borderRadius: 3, boxShadow: 8 }}>\n"
        "        <CardContent sx={{ p: 4 }}>\n"
        "          <Typography variant=\"h5\" fontWeight={800} sx={{ mb: 1, color: '__PRIMARY__' }}>\n"
        "            __TITLE__\n"
        "          </Typography>\n"
        "          <Typography variant=\"body2\" color=\"text.secondary\" sx={{ mb: 3 }}>Sign in to continue</Typography>\n"
        "          {error && <Alert severity=\"error\" sx={{ mb: 2 }}>{error}</Alert>}\n"
        "          <Box component=\"form\" onSubmit={submit} sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>\n"
        "            <TextField label=\"Email\" type=\"email\" value={email} onChange={(e) => setEmail(e.target.value)} required fullWidth />\n"
        "            <TextField label=\"Password\" type=\"password\" value={password} onChange={(e) => setPassword(e.target.value)} required fullWidth />\n"
        "            <Button type=\"submit\" variant=\"contained\" size=\"large\" disabled={loading} fullWidth\n"
        "              sx={{ bgcolor: '__PRIMARY__', '&:hover': { bgcolor: '__PRIMARY_DARK__' } }}>\n"
        "              {loading ? 'Signing in…' : 'Sign In'}\n"
        "            </Button>\n"
        "          </Box>\n"
        "          <Typography variant=\"caption\" color=\"text.secondary\" sx={{ mt: 2, display: 'block', textAlign: 'center' }}>\n"
        "            Demo: admin@example.com / admin123\n"
        "          </Typography>\n"
        "        </CardContent>\n"
        "      </Card>\n"
        "    </Box>\n"
        "  );\n"
        "}\n"
    )
    return (tpl
            .replace("__GRADIENT__", gradient)
            .replace("__PRIMARY_DARK__", primary_dark)
            .replace("__PRIMARY__", primary)
            .replace("__TITLE__", safe_title)
            .replace("__BG__", bg))


def _build_mock_ts_for_pages(pages: List[Tuple[str, str]], title: str) -> str:
    """Generate a mock.ts that handles all page API calls. No f-string to avoid brace escaping."""
    lines = [
        f"// Auto-generated mock API — {title}",
        "// All API calls fall back here when backend is unreachable.",
        "const USERS = [",
        "  { id: 1, email: 'admin@example.com', password: 'admin123', name: 'Admin User', role: 'ADMIN' },",
        "  { id: 2, email: 'user@example.com', password: 'user123', name: 'Demo User', role: 'USER' },",
        "];",
        "let _id = 100;",
        "const _store: Record<string, any[]> = { items: [], records: [], entries: [] };",
        "",
        "function ok<T>(data: T): Promise<T> { return Promise.resolve(data); }",
        "function err(msg: string): Promise<never> { return Promise.reject(new Error(msg)); }",
        "",
        "export async function mockFetch<T = unknown>(path: string, options: RequestInit = {}): Promise<T> {",
        "  const method = (options.method || 'GET').toUpperCase();",
        "  const clean = path.replace(/^\\/api\\//, '').replace(/\\?.*$/, '');",
        "  const parts = clean.split('/').filter(Boolean);",
        "",
        "  // Auth",
        "  if (clean === 'auth/login' && method === 'POST') {",
        "    const b = JSON.parse(String(options.body || '{}'));",
        "    const u = USERS.find(u => u.email === b.email && u.password === b.password);",
        "    if (!u) return err('Invalid credentials') as any;",
        "    return ok({ access_token: 'mock-jwt-' + u.id, user: { id: u.id, name: u.name, email: u.email, role: u.role } }) as T;",
        "  }",
        "  if (clean === 'auth/register' && method === 'POST') {",
        "    const b = JSON.parse(String(options.body || '{}'));",
        "    USERS.push({ id: USERS.length + 1, email: b.email, password: b.password, name: b.name || b.email, role: 'USER' });",
        "    return ok({ ok: true }) as T;",
        "  }",
        "",
        "  if ((clean === 'dashboard' || clean === 'stats') && method === 'GET') {",
        "    return ok({ total: 128, active: 94, pending: 22, completed: 12,",
        "      trend: [60, 72, 80, 85, 91, 94, 100, 110, 118, 128],",
        "      recent: _store.items.slice(-5).reverse() }) as T;",
        "  }",
        "",
        "  const resource = parts[0] || 'items';",
        "  if (!_store[resource]) {",
        "    _store[resource] = Array.from({ length: 5 }, (_, i) => ({",
        "      id: i + 1, name: `${resource.replace(/s$/, '')} ${i + 1}`, status: 'Active',",
        "      created_at: new Date(Date.now() - i * 86400000).toISOString(),",
        "    }));",
        "  }",
        "",
        "  if (parts.length === 1) {",
        "    if (method === 'GET') return ok({ items: _store[resource], total: _store[resource].length }) as T;",
        "    if (method === 'POST') {",
        "      const b = JSON.parse(String(options.body || '{}'));",
        "      const item = { id: ++_id, ...b, created_at: new Date().toISOString() };",
        "      _store[resource].push(item);",
        "      return ok(item) as T;",
        "    }",
        "  }",
        "  if (parts.length === 2) {",
        "    const id = Number(parts[1]);",
        "    const idx = _store[resource].findIndex((x: any) => x.id === id);",
        "    if (method === 'GET') return idx >= 0 ? ok(_store[resource][idx]) as T : err('Not found') as any;",
        "    if (method === 'PUT' || method === 'PATCH') {",
        "      const b = JSON.parse(String(options.body || '{}'));",
        "      if (idx >= 0) _store[resource][idx] = { ..._store[resource][idx], ...b };",
        "      return ok(_store[resource][idx]) as T;",
        "    }",
        "    if (method === 'DELETE') {",
        "      if (idx >= 0) _store[resource].splice(idx, 1);",
        "      return ok({ ok: true }) as T;",
        "    }",
        "  }",
        "",
        "  return ok({ ok: true, mocked: true, path, method }) as T;",
        "}",
    ]
    return "\n".join(lines) + "\n"


async def generate_fullstack_app_with_ai(
    requirement: str,
    prd: str,
    uiux: str,
    architecture: Dict[str, Any],
    design_tokens: Optional[Dict[str, Any]] = None,
    llm=None,
) -> Dict[str, str]:
    """
    Main entry point: generates a complete fullstack app using LLM.
    1. Extracts design tokens from prompt (colors, style)
    2. Extracts page list from architecture/PRD
    3. Generates each page with LLM in parallel
    4. Generates backend routes with LLM
    5. Assembles the full file set
    """
    from app_builder.services.fullstack_codegen import get_fullstack_scaffold_files
    from app_builder.services.fullstack_frontend_generator import theme_from_tokens, build_theme_ts, _auth_ts
    from app_builder.services.fullstack_app_generator import extract_app_title
    from app_builder.services.fullstack_app_generator import generate_fullstack_application

    title = extract_app_title(requirement, prd)
    logger.info("[ai-gen] Starting AI generation | title=%s", title)

    if llm is None:
        try:
            from llm_helper import get_llm_for_user
            llm = get_llm_for_user(None, temperature=0.15)
        except Exception as e:
            logger.warning("[ai-gen] No LLM available: %s — falling back to deterministic", e)
            return generate_fullstack_application(requirement, prd, uiux, architecture, design_tokens)

    # ── Step 1: Extract design tokens with AI ────────────────────────────────
    try:
        ai_colors = await extract_design_from_prompt(requirement, prd, uiux, llm)
    except Exception as e:
        logger.warning("[ai-gen] Design extraction failed: %s", e)
        ai_colors = {}

    # Merge AI colors with existing tokens (AI takes priority for colors)
    base_colors = theme_from_tokens(design_tokens, uiux=uiux)
    if ai_colors:
        for k in ("primary", "primary_dark", "primary_light", "secondary", "accent",
                  "background", "surface", "text", "muted", "border",
                  "danger", "success", "warning", "info"):
            if ai_colors.get(k):
                base_colors[k] = ai_colors[k]

    colors = base_colors
    logger.info("[ai-gen] Colors: primary=%s bg=%s style=%s", colors.get("primary"), colors.get("background"), ai_colors.get("style", "n/a"))

    # ── Step 2: Extract pages from architecture/PRD ───────────────────────────
    pages = _extract_pages_from_architecture(architecture or {}, prd, requirement)

    # Ensure we always have login + dashboard
    page_names = [p[0] for p in pages]
    if not any("login" in n.lower() or "signin" in n.lower() for n in page_names):
        pages.insert(0, ("LoginPage", "User authentication with email and password"))
    if not any("dashboard" in n.lower() or "home" in n.lower() for n in page_names):
        pages.insert(1, ("DashboardPage", "Main dashboard with key metrics and overview"))

    logger.info("[ai-gen] Pages to generate: %s", [p[0] for p in pages])

    # ── Step 3: Generate pages in parallel ───────────────────────────────────
    async def gen_page(page_name: str, desc: str) -> Tuple[str, Optional[str]]:
        code = await generate_page_with_ai(page_name, desc, title, colors, architecture or {}, llm)
        return page_name, code

    tasks = [gen_page(name, desc) for name, desc in pages]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # ── Step 4: Generate backend routes with AI ───────────────────────────────
    backend_task = generate_backend_with_ai(title, prd, architecture or {}, llm)
    ai_routes = await backend_task

    # ── Step 5: Assemble file set ─────────────────────────────────────────────
    # Start with scaffold
    files = dict(get_fullstack_scaffold_files())

    # Core generated files
    files["frontend/src/theme.ts"] = build_theme_ts(colors)
    files["frontend/src/auth.ts"] = _auth_ts()
    files["frontend/src/App.tsx"] = _build_app_tsx_for_pages(pages)
    files["frontend/src/layout/AppLayout.tsx"] = _build_app_layout_tsx(title, pages, colors)
    files["frontend/src/api/mock.ts"] = _build_mock_ts_for_pages(pages, title)
    files["frontend/src/pages/LoginPage.tsx"] = _build_login_page_tsx(title, colors)

    # Remove old AppShell if we have AppLayout
    files.pop("frontend/src/layout/AppShell.tsx", None)

    # AI-generated page files
    successful_pages = 0
    failed_pages = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("[ai-gen] Page task exception: %s", result)
            continue
        page_name, code = result
        if page_name == "LoginPage":
            continue  # Already generated above
        if code:
            files[f"frontend/src/pages/{page_name}.tsx"] = code
            successful_pages += 1
        else:
            failed_pages.append(page_name)
            logger.warning("[ai-gen] Failed to generate %s — will use fallback", page_name)

    # Fallback for failed pages
    if failed_pages:
        _add_fallback_pages(files, failed_pages, colors, architecture or {})

    # AI-generated backend
    if ai_routes:
        files["backend/routes.py"] = ai_routes
    else:
        # Fallback to deterministic backend
        logger.warning("[ai-gen] AI backend failed — using deterministic backend")
        _add_fallback_backend(files, title, architecture or {})

    logger.info(
        "[ai-gen] Complete | pages=%d/%d AI-generated | total_files=%d",
        successful_pages, len(pages), len(files),
    )

    # Clean up stale files
    files.pop("frontend/src/App.jsx", None)
    files.pop("frontend/src/styles/app.css", None)
    return files


def _add_fallback_pages(
    files: Dict[str, str],
    failed_pages: List[str],
    colors: Dict[str, str],
    architecture: Dict[str, Any],
) -> None:
    """Generate simple fallback pages for any AI generation failures."""
    primary = colors.get("primary", "#1565C0")
    bg = colors.get("background", "#F5F5F5")
    surface = colors.get("surface", "#FFFFFF")

    for page_name in failed_pages:
        label = re.sub(r'Page$|View$', '', page_name)
        label = re.sub(r'(?<!^)(?=[A-Z])', ' ', label).strip()
        api_path = _page_name_to_path(page_name).strip("/") or "items"

        files[f"frontend/src/pages/{page_name}.tsx"] = f"""import {{ useState }} from 'react';
import {{ useQuery, useMutation, useQueryClient }} from '@tanstack/react-query';
import {{ Box, Button, Card, CardContent, CircularProgress, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Typography, Chip }} from '@mui/material';
import {{ apiFetch }} from '../api/client';

type Item = Record<string, any>;

export default function {page_name}() {{
  const qc = useQueryClient();
  const {{ data, isLoading }} = useQuery({{
    queryKey: ['{api_path}'],
    queryFn: () => apiFetch<{{ items: Item[] }}>('/api/{api_path}'),
  }});
  const items = data?.items || [];

  if (isLoading) return <Box sx={{{{ display: 'flex', justifyContent: 'center', py: 8 }}}}><CircularProgress sx={{{{ color: '{primary}' }}}} /></Box>;

  return (
    <Box sx={{{{ p: 3, bgcolor: '{bg}', minHeight: '100vh' }}}}>
      <Typography variant="h4" fontWeight={{800}} sx={{{{ mb: 3, color: '{primary}' }}}}>
        {label}
      </Typography>
      <Card sx={{{{ boxShadow: 3, borderRadius: 2 }}}}>
        <CardContent sx={{{{ p: 0 }}}}>
          <TableContainer>
            <Table>
              <TableHead sx={{{{ bgcolor: '{primary}' }}}}>
                <TableRow>
                  {{Object.keys(items[0] || {{ id: '', name: '', status: '' }}).map(k => (
                    <TableCell key={{k}} sx={{{{ color: '#fff', fontWeight: 700 }}}}>{{k.toUpperCase()}}</TableCell>
                  ))}}
                </TableRow>
              </TableHead>
              <TableBody>
                {{items.map((row, i) => (
                  <TableRow key={{i}} hover>
                    {{Object.values(row).map((v, j) => (
                      <TableCell key={{j}}>
                        {{typeof v === 'boolean' ? <Chip label={{v ? 'Yes' : 'No'}} size="small" color={{v ? 'success' : 'default'}} />
                          : String(v ?? '—')}}
                      </TableCell>
                    ))}}
                  </TableRow>
                ))}}
              </TableBody>
            </Table>
          </TableContainer>
          {{items.length === 0 && (
            <Box sx={{{{ p: 4, textAlign: 'center' }}}}>
              <Typography color="text.secondary">No {label.lower()} found.</Typography>
            </Box>
          )}}
        </CardContent>
      </Card>
    </Box>
  );
}}
"""


def _add_fallback_backend(files: Dict[str, str], title: str, architecture: Dict[str, Any]) -> None:
    """Add simple deterministic backend if AI backend generation fails."""
    tables = (architecture.get("database_schema") or {}).get("tables") or []
    table_names = [
        (t.get("table_name") or t.get("name") or "items").lower()
        for t in tables if isinstance(t, dict)
    ] or ["items"]

    routes_code = f'''from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db

router = APIRouter()

@router.post("/api/auth/login")
def login(payload: Dict[str, Any]):
    email = payload.get("email", "")
    password = payload.get("password", "")
    if email and password:
        return {{"access_token": "dev-token-" + email[:8], "user": {{"id": 1, "email": email, "name": email.split("@")[0]}}}}
    raise HTTPException(401, "Invalid credentials")

@router.get("/api/dashboard")
def dashboard():
    return {{"total": 128, "active": 94, "pending": 22, "completed": 12}}
'''

    for table in table_names[:5]:
        safe = table.replace("-", "_")
        routes_code += f'''
@router.get("/api/{table}")
def list_{safe}(db: Session = Depends(get_db)):
    return {{"items": [], "total": 0}}

@router.post("/api/{table}")
def create_{safe}(payload: Dict[str, Any], db: Session = Depends(get_db)):
    return {{"id": 1, **payload}}

@router.get("/api/{table}/{{item_id}}")
def get_{safe}(item_id: int, db: Session = Depends(get_db)):
    return {{"id": item_id}}
'''

    files["backend/routes.py"] = routes_code
