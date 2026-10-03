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


PAGE_GENERATION_PROMPT = '''You are a senior React/TypeScript engineer at a product company like Lovable or Vercel.
Generate a complete, production-quality page that feels handcrafted — not generic.

═══ APP CONTEXT ═══
App: {title}
What this app does: {requirement}

Key product requirements:
{prd_summary}

═══ PAGE TO BUILD ═══
Page: {page_name}
Purpose: {page_description}

Screen spec from architecture:
{screen_spec}

Related API endpoints for this page:
{api_endpoints}

Other pages in the app (for navigation links): {all_pages}

═══ DESIGN SYSTEM ═══
Primary: {primary}
Primary Dark: {primary_dark}
Secondary: {secondary}
Accent: {accent}
Background: {background}
Surface: {surface}
Text: {text}
Muted: {muted}
Style: {style}

═══ TECH RULES ═══
- MUI v6: Box, Card, Grid2, Stack, Typography, Button, TextField, Table, Chip, Avatar, IconButton, Fab, Drawer, Dialog, etc.
- Data fetching: useQuery from @tanstack/react-query  
- API calls: `import { apiFetch } from '../api/client'`  ← NAMED import, NOT default
- Routing: useNavigate from react-router-dom (for links to other pages)
- Colors: use sx={{}} with the hex values above directly
- Loading: <CircularProgress sx={{{{ color: '{primary}' }}}} />
- Error: friendly error Card with retry button

═══ QUALITY REQUIREMENTS ═══
- Build what the page ACTUALLY needs — not a generic table
- For dashboard: KPI cards with real metrics + activity feed or chart area
- For forms: complete form with validation, proper fields, submit handler
- For lists: filtering, search, add/edit modal or inline actions
- For e-commerce: product cards with images, prices, cart buttons
- For auth: clean login/register form matching the brand theme
- Use real fake data in mock states (realistic names, numbers, statuses)
- Responsive layout using Grid2 or Stack
- Typography hierarchy: h4 for title, h6 for sections, body2 for labels
- At least 60-80 lines of JSX — no stub components

Export default function {page_name}(). No markdown, no explanation, just the code.'''


LAYOUT_GENERATION_PROMPT = '''You are a senior React/TypeScript engineer. Generate the main app layout with sidebar navigation.

App: {title}
User requirement: {requirement}
Style: {style}

Pages to show in sidebar:
{pages_list}

Design system:
Primary: {primary} (sidebar background)
Primary Dark: {primary_dark}
Surface: {surface} (content area background)
Text on primary: #FFFFFF
Muted: {muted}
Background: {background}

Rules:
- Sidebar: fixed left, 240px wide, bg={primary}, white text and icons
- Top bar: app title + user avatar/menu
- Content area: bg={background}, right of sidebar, fills screen
- Use Drawer (permanent variant) for sidebar
- Icons from @mui/icons-material — pick appropriate icons per page name
- Active route highlighted with slightly lighter bg
- useNavigate + useLocation for routing and active state
- Export default function AppLayout() with <Outlet /> for content

Structure:
```
<Box sx={{ display: 'flex' }}>
  <Drawer permanent> ... nav items ... </Drawer>
  <Box component="main" sx={{ flexGrow: 1 }}>
    <TopBar />
    <Outlet />
  </Box>
</Box>
```

No markdown, no explanation. Return ONLY the TypeScript code.'''


LOGIN_PAGE_PROMPT = '''You are a senior React/TypeScript engineer. Generate a stunning login/auth page.

App: {title}
Requirement: {requirement}
Style: {style}

Design:
Primary: {primary}
Primary Dark: {primary_dark}
Background: {background}
Surface: {surface}
Text: {text}

Requirements:
- Left half: brand panel with gradient (primary → primary_dark), app name, tagline matching the requirement
- Right half: login form — email + password + "Sign In" button using primary color
- "Register" link below the form
- Show/hide password toggle (VisibilityOff icon)
- Form state with useState, fake login: store token in localStorage, navigate to /
- Clean, professional. Inspired by Figma/Notion/Linear login pages
- useNavigate from react-router-dom for post-login redirect
- `import {{ apiFetch }} from '../api/client'` — named import (NOT default import)

No markdown. Return ONLY the TypeScript code. Export default function LoginPage().'''


BACKEND_GENERATION_PROMPT = '''You are a senior Python/FastAPI developer. Generate production-ready backend code.

App: {title}
Domain: {requirement}
PRD: {prd}

Database tables from architecture:
{tables_json}

Generate a complete FastAPI routes.py with:
1. SQLAlchemy ORM models matching the tables with __tablename__, columns, relationships
2. Pydantic v2 schemas (BaseModel) for request/response
3. CRUD endpoints for each table: GET list, POST create, GET by id, PUT update, DELETE
4. Auth endpoints: POST /api/auth/login, POST /api/auth/register
5. A startup seed function that inserts 10 realistic sample records matching the DOMAIN
   - If pharma ecommerce: seed with real medicine/pharma product names (Paracetamol, Amoxicillin, etc.)
   - If food delivery: seed with real food items
   - If fashion: seed with clothing items
   - Always match the domain from "App" and "Domain" fields above
6. from auth import get_current_user for protected routes
7. from database import Base, SessionLocal, engine, get_db
8. Use @app.on_event("startup") or lifespan to run seed

Return ONLY the complete routes.py. Python 3.10 compatible. No markdown fences.'''


MOCK_DATA_PROMPT = '''You are a senior TypeScript developer. Generate realistic mock API data for a frontend app.

App: {title}
Domain: {requirement}
Pages in the app: {pages}
PRD summary: {prd_summary}

Generate a TypeScript mock.ts file that:
1. Has realistic arrays of 8-10 sample items matching the DOMAIN:
   - If pharma ecommerce: medicines — Paracetamol 500mg, Amoxicillin 250mg, Vitamin C tablets, etc. with MRP prices
   - If food delivery: dishes with restaurant names, prices in INR
   - If fashion: clothes with sizes, colors, brands
   - If HR/employee: employees with realistic Indian names, departments, salaries
   - ALWAYS match the actual domain — never use generic Laptop/Mouse data
2. Exports `mockFetch<T>(path, options): Promise<T>` function that:
   - Handles auth/login, auth/register
   - Handles GET /api/<entity> → returns {{ items: [...], total: N }}
   - Handles POST /api/<entity> → creates item, returns created item
   - Handles GET /api/<entity>/{{id}} → returns single item
   - Matches the pages listed above
3. Types: `type Json = Record<string, unknown>`

Important rules:
- All prices in INR (₹) if e-commerce
- Use realistic domain-specific field names
- No imports needed — pure TypeScript

Return ONLY the TypeScript code. No markdown fences.'''


def _fix_ts_imports(code: str) -> str:
    """
    Deterministically fix common import mistakes LLMs make for our frozen client.ts.
    client.ts uses named exports — LLMs often generate wrong default imports.
    """
    # Fix: import apiFetch from '../api/client' → import { apiFetch } from '../api/client'
    code = re.sub(
        r"import\s+apiFetch\s+from\s+['\"](\.\./)*api/client['\"]",
        "import { apiFetch } from '../api/client'",
        code,
    )
    # Fix: import apiFetch, { X } from '../api/client' → import { apiFetch, X } from '...'
    code = re.sub(
        r"import\s+apiFetch\s*,\s*\{([^}]+)\}\s+from\s+['\"](\.\./)*api/client['\"]",
        lambda m: "import { apiFetch, " + m.group(1).strip() + " } from '../api/client'",
        code,
    )
    # Fix: import { default as apiFetch } → import { apiFetch }
    code = re.sub(
        r"import\s*\{\s*default\s+as\s+apiFetch\s*\}\s+from\s+['\"](\.\./)*api/client['\"]",
        "import { apiFetch } from '../api/client'",
        code,
    )
    return code


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
    requirement: str = "",
    prd: str = "",
    all_pages: Optional[List[Tuple[str, str]]] = None,
) -> Optional[str]:
    """Use LLM to generate a fully custom MUI TypeScript page component with full app context."""
    if llm is None:
        try:
            from llm_helper import get_llm_for_user
            llm = get_llm_for_user(None, temperature=0.2)
        except Exception:
            return None

    # ── Build rich architecture context for THIS specific page ────────────────
    tables = (architecture.get("database_schema") or {}).get("tables") or []
    table_names = [t.get("table_name") or t.get("name") or "" for t in tables if isinstance(t, dict)]

    # Find this page's screen spec from architecture
    page_lower = page_name.lower()
    screen_spec = page_description
    api_endpoints = []

    screens = architecture.get("screens") or []
    for s in screens:
        if not isinstance(s, dict):
            continue
        if (s.get("name") or "").lower().replace(" ", "") in page_lower.replace("page", ""):
            if s.get("description"):
                screen_spec = s["description"]
            if s.get("components"):
                screen_spec += f"\nComponents: {', '.join(s['components'][:8])}"
            if s.get("api_calls"):
                api_endpoints = s["api_calls"][:6]
            break

    # Derive API endpoints from tables if not found in architecture
    if not api_endpoints:
        for tname in table_names[:4]:
            api_endpoints.append(f"GET /api/{tname}")
        # Page-specific endpoints
        if "dashboard" in page_lower or "home" in page_lower:
            api_endpoints = ["GET /api/dashboard", "GET /api/stats"]
        elif "product" in page_lower:
            api_endpoints = ["GET /api/products", "POST /api/products", "DELETE /api/products/{id}"]
        elif "order" in page_lower:
            api_endpoints = ["GET /api/orders", "POST /api/orders", "PUT /api/orders/{id}"]
        elif "user" in page_lower:
            api_endpoints = ["GET /api/users", "POST /api/users", "DELETE /api/users/{id}"]

    # All pages list for navigation context
    other_pages = [(n, _page_name_to_path(n)) for n, _ in (all_pages or []) if n != page_name and n != "LoginPage"]
    pages_nav = ", ".join(f"{n} ({p})" for n, p in other_pages[:8]) or "Dashboard (/)"

    # PRD summary — extract bullet points from PRD
    prd_summary = ""
    if prd:
        lines = [l.strip() for l in prd.split("\n") if l.strip().startswith(("- ", "* ", "•")) or "##" in l]
        prd_summary = "\n".join(lines[:15]) or prd[:600]

    prompt = PAGE_GENERATION_PROMPT.format(
        title=title,
        requirement=(requirement or title)[:600],
        prd_summary=prd_summary[:800],
        page_name=page_name,
        page_description=page_description,
        screen_spec=screen_spec[:600],
        api_endpoints="\n".join(api_endpoints) or f"GET /api/{page_lower.replace('page','')}",
        all_pages=pages_nav,
        primary=colors.get("primary", "#1565C0"),
        primary_dark=colors.get("primary_dark", "#0D47A1"),
        secondary=colors.get("secondary", "#7B1FA2"),
        accent=colors.get("accent", "#00897B"),
        background=colors.get("background", "#F5F5F5"),
        surface=colors.get("surface", "#FFFFFF"),
        text=colors.get("text", "#212121"),
        muted=colors.get("muted", "#757575"),
        style=colors.get("style", "corporate"),
    )

    try:
        resp = await llm.ainvoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        code = _extract_code_from_llm(text)
        code = _fix_ts_imports(code)
        if code and len(code) > 300 and "export default" in code:
            logger.info("[ai-page] ✓ %s  (%d chars)", page_name, len(code))
            return code
        logger.warning("[ai-page] %s: output too short (%d chars) — will fallback", page_name, len(code))
    except Exception as e:
        logger.warning("[ai-page] Failed to generate %s: %s", page_name, e)
    return None


async def generate_layout_with_ai(
    title: str,
    pages: List[Tuple[str, str]],
    colors: Dict[str, str],
    requirement: str = "",
    llm=None,
) -> Optional[str]:
    """Use LLM to generate a custom sidebar AppLayout instead of hardcoded template."""
    if llm is None:
        try:
            from llm_helper import get_llm_for_user
            llm = get_llm_for_user(None, temperature=0.15)
        except Exception:
            return None

    nav_pages = [(n, _page_name_to_path(n)) for n, _ in pages if n != "LoginPage"]
    pages_list = "\n".join(f"- {n} → {p}" for n, p in nav_pages)

    prompt = LAYOUT_GENERATION_PROMPT.format(
        title=title,
        requirement=(requirement or title)[:400],
        pages_list=pages_list,
        style=colors.get("style", "corporate"),
        primary=colors.get("primary", "#1565C0"),
        primary_dark=colors.get("primary_dark", "#0D47A1"),
        surface=colors.get("surface", "#FFFFFF"),
        muted=colors.get("muted", "#757575"),
        background=colors.get("background", "#F5F5F5"),
    )

    try:
        resp = await llm.ainvoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        code = _fix_ts_imports(_extract_code_from_llm(text))
        if code and len(code) > 300 and "export default" in code:
            logger.info("[ai-layout] ✓ AppLayout (%d chars)", len(code))
            return code
    except Exception as e:
        logger.warning("[ai-layout] AppLayout generation failed: %s", e)
    return None


async def generate_login_with_ai(
    title: str,
    colors: Dict[str, str],
    requirement: str = "",
    llm=None,
) -> Optional[str]:
    """Use LLM to generate a branded login page instead of hardcoded template."""
    if llm is None:
        try:
            from llm_helper import get_llm_for_user
            llm = get_llm_for_user(None, temperature=0.2)
        except Exception:
            return None

    prompt = LOGIN_PAGE_PROMPT.format(
        title=title,
        requirement=(requirement or title)[:400],
        style=colors.get("style", "corporate"),
        primary=colors.get("primary", "#1565C0"),
        primary_dark=colors.get("primary_dark", "#0D47A1"),
        background=colors.get("background", "#F5F5F5"),
        surface=colors.get("surface", "#FFFFFF"),
        text=colors.get("text", "#212121"),
    )

    try:
        resp = await llm.ainvoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        code = _fix_ts_imports(_extract_code_from_llm(text))
        if code and len(code) > 300 and "export default" in code:
            logger.info("[ai-login] ✓ LoginPage (%d chars)", len(code))
            return code
    except Exception as e:
        logger.warning("[ai-login] LoginPage generation failed: %s", e)
    return None


async def generate_backend_with_ai(
    title: str,
    prd: str,
    architecture: Dict[str, Any],
    llm=None,
    requirement: str = "",
) -> Optional[str]:
    """Use LLM to generate a domain-aware FastAPI routes.py with seed data matching the domain."""
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
        requirement=(requirement or title)[:400],
        prd=(prd or "")[:3000],
        tables_json=tables_json,
    )

    try:
        resp = await llm.ainvoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        code = _extract_code_from_llm(text)
        if code and len(code) > 300 and ("@router" in code or "APIRouter" in code):
            logger.info("[ai-backend] Generated routes.py (%d chars)", len(code))
            return code
    except Exception as e:
        logger.warning("[ai-backend] routes.py generation failed: %s", e)
    return None


async def generate_mock_data_with_ai(
    title: str,
    requirement: str,
    prd: str,
    pages: List[Tuple[str, str]],
    llm=None,
) -> Optional[str]:
    """Generate domain-specific mock data using LLM — no hardcoded products."""
    if llm is None:
        try:
            from llm_helper import get_llm_for_user
            llm = get_llm_for_user(None, temperature=0.2)
        except Exception:
            return None

    pages_str = ", ".join(n for n, _ in pages[:10])
    prd_lines = [l.strip() for l in (prd or "").split("\n") if l.strip().startswith(("- ", "* ", "•"))]
    prd_summary = "\n".join(prd_lines[:10]) or (prd or "")[:400]

    prompt = MOCK_DATA_PROMPT.format(
        title=title,
        requirement=(requirement or title)[:500],
        pages=pages_str,
        prd_summary=prd_summary,
    )

    try:
        resp = await llm.ainvoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        code = _extract_code_from_llm(text)
        if code and len(code) > 300 and "mockFetch" in code:
            logger.info("[ai-mock] ✓ mock.ts generated (%d chars) — domain: %s", len(code), title)
            return code
    except Exception as e:
        logger.warning("[ai-mock] mock.ts generation failed: %s", e)
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

    # ── Step 3: Generate ALL files in parallel (pages + layout + login + backend) ──
    logger.info("[ai-gen] Generating %d pages + layout + login + backend in parallel...", len(pages))

    async def gen_page(page_name: str, desc: str) -> Tuple[str, Optional[str]]:
        code = await generate_page_with_ai(
            page_name, desc, title, colors, architecture or {}, llm,
            requirement=requirement, prd=prd, all_pages=pages,
        )
        return page_name, code

    # Run everything in parallel — true Lovable-style parallel generation
    page_tasks = [gen_page(name, desc) for name, desc in pages]
    layout_task  = generate_layout_with_ai(title, pages, colors, requirement, llm)
    login_task   = generate_login_with_ai(title, colors, requirement, llm)
    backend_task = generate_backend_with_ai(title, prd, architecture or {}, llm, requirement=requirement)
    mock_task    = generate_mock_data_with_ai(title, requirement, prd, pages, llm)

    all_results = await asyncio.gather(
        *page_tasks, layout_task, login_task, backend_task, mock_task,
        return_exceptions=True,
    )

    n_pages      = len(page_tasks)
    page_results = all_results[:n_pages]
    ai_layout    = all_results[n_pages]
    ai_login     = all_results[n_pages + 1]
    ai_routes    = all_results[n_pages + 2]
    ai_mock      = all_results[n_pages + 3]

    # ── Step 4: Assemble file set ─────────────────────────────────────────────
    files = dict(get_fullstack_scaffold_files())

    # Always-present infrastructure files
    files["frontend/src/theme.ts"] = build_theme_ts(colors)
    files["frontend/src/auth.ts"] = _auth_ts()
    files["frontend/src/App.tsx"] = _build_app_tsx_for_pages(pages)
    files.pop("frontend/src/layout/AppShell.tsx", None)

    # mock.ts — LLM-generated with domain-specific data; fallback to template
    if isinstance(ai_mock, str) and ai_mock and "mockFetch" in ai_mock:
        files["frontend/src/api/mock.ts"] = ai_mock
        logger.info("[ai-gen] ✓ mock.ts from LLM (domain-aware)")
    else:
        logger.warning("[ai-gen] mock.ts LLM failed — using template fallback")
        files["frontend/src/api/mock.ts"] = _build_mock_ts_for_pages(pages, title)

    # AppLayout — LLM-generated, fallback to template
    if isinstance(ai_layout, str) and ai_layout and len(ai_layout) > 300:
        files["frontend/src/layout/AppLayout.tsx"] = ai_layout
        logger.info("[ai-gen] ✓ AppLayout from LLM")
    else:
        logger.warning("[ai-gen] AppLayout LLM failed — using template fallback")
        files["frontend/src/layout/AppLayout.tsx"] = _build_app_layout_tsx(title, pages, colors)

    # LoginPage — LLM-generated, fallback to template
    if isinstance(ai_login, str) and ai_login and len(ai_login) > 300:
        files["frontend/src/pages/LoginPage.tsx"] = ai_login
        logger.info("[ai-gen] ✓ LoginPage from LLM")
    else:
        logger.warning("[ai-gen] LoginPage LLM failed — using template fallback")
        files["frontend/src/pages/LoginPage.tsx"] = _build_login_page_tsx(title, colors)

    # AI-generated page files
    successful_pages = 0
    failed_pages = []
    for result in page_results:
        if isinstance(result, Exception):
            logger.warning("[ai-gen] Page task exception: %s", result)
            continue
        page_name, code = result
        if page_name == "LoginPage":
            continue  # Already handled above
        if code:
            files[f"frontend/src/pages/{page_name}.tsx"] = _fix_ts_imports(code)
            successful_pages += 1
        else:
            failed_pages.append(page_name)

    if failed_pages:
        logger.warning("[ai-gen] Fallback for pages: %s", failed_pages)
        _add_fallback_pages(files, failed_pages, colors, architecture or {})

    # Backend routes — LLM-generated, fallback to deterministic
    if isinstance(ai_routes, str) and ai_routes and len(ai_routes) > 300:
        files["backend/routes.py"] = ai_routes
        logger.info("[ai-gen] ✓ routes.py from LLM")
    else:
        logger.warning("[ai-gen] Backend LLM failed — using deterministic fallback")
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
