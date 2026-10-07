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
App domain: {domain}
What this app does: {requirement}

Key product requirements:
{prd_summary}

Generate pages that are specific and functional for the "{domain}" domain.
Include domain-appropriate data, terminology, UI patterns, and realistic sample values.

═══ PAGE TO BUILD ═══
Page: {page_name}
Purpose: {page_description}

Screen spec from architecture:
{screen_spec}

Related API endpoints for this page:
{api_endpoints}

Other pages in the app (for navigation links): {all_pages}

{actual_api_spec_section}
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
- Every component must be FULLY FUNCTIONAL with real state management (useState, useQuery)
- Forms must have validation and submit handlers that call the API
- Tables must handle loading states (MUI Skeleton or CircularProgress)
- Error states must be handled with user-friendly messages (no console.log in UI)
- All buttons must have onClick handlers that do real things
- Navigation links must use react-router-dom useNavigate with correct paths
- No placeholder text like 'TODO', 'Coming soon', 'Lorem ipsum'
- No disabled buttons without explanation
- Data must flow: fetch from API → display in UI → user can interact → update sent to API
- Use MUI Alert for success/error notifications after form submissions
- Every page must have a proper page title (Typography variant='h4' or h5)

Export default function {page_name}(). No markdown, no explanation, just the code.'''


LAYOUT_GENERATION_PROMPT = '''You are a senior React/TypeScript engineer. Generate the main app layout with sidebar navigation.

App: {title}
User requirement: {requirement}
Style: {style}

The sidebar navigation MUST include exactly these pages (and no others):
{pages_list}

Each nav item label should be user-friendly (no 'Page' suffix — 'Dashboard' not 'DashboardPage').
Icons should match the page purpose (import from @mui/icons-material):
- Dashboard / Overview / Home → DashboardIcon or BarChartIcon
- Products / Inventory / Items → InventoryIcon
- Users / Customers / Members / Employees → PeopleIcon
- Orders / Purchases / Transactions → ShoppingCartIcon
- Reports / Analytics / Statistics → AnalyticsIcon
- Settings / Config / Preferences → SettingsIcon
- Anomalies / Alerts / Warnings / Notifications → WarningAmberIcon or NotificationsIcon
- Messages / Chat / Communication → ChatIcon
- Suppliers / Vendors → LocalShippingIcon
- Finance / Billing / Payments → AccountBalanceIcon
- For any other page: use a sensible matching icon

Design system:
Primary: {primary} (sidebar background)
Primary Dark: {primary_dark}
Surface: {surface} (content area background)
Text on primary: #FFFFFF
Muted: {muted}
Background: {background}

Rules:
- Sidebar: fixed left, 240px wide, bg={primary}, white text and icons
- Top bar: app title only (NO login/logout button, NO user session display)
- Content area: bg={background}, right of sidebar, fills screen
- Use Drawer (permanent variant) for sidebar
- Show icon + label for each nav item
- Active route highlighted with slightly lighter bg
- useNavigate + useLocation for routing and active state
- Export default function AppLayout() with <Outlet /> for content
- Do NOT add any login, logout, or authentication-related UI elements

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
App type/domain: {app_domain}
PRD: {prd}

Generate backend code that is tailored specifically for a "{app_domain}" application.
Use domain-appropriate models, endpoints, seed data, and business logic.

Database tables from architecture:
{tables_json}

Generate a complete FastAPI routes.py with:

1. SQLAlchemy ORM models (Base, columns, relationships)
2. Pydantic v2 schemas for request/response  
3. CRUD endpoints — use RESOURCE-SPECIFIC paths like:
   - GET  /api/products          → list all (NOT /{item_id} catch-alls)
   - POST /api/products          → create
   - GET  /api/products/{{id}}   → get one  (integer id, resource-prefixed)
   - PUT  /api/products/{{id}}   → update
   - DELETE /api/products/{{id}} → delete
   CRITICAL: NEVER use bare `/{item_id}` or `/{id}` routes without resource prefix!
   BAD:  @router.get("/{item_id}")  ← this matches EVERYTHING
   GOOD: @router.get("/products/{product_id}")  ← resource-specific

4. Seed 10 realistic records matching the DOMAIN on startup:
   - Pharma ecommerce: Paracetamol 500mg ₹45, Amoxicillin 250mg ₹120, Vitamin C 1000mg ₹85...
   - Food delivery: Paneer Tikka ₹180, Chicken Biryani ₹220...
   - Fashion: Blue Denim Jeans M ₹899, White Cotton Shirt L ₹599...
   - ALWAYS match the actual domain, never use generic item data

5. from database import Base, SessionLocal, engine, get_db
6. router = APIRouter() — main.py includes it with prefix="/api"
7. Every endpoint listed in the architecture MUST be implemented
8. All CRUD operations must be complete (not just GET, include POST/PUT/DELETE)
9. Response models must use consistent field names across all endpoints
10. Include proper HTTP status codes (201 for created, 404 for not found, 400 for bad request)
11. Seed data must be realistic and substantial (at least 10-20 records per entity)
12. All endpoints must have correct resource-prefixed paths (already enforced above)
13. Include error handling: try/except with HTTPException for all database operations
14. Use consistent ID fields: always 'id' (not 'userId', 'itemId', etc.)

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
- Generate at least 15-25 mock records per entity
- Use realistic domain-specific data (real-sounding names, realistic values, proper date ranges)
- All IDs must be unique integers starting from 1
- All foreign key IDs must reference valid parent IDs
- Status fields should have variety (mix of active/inactive, pending/approved/rejected, etc.)
- Include edge cases: some items with optional fields null, some with max values

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
        resp = await llm.ainvoke(prompt)
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
    domain: str = "general",
    actual_api_spec: str = "",
    existing_content: str = None,
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
    # Use token-based matching so "ProductListPage" matches screen name "Products"
    import re as _re
    _page_tokens = set(_re.split(r'[^a-z0-9]+', page_lower.replace("page", "").replace("view", ""))) - {""}
    for s in screens:
        if not isinstance(s, dict):
            continue
        _screen_tokens = set(_re.split(r'[^a-z0-9]+', (s.get("name") or "").lower())) - {""}
        _matched = bool(_screen_tokens and _screen_tokens & _page_tokens)
        if not _matched:
            # Fallback: old substring check
            _matched = (s.get("name") or "").lower().replace(" ", "") in page_lower.replace("page", "")
        if _matched:
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

    # Build actual API spec section if provided
    actual_api_spec_section = ""
    if actual_api_spec:
        actual_api_spec_section = (
            "═══ ACTUAL BACKEND API (generated, use these exact paths) ═══\n"
            + actual_api_spec
            + "\n\nIMPORTANT: Only call endpoints listed above. Use exact paths. Do NOT invent new endpoints.\n"
        )

    prompt = PAGE_GENERATION_PROMPT.format(
        title=title,
        domain=domain,
        requirement=(requirement or title)[:600],
        prd_summary=prd_summary[:800],
        page_name=page_name,
        page_description=page_description,
        screen_spec=screen_spec[:600],
        api_endpoints="\n".join(api_endpoints) or f"GET /api/{page_lower.replace('page','')}",
        actual_api_spec_section=actual_api_spec_section,
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

    # ── When modifying an existing page, append the existing code so LLM preserves it ──
    if existing_content and len(existing_content) > 50:
        prompt += (
            "\n\n═══ EXISTING PAGE CODE (PRESERVE WHAT WORKS) ═══\n"
            "The page already has the following implementation. "
            "Your job is to MODIFY it to address the user's request "
            "while preserving all existing functionality, logic, and structure "
            "that is NOT related to the change.\n\n"
            f"User's change request: {page_description}\n\n"
            "EXISTING CODE:\n"
            f"{existing_content[:6000]}{'...' if len(existing_content) > 6000 else ''}\n\n"
            "INSTRUCTIONS:\n"
            "- Make ONLY the changes needed for the user request\n"
            "- Keep all existing imports, state, handlers, and JSX structure intact\n"
            "- Do not remove functionality that wasn't part of the change request\n"
            "- Output the COMPLETE updated file"
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
    domain: str = "general",
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
        app_domain=domain,
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


def _default_seed_data(domain: str = "general") -> Dict[str, Any]:
    """Return a minimal seed_data.json dict with non-zero demo KPIs for the dynamic router fallback."""
    kpis: Dict[str, Any] = {
        "total_lots": 142,
        "pending_inspections": 23,
        "released": 98,
        "held": 12,
        "rejected": 9,
        "open_capa": 7,
        "incoming_lots": 18,
        "total_items": 3420,
        "low_stock_alerts": 14,
        "total_orders": 287,
        "pending_orders": 34,
        "total_suppliers": 67,
    }
    return {
        "kpis": kpis,
        "dashboard/kpis": kpis,
        "dashboard/stats": kpis,
        "stats": kpis,
    }


async def _generate_seed_json(
    mock_ts_content: str,
    domain: str = "general",
    llm=None,
) -> Optional[str]:
    """Convert mock.ts TypeScript data to a seed_data.json string via LLM.

    Returns a JSON string (for writing to backend/seed_data.json) or None on failure.
    The file is later read by the dynamic router to serve realistic data in preview mode.
    """
    if not mock_ts_content or not mock_ts_content.strip():
        return None
    if llm is None:
        try:
            from llm_helper import get_llm_for_user
            llm = get_llm_for_user(None, temperature=0.1)
        except Exception:
            return None

    prompt = (
        "You are a data engineer. Convert the following TypeScript mock data file into a valid JSON object "
        "that can be used as a seed data file for a REST API backend.\n\n"
        "Rules:\n"
        "1. Extract ALL data arrays into top-level keys using their collection name (e.g. 'products', 'users', 'orders').\n"
        "2. Extract any KPI/stats/dashboard values into a 'kpis' key AND a 'dashboard/kpis' key (same object).\n"
        "3. If the mock file has counters or summary numbers, include them in 'kpis'.\n"
        "4. Array items must be plain JSON objects (no TypeScript types, no Date constructors, no undefined).\n"
        "5. Return ONLY valid JSON. No markdown fences, no explanation, no trailing commas.\n\n"
        f"TypeScript mock.ts (domain: {domain}):\n"
        f"{mock_ts_content[:5000]}\n"
    )

    try:
        resp = await llm.ainvoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        obj = _extract_json_from_llm(text)
        if obj and isinstance(obj, dict):
            # Ensure dashboard/kpis alias always exists
            if "kpis" in obj and "dashboard/kpis" not in obj:
                obj["dashboard/kpis"] = obj["kpis"]
            if "dashboard/kpis" in obj and "kpis" not in obj:
                obj["kpis"] = obj["dashboard/kpis"]
            return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("[ai-gen] _generate_seed_json failed: %s", e)
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


async def _extract_pages_from_requirement(requirement: str, llm=None) -> List[Tuple[str, str]]:
    """
    Use LLM to extract logical pages from the user's raw requirement text.
    Returns list of (PageComponentName, description) tuples.
    Falls back gracefully to an empty list on any failure.
    """
    if llm is None:
        try:
            from llm_helper import get_llm_for_user
            llm = get_llm_for_user(None, temperature=0.1)
        except Exception:
            return []

    prompt = f"""Given this app description, list the main pages/screens this app should have.

App description: {requirement}

Respond with ONLY a JSON array:
[
  {{"name": "DashboardPage", "description": "Main overview with key metrics"}},
  {{"name": "ProductsPage", "description": "Browse and manage products"}}
]

Rules:
- 3-8 pages max
- Names must be PascalCase ending with 'Page'
- Match exactly what the user described — don't add generic pages they didn't ask for
- First page should be the most important/main page
- Do NOT include LoginPage"""

    try:
        resp = await llm.ainvoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        # Strip markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.M)
        text = re.sub(r'```\s*$', '', text, flags=re.M)
        # Find JSON array
        m = re.search(r'\[[\s\S]*\]', text)
        if not m:
            return []
        items = json.loads(m.group(0))
        if not isinstance(items, list):
            return []
        result: List[Tuple[str, str]] = []
        seen: set = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            desc = str(item.get("description") or name).strip()
            if not name or name.lower() in seen:
                continue
            # Enforce PascalCase + Page suffix
            if not name.endswith("Page") and not name.endswith("View"):
                name += "Page"
            seen.add(name.lower())
            result.append((name, desc[:400]))
        logger.info("[req-pages] Extracted %d pages from requirement via LLM", len(result))
        return result[:8]
    except Exception as e:
        logger.warning("[req-pages] _extract_pages_from_requirement failed: %s", e)
        return []


def _to_page_component_name(name: str) -> str:
    """Convert 'Login Page' → 'LoginPage', 'Product List' → 'ProductListPage'."""
    # Remove special chars, title-case each word
    name = re.sub(r'[^a-zA-Z0-9 ]', ' ', name)
    parts = [p.capitalize() for p in name.split()]
    result = "".join(parts)
    if not result.endswith("Page") and not result.endswith("View"):
        result += "Page"
    return result


def _build_app_tsx_for_pages(pages: List[Tuple[str, str]], has_auth: bool = False) -> str:
    """Generate App.tsx routing for a list of pages.

    has_auth=False (default): plain routes with no login, no PrivateRoute, no /login route.
    has_auth=True: legacy auth path retained for backward compatibility.

    Fix 4: `/` redirects to the FIRST content page (priority: Dashboard/Home/Overview → else first page).
    """
    imports = []
    routes = []

    if has_auth:
        imports.append("import LoginPage from './pages/LoginPage';")

    # ── Determine landing page with priority: Dashboard/Home/Overview → else first ──
    _preferred_landing = {"dashboardpage", "homepage", "overviewpage", "mainpage"}
    landing_page: Optional[str] = None
    # First pass: preferred names
    for page_name, _ in pages:
        if page_name == "LoginPage":
            continue
        if page_name.lower() in _preferred_landing:
            landing_page = page_name
            break
    # Second pass: fallback to first non-login page
    if landing_page is None:
        for page_name, _ in pages:
            if page_name != "LoginPage":
                landing_page = page_name
                break

    landing_path = _page_name_to_path(landing_page) if landing_page else "/"

    for page_name, _ in pages:
        if page_name == "LoginPage":
            continue
        imports.append(f"import {page_name} from './pages/{page_name}';")
        route_path = _page_name_to_path(page_name)

        if has_auth:
            entry = f'        <Route path="{route_path}" element={{<PrivateRoute><{page_name} /></PrivateRoute>}} />'
        else:
            entry = f'        <Route path="{route_path}" element={{<{page_name} />}} />'

        # Dashboard/Home gets inserted first so it appears at top of route list
        if page_name == landing_page and landing_path == "/":
            routes.insert(0, entry)
        else:
            routes.append(entry)

    imports_str = "\n".join(imports)
    routes_str = "\n".join(routes)

    # Build explicit "/" → landing_path redirect only when landing page doesn't own "/"
    root_redirect_line = ""
    if landing_path != "/":
        root_redirect_line = f'\n      <Route path="/" element={{<Navigate to="{landing_path}" replace />}} />'

    if has_auth:
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
      <Route path="/login" element={{<LoginPage />}} />{root_redirect_line}
      <Route element={{<AppLayout />}}>
{routes_str}
      </Route>
      <Route path="*" element={{<Navigate to="{landing_path}" replace />}} />
    </Routes>
  );
}}
"""
    else:
        return f"""import {{ Navigate, Route, Routes }} from 'react-router-dom';
import AppLayout from './layout/AppLayout';
{imports_str}

export default function App() {{
  return (
    <Routes>{root_redirect_line}
      <Route element={{<AppLayout />}}>
{routes_str}
      </Route>
      <Route path="*" element={{<Navigate to="{landing_path}" replace />}} />
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
        "import { AppBar, Box, Drawer, List, ListItemButton, ListItemText, Toolbar, Typography } from '@mui/material';\n"
        "import { Outlet, Link, useLocation } from 'react-router-dom';\n\n"
        "const DRAWER = 240;\n"
        "const NAV = [\n__NAV__\n];\n\n"
        "export default function AppLayout() {\n"
        "  const location = useLocation();\n"
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
        "          <Toolbar>\n"
        "            <Typography variant=\"body2\" color=\"__MUTED__\">__TITLE__</Typography>\n"
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


def _build_mock_ts_for_pages(pages: List[Tuple[str, str]], title: str, requirement: str = "", prd: str = "") -> str:
    """Generate a domain-aware mock.ts for all page API calls."""
    # Build a domain-specific products seed if this looks like a shopping/product app
    product_seed_lines: List[str] = []
    try:
        from app_builder.services.fullstack_ecommerce_generator import (
            _detect_product_domain, _DOMAIN_PRODUCTS, _make_products_ts,
        )
        has_products_page = any(
            "product" in n.lower() or "shop" in n.lower() or "medicine" in n.lower()
            for n, _ in pages
        )
        if has_products_page:
            domain = _detect_product_domain(requirement or title, prd)
            domain_products = _DOMAIN_PRODUCTS.get(domain, [])
            if domain_products:
                products_ts = _make_products_ts(domain_products)
                product_seed_lines = [
                    "// Domain-specific products (" + domain + ")",
                    "const _domainProducts: any[] = " + products_ts + ";",
                ]
    except Exception:
        pass

    lines = [
        f"// Auto-generated mock API — {title}",
        "// All API calls fall back here when backend is unreachable.",
        *product_seed_lines,
        "const USERS = [",
        "  { id: 1, email: 'admin@example.com', password: 'admin123', name: 'Admin User', role: 'ADMIN' },",
        "  { id: 2, email: 'user@example.com', password: 'user123', name: 'Demo User', role: 'USER' },",
        "];",
        "let _id = 100;",
        "const _store: Record<string, any[]> = { items: [], records: [], entries: [] };",
        # Pre-seed products with domain data so /api/products returns correct items
        "if (typeof _domainProducts !== 'undefined') { _store['products'] = [..._domainProducts]; }",
        "",
        "// Exact-path overrides for endpoints that don't map 1-to-1 to a collection.",
        "// /api/dashboard/kpis, /api/stats, /api/metrics etc. are returned directly.",
        "const _ENDPOINTS: Record<string, unknown> = {",
        "  '/api/dashboard/kpis': {",
        "    total_lots: 142, pending_inspections: 23, released: 98, held: 12, rejected: 9,",
        "    open_capa: 7, incoming_lots: 18, total_items: 3420, low_stock_alerts: 14,",
        "    on_time_delivery_rate: 94.2, inventory_turnover: 8.3,",
        "    total_orders: 287, pending_orders: 34, total_suppliers: 67, active_suppliers: 54,",
        "  },",
        "  '/api/dashboard': {",
        "    total_sales: 284500, orders_today: 47, active_users: 1284, revenue: 284500,",
        "    growth_rate: 12.4, conversion_rate: 3.8,",
        "    total_lots: 142, pending_inspections: 23, released: 98, held: 12, rejected: 9,",
        "    open_capa: 7,",
        "    total: 1284, active: 987, pending: 142, completed: 1089,",
        "    trend: [60, 72, 80, 85, 91, 94, 100, 110, 118, 128],",
        "  },",
        "  '/api/stats': { total: 1284, active: 987, pending: 142, completed: 1089 },",
        "  '/api/metrics': { total: 500, anomalies: 23, alerts: 7, score: 94.2 },",
        "  '/api/reports/summary': { total_anomalies: 156, mttd_minutes: 4.2, anomaly_rate: 3.1 },",
        "};",
        "",
        "function ok<T>(data: T): Promise<T> { return Promise.resolve(data); }",
        "function err(msg: string): Promise<never> { return Promise.reject(new Error(msg)); }",
        "",
        "export async function mockFetch<T = unknown>(path: string, options: RequestInit = {}): Promise<T> {",
        "  const method = (options.method || 'GET').toUpperCase();",
        "  const clean = path.replace(/^\\/api\\//, '').replace(/\\?.*$/, '');",
        "  const parts = clean.split('/').filter(Boolean);",
        "",
        "  // Exact-path overrides (GET only) — checked before collection routing.",
        "  if (method === 'GET') {",
        "    const pathNoQuery = path.split('?')[0];",
        "    if (Object.prototype.hasOwnProperty.call(_ENDPOINTS, pathNoQuery)) {",
        "      return _ENDPOINTS[pathNoQuery] as T;",
        "    }",
        "  }",
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
        "  // Generic dashboard / stats fallback (when path didn't match exact override above)",
        "  if ((clean === 'dashboard' || clean === 'stats') && method === 'GET') {",
        "    return ok({ total: 1284, active: 987, pending: 142, completed: 1089,",
        "      total_sales: 284500, orders_today: 47, revenue: 284500, growth_rate: 12.4,",
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


def _extract_actual_api_spec(backend_code: str) -> str:
    """
    Parse generated routes.py to extract actual endpoint paths, methods, and response models.
    Returns a short spec string to pass back to page generators so they use real paths.
    """
    if not backend_code:
        return ""
    lines: List[str] = []
    # Match @router.METHOD("path") or @app.METHOD("path") with optional response_model
    route_pattern = re.compile(
        r'@(?:router|app)\.(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']'
        r'(?:[^)]*response_model\s*=\s*(\w+))?',
        re.IGNORECASE,
    )
    for m in route_pattern.finditer(backend_code):
        method = m.group(1).upper()
        path = m.group(2)
        response_model = m.group(3) or ""
        # Prepend /api if not already there (routes.py paths are usually without /api prefix
        # since main.py adds prefix="/api")
        display_path = path if path.startswith("/api") else f"/api{path}"
        entry = f"{method} {display_path}"
        if response_model:
            entry += f"  → {response_model}"
        lines.append(entry)
    if not lines:
        return ""
    # Deduplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            unique.append(l)
    return "\n".join(unique[:40])


async def _fix_api_consistency(files: Dict[str, str], llm=None) -> Dict[str, str]:
    """
    Post-generation consistency check:
    1. Extract all apiFetch / fetch calls from frontend files
    2. Extract all route paths from backend/routes.py or backend/main.py
    3. Find mismatches (frontend calls a path not present in backend)
    4. If mismatches found, ask LLM to add the missing routes to backend
    Returns the (possibly updated) files dict. Never raises — on any error returns files unchanged.
    """
    if llm is None:
        return files

    try:
        # ── Collect frontend API calls ────────────────────────────────────────
        frontend_calls: set = set()
        api_call_pattern = re.compile(
            r'''apiFetch\s*(?:<[^>]*>)?\s*\(\s*[`'"]((?:/api)?/[^`'"?#\s]+)[`'"]''',
            re.MULTILINE,
        )
        fetch_pattern = re.compile(
            r'''fetch\s*\(\s*[`'"]((?:/api)/[^`'"?#\s]+)[`'"]''',
            re.MULTILINE,
        )
        for path, content in files.items():
            if not path.startswith("frontend/") or not path.endswith((".ts", ".tsx")):
                continue
            for m in api_call_pattern.finditer(content):
                # normalise: strip trailing /{id} style segments for matching
                endpoint = re.sub(r"/\{[^}]+\}$", "/{id}", m.group(1))
                frontend_calls.add(endpoint)
            for m in fetch_pattern.finditer(content):
                endpoint = re.sub(r"/\{[^}]+\}$", "/{id}", m.group(1))
                frontend_calls.add(endpoint)

        # ── Collect backend routes ────────────────────────────────────────────
        backend_key = "backend/routes.py"
        backend_content = files.get(backend_key) or files.get("backend/main.py") or ""
        if not backend_content:
            return files

        backend_routes: set = set()
        route_pat = re.compile(
            r'@(?:router|app)\.(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']',
            re.IGNORECASE,
        )
        for m in route_pat.finditer(backend_content):
            raw = m.group(2)
            # Normalise to /api-prefixed path
            normalized = raw if raw.startswith("/api") else f"/api{raw}"
            normalized = re.sub(r"/\{[^}]+\}$", "/{id}", normalized)
            backend_routes.add(normalized)

        # ── Find mismatches ───────────────────────────────────────────────────
        missing = sorted(
            fc for fc in frontend_calls
            if fc not in backend_routes
            and "/auth/" not in fc  # auth is expected to already be there
            and fc not in ("/api/dashboard", "/api/stats")  # generic paths OK
        )

        if not missing:
            logger.info("[consistency] ✓ No API mismatches found")
            return files

        logger.warning("[consistency] Frontend calls %d routes not in backend: %s", len(missing), missing)

        # Limit how many we ask LLM to add to avoid bloating the prompt
        missing_to_fix = missing[:10]
        missing_list = "\n".join(f"- {r}" for r in missing_to_fix)
        prompt = (
            "The following frontend API calls have no matching backend routes.\n"
            "Add the missing routes to the FastAPI routes.py below.\n"
            "Keep all existing routes intact. Only append the new ones at the end.\n\n"
            f"Missing routes:\n{missing_list}\n\n"
            "Current backend/routes.py:\n"
            f"{backend_content[:6000]}\n\n"
            "Return ONLY the complete updated routes.py. No markdown fences. Python 3.10 compatible."
        )
        resp = await llm.ainvoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        updated = _extract_code_from_llm(text)
        if updated and len(updated) > len(backend_content) * 0.8 and ("@router" in updated or "APIRouter" in updated):
            files[backend_key] = updated
            logger.info("[consistency] ✓ Backend updated with %d missing routes", len(missing_to_fix))
        else:
            logger.warning("[consistency] LLM consistency fix returned invalid/short code — skipping")
    except Exception as exc:
        logger.warning("[consistency] _fix_api_consistency failed (non-fatal): %s", exc)

    return files


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
    3. Generates backend FIRST (so we know real API paths/shapes)
    4. Extracts actual API spec from generated backend
    5. Generates all pages in parallel (with real API spec injected)
    6. Assembles file set using ACTUALLY generated pages for routing
    7. Runs API consistency check to patch any mismatched routes
    """
    from app_builder.services.fullstack_codegen import get_fullstack_scaffold_files
    from app_builder.services.fullstack_frontend_generator import theme_from_tokens, build_theme_ts
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

    # ── Step 0: Detect app domain ─────────────────────────────────────────────
    try:
        from app_builder.services.fullstack_app_generator import _classify_with_llm, _keyword_classify
        detected_domain = await _classify_with_llm(requirement, prd) or _keyword_classify(requirement, prd)
    except Exception:
        try:
            from app_builder.services.fullstack_app_generator import _keyword_classify
            detected_domain = _keyword_classify(requirement, prd)
        except Exception:
            detected_domain = "generic"
    logger.info("[ai-gen] Detected domain: %s", detected_domain)

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

    # Fix 1: If architecture produced fewer than 3 pages, fall back to LLM-based
    # extraction directly from the raw requirement text (smarter page extraction).
    if len(pages) < 3:
        logger.info(
            "[ai-gen] Only %d pages from architecture/PRD — trying LLM requirement extraction",
            len(pages),
        )
        try:
            req_pages = await _extract_pages_from_requirement(requirement, llm)
            if req_pages:
                # Merge: start with requirement-derived pages, append any non-duplicate
                # pages that were already extracted from architecture/PRD.
                existing_names_lower = {p[0].lower() for p in req_pages}
                merged = list(req_pages)
                for name, desc in pages:
                    if name.lower() not in existing_names_lower:
                        merged.append((name, desc))
                pages = merged[:12]
                logger.info("[ai-gen] Requirement-extracted pages: %s", [p[0] for p in pages])
        except Exception as _re:
            logger.warning("[ai-gen] Requirement page extraction failed: %s", _re)

    # Ensure we always have a dashboard/home page
    page_names = [p[0] for p in pages]
    if not any("dashboard" in n.lower() or "home" in n.lower() for n in page_names):
        pages.insert(0, ("DashboardPage", "Main dashboard with key metrics and overview"))

    logger.info("[ai-gen] Pages to generate: %s", [p[0] for p in pages])

    # ── Step 3a: Generate backend FIRST so pages know the actual API shape ────
    logger.info("[ai-gen] Generating backend first to extract real API spec...")
    ai_routes: Any = None
    actual_api_spec = ""
    try:
        ai_routes = await generate_backend_with_ai(
            title, prd, architecture or {}, llm,
            requirement=requirement, domain=detected_domain,
        )
        if isinstance(ai_routes, str) and ai_routes:
            actual_api_spec = _extract_actual_api_spec(ai_routes)
            logger.info("[ai-gen] Extracted %d backend routes for page context",
                        actual_api_spec.count("\n") + 1 if actual_api_spec else 0)
    except Exception as _be:
        logger.warning("[ai-gen] Backend-first generation failed: %s — continuing without spec", _be)

    # ── Step 3b: Generate pages in parallel (now with real API spec) ──────────
    logger.info("[ai-gen] Generating %d pages + layout + login + mock in parallel...", len(pages))

    async def gen_page(page_name: str, desc: str) -> Tuple[str, Optional[str]]:
        code = await generate_page_with_ai(
            page_name, desc, title, colors, architecture or {}, llm,
            requirement=requirement, prd=prd, all_pages=pages,
            domain=detected_domain,
            actual_api_spec=actual_api_spec,
        )
        return page_name, code

    # Run layout, mock, and all pages in parallel — backend is already done
    page_tasks = [gen_page(name, desc) for name, desc in pages]
    layout_task = generate_layout_with_ai(title, pages, colors, requirement, llm)
    mock_task   = generate_mock_data_with_ai(title, requirement, prd, pages, llm)

    all_results = await asyncio.gather(
        *page_tasks, layout_task, mock_task,
        return_exceptions=True,
    )

    n_pages      = len(page_tasks)
    page_results = all_results[:n_pages]
    ai_layout    = all_results[n_pages]
    ai_mock      = all_results[n_pages + 1]

    # ── Step 4: Assemble file set ─────────────────────────────────────────────
    files = dict(get_fullstack_scaffold_files())

    # Always-present infrastructure files
    files["frontend/src/theme.ts"] = build_theme_ts(colors)
    files.pop("frontend/src/layout/AppShell.tsx", None)

    # mock.ts — LLM-generated with domain-specific data; fallback to template
    if isinstance(ai_mock, str) and ai_mock and "mockFetch" in ai_mock:
        files["frontend/src/api/mock.ts"] = ai_mock
        logger.info("[ai-gen] ✓ mock.ts from LLM (domain-aware)")
    else:
        logger.warning("[ai-gen] mock.ts LLM failed — using domain-aware template fallback")
        files["frontend/src/api/mock.ts"] = _build_mock_ts_for_pages(pages, title, requirement=requirement, prd=prd)

    # AppLayout — LLM-generated, fallback to template (use planned page list for now; refined below)
    if isinstance(ai_layout, str) and ai_layout and len(ai_layout) > 300:
        files["frontend/src/layout/AppLayout.tsx"] = ai_layout
        logger.info("[ai-gen] ✓ AppLayout from LLM")
    else:
        logger.warning("[ai-gen] AppLayout LLM failed — using template fallback")
        files["frontend/src/layout/AppLayout.tsx"] = _build_app_layout_tsx(title, pages, colors)

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

    # Backend routes — use backend generated in Step 3a; fallback to deterministic
    if isinstance(ai_routes, str) and ai_routes and len(ai_routes) > 300:
        files["backend/routes.py"] = ai_routes
        logger.info("[ai-gen] ✓ routes.py from LLM (pre-generated)")
    else:
        logger.warning("[ai-gen] Backend LLM failed — using deterministic fallback")
        _add_fallback_backend(files, title, architecture or {})

    # ── Step 5: Derive ACTUAL page list from generated files and rebuild routing ──
    actual_page_keys = sorted(
        k for k in files
        if k.startswith("frontend/src/pages/") and k.endswith(".tsx")
    )
    actual_pages: List[Tuple[str, str]] = []
    for key in actual_page_keys:
        page_file_name = key.split("/")[-1].replace(".tsx", "")
        # Find the original description if available
        orig_desc = next((desc for name, desc in pages if name == page_file_name), page_file_name)
        actual_pages.append((page_file_name, orig_desc))

    # Rebuild App.tsx with the actual pages (not the planned list)
    files["frontend/src/App.tsx"] = _build_app_tsx_for_pages(actual_pages, has_auth=False)
    logger.info("[ai-gen] App.tsx built from %d actual pages (no auth)", len(actual_pages))

    # Rebuild AppLayout if it was template-generated (LLM-generated already has correct nav)
    if not (isinstance(ai_layout, str) and ai_layout and len(ai_layout) > 300):
        files["frontend/src/layout/AppLayout.tsx"] = _build_app_layout_tsx(title, actual_pages, colors)
        logger.info("[ai-gen] AppLayout rebuilt from actual page list (%d pages)", len(actual_pages))

    # ── Step 6: Post-generation API consistency check ─────────────────────────
    logger.info("[ai-gen] Running API consistency check...")
    files = await _fix_api_consistency(files, llm)

    # ── Step 7: Write backend/seed_data.json for dynamic router fallback ──────
    # This lets the dynamic router serve realistic data when previewing the app,
    # so dashboards never show zeros.
    mock_content = files.get("frontend/src/api/mock.ts", "")
    try:
        seed_json = await _generate_seed_json(mock_content, detected_domain, llm)
        if seed_json:
            files["backend/seed_data.json"] = seed_json
            logger.info("[ai-gen] ✓ backend/seed_data.json written from mock.ts")
        else:
            files["backend/seed_data.json"] = json.dumps(
                _default_seed_data(detected_domain), indent=2, ensure_ascii=False
            )
            logger.info("[ai-gen] backend/seed_data.json: using default demo fallback")
    except Exception as _seed_exc:
        logger.warning("[ai-gen] seed_data.json generation failed, using defaults: %s", _seed_exc)
        files["backend/seed_data.json"] = json.dumps(
            _default_seed_data(detected_domain), indent=2, ensure_ascii=False
        )

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
