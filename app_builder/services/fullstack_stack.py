"""Locked tech stack and prompt addendums for Fullstack Builder."""
from __future__ import annotations

from typing import Any, Dict, Optional

FULLSTACK_KIND = "fullstack"

LOCKED_FRONTEND = {
    "framework": "React",
    "language": "TypeScript",
    "template": "Vite + React + TypeScript",
    "ui_library": "Material UI (MUI)",
    "routing": "React Router",
    "data_fetching": "TanStack Query",
    "charts": "Recharts",
    "state_management": "TanStack Query + React state",
}

LOCKED_BACKEND = {
    "framework": "FastAPI",
    "language": "Python",
    "validation": "Pydantic",
    "orm": "SQLAlchemy",
    "database": "PostgreSQL",
    "api_style": "REST",
    "auth": "JWT",
}

STACK_SUMMARY = """LOCKED TECH STACK (do not deviate):
Frontend: React, TypeScript, Vite, Material UI, React Router, TanStack Query, Recharts.
Backend: Python, FastAPI, Pydantic, SQLAlchemy, PostgreSQL, REST APIs, JWT authentication.
Architecture layers: React Frontend → FastAPI REST API → Service Layer → PostgreSQL → AI/Risk services.
Frontend must work end-to-end even if the backend is down, using a mock API layer."""


def is_fullstack(builder_kind: Optional[str]) -> bool:
    return (builder_kind or "").strip().lower() == FULLSTACK_KIND


def lock_architecture(architecture: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    arch = dict(architecture or {})
    frontend = dict(arch.get("frontend_structure") or {})
    backend = dict(arch.get("backend_structure") or {})
    frontend.update(
        {
            "framework": "React",
            "language": "TypeScript",
            "template": "Vite + React + TypeScript",
            "ui_library": "Material UI (MUI)",
            "routing": "React Router",
            "data_fetching": "TanStack Query",
            "charts": "Recharts",
            "state_management": "TanStack Query + React state",
        }
    )
    backend.update(
        {
            "framework": "FastAPI",
            "language": "Python",
            "validation": "Pydantic",
            "orm": "SQLAlchemy",
            "database": "PostgreSQL",
            "api_style": "REST",
            "auth": "JWT",
        }
    )
    arch["frontend_structure"] = frontend
    arch["backend_structure"] = backend
    deployment = dict(arch.get("deployment") or {})
    deployment.setdefault("containerization", "Docker")
    deployment.setdefault("orchestration", "Docker Compose")
    arch["deployment"] = deployment
    arch.setdefault(
        "project_structure",
        {
            "backend": [
                "main.py",
                "requirements.txt",
                "database.py",
                "models.py",
                "schemas.py",
                "routes.py",
                "auth.py",
                "services/",
                "alembic/",
                "seed.py",
            ],
            "frontend": [
                "package.json",
                "tsconfig.json",
                "vite.config.ts",
                "src/main.tsx",
                "src/App.tsx",
                "src/theme.ts",
                "src/api/client.ts",
                "src/api/mock.ts",
                "src/pages/",
                "src/components/",
            ],
        },
    )
    return arch


def prd_system_addendum() -> str:
    return f"""
{STACK_SUMMARY}

For fullstack enterprise apps you MUST also include:
# 5. Application Modules
- Every module/screen from the requirement (dashboard, entities, workflows, AI assistant, reports).
# 6. API Contract
- REST endpoints with method, path, request, response, roles.
# 7. Data Model
- Tables, PKs, FKs, indexes, timestamps, constraints.
# 8. Roles & Permissions
- ADMIN, QUALITY_MANAGER / manager, INSPECTOR / operator, VIEWER (or roles from the prompt).
# 9. Quality Requirements
- Loading, empty, error states; pagination; search; filters; audit; mock fallback if APIs fail.
# 10. Deliverables
- Frontend, backend, Postgres schema, migrations, seed data, docker-compose, .env.example, README, tests.

Extract colors, density, and industry UI tone from the prompt. Do not invent a different tech stack.
"""


def uiux_system_addendum() -> str:
    return f"""
{STACK_SUMMARY}

UI/UX rules for fullstack enterprise apps:
- Professional industry UI matching the prompt (automotive/manufacturing/healthcare etc.).
- Light/white backgrounds unless the prompt specifies otherwise.
- Strong information hierarchy, compact data tables, KPI cards, status chips, risk indicators.
- Responsive desktop/tablet layout, accessible typography, consistent spacing.
- Extract exact hex colors from the prompt when present; otherwise derive a cohesive palette from industry tone.
- List EVERY screen from the requirement with layout, components, empty/loading/error states, and actions.
- Reusable components MUST be named (KPICard, StatusChip, RiskBadge, tables, charts, AIChatPanel, DecisionPanel, etc.).
- Material UI component mapping for each screen (AppBar, Drawer, DataGrid/Table, Cards, Chips, Dialogs).
- Frontend remains fully usable with mock data when APIs fail.
"""


def architecture_system_addendum() -> str:
    return f"""
{STACK_SUMMARY}

Architecture JSON MUST lock:
- frontend_structure.framework = React
- frontend_structure.language = TypeScript
- frontend_structure.template = Vite + React + TypeScript
- frontend_structure.ui_library = Material UI (MUI)
- frontend_structure.routing = React Router
- frontend_structure.data_fetching = TanStack Query
- frontend_structure.charts = Recharts
- backend_structure.framework = FastAPI
- backend_structure.database = PostgreSQL
- backend_structure.orm = SQLAlchemy
- backend_structure.auth = JWT

Also include:
- screens[] with path, name, required API endpoints, mock fixtures
- api_contract.endpoints[] with method, path, auth role
- database_schema.tables[] with typed columns, FKs, indexes
- layers: presentation / API / business logic / database / AI services
- mock_strategy: frontend src/api/mock.ts used whenever fetch fails
- docker_compose: db + backend + frontend

Do NOT use SQLite as the production database. SQLite is only an optional local fallback when DATABASE_URL is not set.
Do NOT use Tailwind as the UI library. Use Material UI.
"""


def codegen_system_addendum() -> str:
    return f"""
{STACK_SUMMARY}

CODEGEN RULES:
1. Frontend is TypeScript + Vite + MUI. Pages live under frontend/src/pages/. Shared components under frontend/src/components/.
2. Use React Router for every module/screen in the PRD. Sidebar navigation must list all modules.
3. Use TanStack Query for server state. Charts use Recharts.
4. Theme (palette, typography, spacing) MUST come from the UI/UX color palette / design tokens. Apply via MUI createTheme in frontend/src/theme.ts.
5. API client (frontend/src/api/client.ts) MUST catch network/4xx/5xx and fall back to frontend/src/api/mock.ts so every screen still works.
6. Backend is FastAPI + SQLAlchemy + Pydantic + JWT. PostgreSQL via DATABASE_URL. SQLite fallback only if DATABASE_URL unset (for preview).
7. Implement all listed REST endpoints. Validate input. Swagger via FastAPI. Audit log writes for mutations.
8. Include docker-compose.yml, .env.example, README, seed.py, and Alembic migration stub.
9. Do not ship mock-only screens. Wire every screen to the API client (which itself uses mocks as fallback).
10. Roles: implement login/logout JWT and hide actions the role cannot perform.
"""
