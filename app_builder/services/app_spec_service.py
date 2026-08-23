"""
App Spec — machine-readable product definition derived from PRD, architecture, and requirement.
Drives codegen prompts, dynamic allowlists, and validation (any app type).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_SKIP_TABLES = frozenset({
    "users", "user", "sessions", "session", "auth", "password_resets",
})

_BASE_ALLOWLIST = (
    "frontend/src/App.jsx",
    "frontend/src/App.js",
    "frontend/src/styles/app.css",
    "frontend/src/styles.css",
    "backend/models.py",
    "backend/schemas.py",
    "backend/routes.py",
)

_COMPONENT_ALLOWLIST_PREFIX = "frontend/src/components/"


def singularize(table_name: str) -> str:
    name = (table_name or "item").lower().strip()
    special = {"quizzes": "quiz", "categories": "category", "people": "person", "children": "child"}
    if name in special:
        return special[name]
    if name.endswith("ies"):
        return name[:-3] + "y"
    if name.endswith("ses") and not name.endswith("sses"):
        return name[:-2]
    if name.endswith("s") and not name.endswith("ss"):
        return name[:-1]
    return name


def detect_app_kind(requirement: str, prd: str = "", architecture: Optional[Dict[str, Any]] = None) -> str:
    text = f"{requirement} {prd}".lower()
    arch_text = json.dumps(architecture or {}).lower()

    if any(k in text for k in ("quiz", "mcq", "multiple choice", "multiple-choice", "question bank")):
        return "quiz"
    if any(k in text for k in ("todo", "to-do", "task list", "checklist", "task manager")):
        return "crud"
    if any(k in text for k in ("translator", "translate", "translation")):
        return "form"
    if any(k in text for k in ("idea", "generator", "linkedin", "blog post", "content generator")):
        return "content"
    if any(k in text for k in ("dashboard", "analytics", "metrics", "chart")):
        return "dashboard"
    if any(k in text for k in ("travel", "trip", "itinerary", "planner")):
        return "crud"
    if "quiz" in arch_text or "questions" in arch_text and "options" in arch_text:
        return "quiz"
    return "custom"


def _table_names(architecture: Dict[str, Any]) -> List[str]:
    tables = (architecture.get("database_schema") or {}).get("tables") or []
    names: List[str] = []
    for table in tables:
        if isinstance(table, dict):
            name = (table.get("table_name") or table.get("name") or "").lower().strip()
            if name:
                names.append(name)
    return names


def _table_def(architecture: Dict[str, Any], table_name: str) -> Dict[str, Any]:
    tables = (architecture.get("database_schema") or {}).get("tables") or []
    for table in tables:
        if isinstance(table, dict):
            name = (table.get("table_name") or table.get("name") or "").lower()
            if name == table_name:
                return table
    return {}


def _fields_from_table(table: Dict[str, Any]) -> List[str]:
    cols = table.get("columns") or table.get("fields") or []
    names: List[str] = []
    for col in cols:
        if isinstance(col, dict):
            names.append(col.get("name", ""))
        elif isinstance(col, str):
            names.append(col)
    return [n for n in names if n]


def _pick_primary_table(architecture: Dict[str, Any], app_kind: str) -> str:
    names = _table_names(architecture)
    if not names:
        return "items"

    kind_primary = {
        "quiz": ("quizzes", "quiz", "questions"),
        "crud": ("tasks", "todos", "items", "lists", "trips", "entries"),
        "content": ("ideas", "posts", "content"),
        "form": ("translations", "messages"),
    }
    for candidate in kind_primary.get(app_kind, ()):
        if candidate in names:
            return candidate

    for name in names:
        if name not in _SKIP_TABLES:
            return name
    return names[0]


def _mvp_tables(app_kind: str, all_tables: List[str], primary: str) -> List[str]:
    if app_kind == "quiz":
        mvp = []
        for t in ("quizzes", "quiz", "questions", "options"):
            if t in all_tables:
                mvp.append(t)
        return mvp or [primary]

    if app_kind == "crud":
        return [primary]

    # custom: primary + one child table if present
    extras = [t for t in all_tables if t != primary and t not in _SKIP_TABLES]
    return [primary] + extras[:2]


def _mvp_features(app_kind: str, primary: str, mvp_tables: List[str]) -> List[str]:
    entity = singularize(primary)
    if app_kind == "quiz":
        return [
            "List quizzes",
            "Create quiz with title",
            "Add multiple-choice questions with options",
            "Take quiz: show one question at a time with radio options",
            "Submit answers and show score",
        ]
    if app_kind == "content":
        return [
            "Input form for topic/prompt",
            "Generate button calling backend",
            "Display generated results",
        ]
    if app_kind == "form":
        return [
            "Input area for user text",
            "Submit to backend API",
            "Display translated/generated output",
        ]
    if app_kind == "dashboard":
        return [
            "Fetch summary data from API",
            "Display metric cards or charts area",
            "Loading and empty states",
        ]
    return [
        f"List {primary} from API",
        f"Create new {entity}",
        f"Update and delete {entity}",
        "Responsive layout with loading/error states",
    ]


def _screens_for_kind(app_kind: str) -> List[Dict[str, Any]]:
    if app_kind == "quiz":
        return [
            {"id": "quiz-list", "title": "Quiz List", "components": ["quiz-list", "create-quiz"]},
            {"id": "edit-quiz", "title": "Edit Quiz", "components": ["question-form", "options-editor"]},
            {"id": "take-quiz", "title": "Take Quiz", "components": ["question-display", "radio-options", "submit-score"]},
        ]
    if app_kind in ("content", "form"):
        return [{"id": "main", "title": "Main", "components": ["input-form", "submit-button", "results"]}]
    if app_kind == "dashboard":
        return [{"id": "dashboard", "title": "Dashboard", "components": ["metric-cards", "data-panel"]}]
    return [{"id": "main", "title": "Main", "components": ["list", "create-form", "actions"]}]


def _api_prefix(architecture: Dict[str, Any], primary_table: str) -> str:
    api_contract = architecture.get("api_contract") or {}
    if isinstance(api_contract, str):
        try:
            api_contract = json.loads(api_contract)
        except json.JSONDecodeError:
            api_contract = {}

    endpoints = api_contract.get("endpoints") or {}
    if isinstance(endpoints, dict):
        for path in endpoints:
            m = re.match(r"^/?([\w-]+)", str(path).lstrip("/"))
            if m and m.group(1) not in _SKIP_TABLES:
                return f"/{m.group(1)}"

    routes = api_contract.get("routes") or []
    if routes and isinstance(routes[0], dict):
        path = routes[0].get("path") or routes[0].get("url") or ""
        m = re.match(r"^/?([\w-]+)", str(path).lstrip("/"))
        if m and m.group(1) not in _SKIP_TABLES:
            return f"/{m.group(1)}"

    return f"/{primary_table}"


def build_app_spec(
    requirement: str,
    architecture: Optional[Dict[str, Any]] = None,
    prd: str = "",
    uiux: str = "",
) -> Dict[str, Any]:
    arch = architecture or {}
    app_kind = detect_app_kind(requirement, prd, arch)
    all_tables = _table_names(arch)
    primary_table = _pick_primary_table(arch, app_kind)
    mvp_tables = _mvp_tables(app_kind, all_tables, primary_table)
    entity = singularize(primary_table)
    primary_def = _table_def(arch, primary_table)

    entities = []
    for table_name in mvp_tables:
        table = _table_def(arch, table_name)
        entities.append({
            "name": singularize(table_name).title(),
            "table_name": table_name,
            "fields": _fields_from_table(table),
        })

    spec: Dict[str, Any] = {
        "app_kind": app_kind,
        "requirement": requirement,
        "primary_entity": entity,
        "primary_table": primary_table,
        "api_prefix": _api_prefix(arch, primary_table),
        "mvp_tables": mvp_tables,
        "mvp_features": _mvp_features(app_kind, primary_table, mvp_tables),
        "screens": _screens_for_kind(app_kind),
        "entities": entities,
        "fields": _fields_from_table(primary_def),
        "deferred": [
            "user authentication",
            "sharing / collaboration",
            "offline sync",
            "import/export",
            "push notifications",
        ],
    }
    return spec


def get_codegen_allowlist(app_spec: Dict[str, Any]) -> tuple[str, ...]:
    """Files the LLM may generate. Adds components/ for multi-screen apps."""
    allowlist = list(_BASE_ALLOWLIST)
    if app_spec.get("app_kind") in ("quiz", "dashboard", "custom"):
        if app_spec.get("screens") and len(app_spec["screens"]) > 1:
            allowlist.append(_COMPONENT_ALLOWLIST_PREFIX)
    return tuple(allowlist)


def is_path_allowlisted(path: str, allowlist: tuple[str, ...]) -> bool:
    norm = path.replace("\\", "/").lstrip("/")
    return any(norm == p or norm.startswith(p) for p in allowlist)


def build_codegen_system_prompt(app_spec: Dict[str, Any], uiux: str, architecture: Dict[str, Any]) -> str:
    spec_json = json.dumps(app_spec, indent=2)
    arch_json = json.dumps(architecture, indent=2)
    allowlist = get_codegen_allowlist(app_spec)
    files_list = "\n".join(f"- {p}" for p in allowlist if not p.endswith("/"))
    if any(p.endswith("/") for p in allowlist):
        files_list += "\n- frontend/src/components/*.jsx (as needed for screens)"

    kind = app_spec.get("app_kind", "custom")
    mvp = "\n".join(f"- {f}" for f in app_spec.get("mvp_features", []))
    screens = json.dumps(app_spec.get("screens", []), indent=2)

    kind_hints = {
        "quiz": (
            "Build a QUIZ app: quizzes, questions, multiple-choice options, take-quiz flow with "
            "radio buttons, score on submit. Use apiFetch for all API calls."
        ),
        "crud": (
            "Build a CRUD app for the primary entity: list, create form, edit/delete actions."
        ),
        "content": (
            "Build a content generator: input, generate button, results list. "
            "Call backend POST endpoint for generation."
        ),
        "form": (
            "Build a form app: text input, submit, display API response."
        ),
        "dashboard": (
            "Build a dashboard: fetch data from API, show cards/summary, loading states."
        ),
        "custom": (
            "Build the app described in the requirement and App Spec — match the domain exactly."
        ),
    }

    return f"""You are an expert full-stack developer customizing a **generic Vite + React + FastAPI shell**.

The base template has NO domain features — you must generate the full application from the App Spec.

**FROZEN (do not generate):** package.json, vite.config.js, index.html, main.jsx, api/client.js, styles/base.css, backend/main.py, backend/database.py, backend/requirements.txt

**GENERATE these files (FILE: path format):**
{files_list}

**APP KIND:** {kind}
{kind_hints.get(kind, kind_hints["custom"])}

**APP SPEC (source of truth):**
```json
{spec_json}
```

**MVP FEATURES (implement all):**
{mvp}

**SCREENS:**
```json
{screens}
```

**API:** Primary prefix `{app_spec.get("api_prefix", "/items")}` — use `import {{ apiFetch }} from './api/client.js'`

**UI/UX (translate to plain CSS variables in styles/app.css):**
{uiux[:4500] if uiux else "Clean modern UI — use :root CSS variables for colors and spacing."}

**RULES:**
1. Generate COMPLETE files — never stubs, placeholders, or empty backend files.
2. models.py: SQLAlchemy models for MVP tables: {", ".join(app_spec.get("mvp_tables", []))}
3. schemas.py: Pydantic v2 (model_config = ConfigDict(from_attributes=True))
4. routes.py: FastAPI APIRouter with CRUD + quiz-specific endpoints as needed
5. App.jsx: full UI for `{kind}` app — may import from ./components/*.jsx
6. styles/app.css: 80+ lines, plain CSS only (no Tailwind)
7. Array safety: `(data || []).map(...)`, useState([]) defaults
8. localStorage optional fallback with try/catch around apiFetch
9. Do NOT build deferred features: {", ".join(app_spec.get("deferred", [])[:4])}

**Architecture reference:**
```json
{arch_json[:6000]}
```
"""


def validate_against_app_spec(files: Dict[str, str], app_spec: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    kind = app_spec.get("app_kind", "custom")
    prefix = app_spec.get("api_prefix", "/items")

    routes = files.get("backend/routes.py", "")
    if not routes.strip() or "APIRouter" not in routes:
        errors.append("backend/routes.py missing APIRouter")
    elif prefix not in routes and f'"{prefix}"' not in routes and f"'{prefix}'" not in routes:
        errors.append(f"backend/routes.py missing API prefix {prefix}")

    models = files.get("backend/models.py", "")
    if "class " not in models:
        errors.append("backend/models.py missing SQLAlchemy model classes")

    schemas = files.get("backend/schemas.py", "")
    if "BaseModel" not in schemas:
        errors.append("backend/schemas.py missing Pydantic schemas")

    app_src = files.get("frontend/src/App.jsx") or files.get("frontend/src/App.js", "")
    components_src = "".join(
        files.get(p, "") for p in files if p.startswith("frontend/src/components/")
    )
    ui_src = app_src + components_src

    if "waiting for application ui from code generation" in ui_src.lower():
        errors.append("App.jsx is still the generic shell — codegen did not customize UI")
    if "shell-notice" in ui_src.lower() and kind != "custom":
        errors.append("App still shows generic shell notice")

    if "apiFetch" not in ui_src and "fetch(" not in ui_src:
        errors.append("Frontend does not call the API (apiFetch)")

    if not any(k in ui_src for k in ("onSubmit", "onClick", "<input", "<button", "<form")):
        errors.append("Frontend missing interactive UI")

    stub_markers = ("minimal app component", "satisfy the build", "placeholder app")
    if any(m in ui_src.lower() for m in stub_markers):
        errors.append("Frontend contains stub/placeholder text")

    if kind == "quiz":
        combined = (models + schemas + ui_src).lower()
        if "question" not in combined:
            errors.append("Quiz app missing questions in models or UI")
        if "option" not in combined:
            errors.append("Quiz app missing options in models or UI")
        if not any(k in ui_src.lower() for k in ('type="radio"', "radio", "choice", "answer")):
            errors.append("Quiz UI missing multiple-choice selection (radio/options)")

    css = files.get("frontend/src/styles/app.css", "")
    if len(css.splitlines()) < 35:
        errors.append("styles/app.css too minimal (< 35 lines)")

    return errors
