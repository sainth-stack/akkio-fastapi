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

_FRONTEND_ONLY_MARKERS = (
    "no network",
    "no fetch",
    "no external data",
    "static and embedded",
    "content is static",
    "self-contained",
    "embedded static",
    "no network calls",
    "without external data",
)

_BASE_ALLOWLIST = (
    "frontend/src/App.jsx",
    "frontend/src/App.js",
    "frontend/src/styles/app.css",
    "frontend/src/styles.css",
)

_BACKEND_ALLOWLIST = (
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


def detect_frontend_only(requirement: str, prd: str = "", uiux: str = "") -> bool:
    text = f"{requirement} {prd} {uiux}".lower()
    return any(m in text for m in _FRONTEND_ONLY_MARKERS)


def detect_app_kind(requirement: str, prd: str = "", architecture: Optional[Dict[str, Any]] = None, uiux: str = "") -> str:
    text = f"{requirement} {prd}".lower()
    is_quiz = any(k in text for k in ("quiz", "mcq", "multiple choice", "multiple-choice", "question bank"))

    if is_quiz and detect_frontend_only(requirement, prd, uiux):
        return "static_quiz"

    if is_quiz:
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

    if detect_frontend_only(requirement, prd, uiux):
        return "static"

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
        "static_quiz": ("questions",),
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


def _mvp_tables(app_kind: str, all_tables: List[str], primary: str, frontend_only: bool) -> List[str]:
    if frontend_only:
        return []

    if app_kind in ("quiz", "static_quiz"):
        mvp = []
        for t in ("quizzes", "quiz", "questions", "options"):
            if t in all_tables:
                mvp.append(t)
        return mvp or ([primary] if primary else [])

    if app_kind == "crud":
        return [primary]

    extras = [t for t in all_tables if t != primary and t not in _SKIP_TABLES]
    return [primary] + extras[:2]


def _mvp_features(app_kind: str, primary: str, frontend_only: bool) -> List[str]:
    if app_kind == "static_quiz":
        return [
            "5 static MCQ questions embedded in React (from UI/UX spec)",
            "One question at a time with 4 radio options",
            "Immediate correct/incorrect feedback after selection",
            "Lock answer after selection; Next button to proceed",
            "Results screen with score and percentage",
            "Restart quiz button",
            "No API calls — useState + embedded QUESTIONS array",
        ]
    if app_kind == "static":
        return [
            "Self-contained UI from PRD — no backend API required",
            "Interactive components with React useState",
            "Responsive layout and accessible controls",
        ]
    if app_kind == "quiz":
        return [
            "Take quiz: one question at a time with radio options",
            "Submit answers and show score",
            "Use apiFetch for questions if backend is used",
        ]
    if app_kind == "content":
        return ["Input form", "Generate button calling backend", "Display results"]
    if app_kind == "form":
        return ["Text input", "Submit to backend API", "Display response"]
    if app_kind == "dashboard":
        return ["Fetch summary data from API", "Metric cards", "Loading states"]
    entity = singularize(primary)
    return [
        f"List {primary} from API",
        f"Create/update/delete {entity}",
        "Loading and error states",
    ]


def _screens_for_kind(app_kind: str) -> List[Dict[str, Any]]:
    if app_kind == "static_quiz":
        return [
            {"id": "question", "title": "Question Screen", "components": ["question", "options-radio", "feedback", "next"]},
            {"id": "results", "title": "Results", "components": ["score", "review", "restart"]},
        ]
    if app_kind == "quiz":
        return [
            {"id": "take-quiz", "title": "Take Quiz", "components": ["question-display", "radio-options", "submit-score"]},
        ]
    if app_kind in ("content", "form", "static"):
        return [{"id": "main", "title": "Main", "components": ["main-ui"]}]
    if app_kind == "dashboard":
        return [{"id": "dashboard", "title": "Dashboard", "components": ["metric-cards"]}]
    return [{"id": "main", "title": "Main", "components": ["list", "form", "actions"]}]


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

    return f"/{primary_table}" if primary_table else "/items"


def build_app_spec(
    requirement: str,
    architecture: Optional[Dict[str, Any]] = None,
    prd: str = "",
    uiux: str = "",
) -> Dict[str, Any]:
    arch = architecture or {}
    frontend_only = detect_frontend_only(requirement, prd, uiux)
    app_kind = detect_app_kind(requirement, prd, arch, uiux)
    all_tables = _table_names(arch)
    primary_table = _pick_primary_table(arch, app_kind)
    mvp_tables = _mvp_tables(app_kind, all_tables, primary_table, frontend_only)
    entity = singularize(primary_table) if primary_table else "item"
    primary_def = _table_def(arch, primary_table)

    entities = []
    for table_name in mvp_tables:
        table = _table_def(arch, table_name)
        entities.append({
            "name": singularize(table_name).title(),
            "table_name": table_name,
            "fields": _fields_from_table(table),
        })

    title = "Quiz"
    if "quiz" in requirement.lower():
        title = "Quiz"
    m = re.search(r"^(?:create|build)\s+(.+?)(?:\s+app)?$", requirement.strip(), re.I)
    if m:
        title = m.group(1).strip().title()[:40]

    return {
        "app_kind": app_kind,
        "frontend_only": frontend_only,
        "requirement": requirement,
        "title": title,
        "primary_entity": entity,
        "primary_table": primary_table,
        "api_prefix": _api_prefix(arch, primary_table),
        "mvp_tables": mvp_tables,
        "mvp_features": _mvp_features(app_kind, primary_table, frontend_only),
        "screens": _screens_for_kind(app_kind),
        "entities": entities,
        "fields": _fields_from_table(primary_def),
        "deferred": [
            "user authentication",
            "sharing / collaboration",
            "offline sync beyond localStorage",
        ],
    }


def get_codegen_allowlist(app_spec: Dict[str, Any]) -> tuple[str, ...]:
    allowlist = list(_BASE_ALLOWLIST)
    if not app_spec.get("frontend_only"):
        allowlist.extend(_BACKEND_ALLOWLIST)
    kind = app_spec.get("app_kind", "custom")
    if kind in ("quiz", "static_quiz", "dashboard", "custom"):
        allowlist.append(_COMPONENT_ALLOWLIST_PREFIX)
    return tuple(allowlist)


def is_path_allowlisted(path: str, allowlist: tuple[str, ...]) -> bool:
    norm = path.replace("\\", "/").lstrip("/")
    return any(norm == p or norm.startswith(p) for p in allowlist)


def build_codegen_system_prompt(app_spec: Dict[str, Any], uiux: str, architecture: Dict[str, Any]) -> str:
    spec_json = json.dumps(app_spec, indent=2)
    allowlist = get_codegen_allowlist(app_spec)
    files_list = "\n".join(f"- {p}" for p in allowlist if not p.endswith("/"))
    if any(p.endswith("/") for p in allowlist):
        files_list += "\n- frontend/src/components/*.jsx (optional, for larger UIs)"

    kind = app_spec.get("app_kind", "custom")
    frontend_only = app_spec.get("frontend_only", False)
    mvp = "\n".join(f"- {f}" for f in app_spec.get("mvp_features", []))

    output_format = """
**OUTPUT FORMAT (required):**
For EACH file, output exactly:
FILE: frontend/src/App.jsx
```jsx
... full file content ...
```
Do NOT skip files. Do NOT use placeholders or "..." omissions.
"""

    if kind == "static_quiz":
        backend_rules = "Backend is optional — minimal empty router is fine. Focus 100% on frontend."
        api_rule = "Do NOT use apiFetch — embed QUESTIONS array in App.jsx from the UI/UX spec."
    elif frontend_only:
        backend_rules = "Skip heavy backend — minimal routes.py with /info only."
        api_rule = "Prefer embedded static data; apiFetch optional only if PRD requires it."
    else:
        backend_rules = f"""
2. models.py: SQLAlchemy models for: {", ".join(app_spec.get("mvp_tables", [])) or "primary entity"}
3. schemas.py: Pydantic v2 (model_config = ConfigDict(from_attributes=True))
4. routes.py: APIRouter prefix `{app_spec.get("api_prefix", "/items")}` with CRUD"""
        api_rule = (
            "Use `import { apiFetch } from './api/client.js'`. "
            "Call as apiFetch('/tasks', { method: 'POST', body: JSON.stringify(data) }). "
            "NEVER apiFetch(path, 'POST', data). "
            f"Collection path must match primary table: /{app_spec.get('primary_table', 'items')}."
        )

    return f"""You are an expert React + FastAPI developer. Customize the generic Vite shell into a working app.

**FROZEN (never generate):** package.json, vite.config.js, index.html, main.jsx, api/client.js, styles/base.css, backend/main.py, backend/database.py, backend/requirements.txt

**GENERATE (FILE: path format):**
{files_list}

**APP KIND:** {kind}
**FRONTEND ONLY:** {frontend_only}

**APP SPEC:**
```json
{spec_json}
```

**MVP (implement all):**
{mvp}

{output_format}

**UI/UX (colors & layout — plain CSS in styles/app.css, NOT Tailwind):**
{uiux[:5000] if uiux else "Clean modern SaaS UI with soft gray background (#f8fafc), white cards, primary accent."}

**PREMIUM UI STRUCTURE (App.jsx — follow exactly):**
```jsx
export default function App() {{
  return (
    <div className="app">
      <div className="app-container">
        <header className="app-header">
          <h1 className="app-title">App Name</h1>
          <p className="app-subtitle">Short description</p>
        </header>
        <div className="card">
          <form className="form-row" onSubmit={{handler}}>
            <input className="input" placeholder="..." aria-label="..." />
            <button type="submit" className="btn btn-primary">Add</button>
          </form>
          <ul className="list">
            <li className="list-item">...</li>
          </ul>
        </div>
      </div>
    </div>
  );
}}
```
Platform injects premium CSS — you MUST use these class names. Never bare unstyled inputs/buttons.

**RULES:**
1. App.jsx: premium SaaS UI with app-container, app-header, card, form-row, input, btn btn-primary, list, list-item.
{backend_rules}
5. styles/app.css: do NOT generate — platform injects premium themed CSS automatically.
6. {api_rule}
7. useState/useEffect only — no external state libraries.
8. Accessible: aria-labels on inputs, keyboard-friendly controls.
9. Never use inline styles for layout/colors — use the CSS class contract above.
"""


def validate_against_app_spec(files: Dict[str, str], app_spec: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    kind = app_spec.get("app_kind", "custom")
    frontend_only = app_spec.get("frontend_only", False)
    prefix = app_spec.get("api_prefix", "/items")

    app_src = files.get("frontend/src/App.jsx") or files.get("frontend/src/App.js", "")
    components_src = "".join(
        files.get(p, "") for p in files if p.startswith("frontend/src/components/")
    )
    ui_src = app_src + components_src

    if "waiting for application ui from code generation" in ui_src.lower():
        errors.append("App.jsx is still the generic shell")
    if "shell-notice" in ui_src.lower():
        errors.append("App still shows generic shell notice")

    if not any(k in ui_src for k in ("onClick", "onSubmit", "<button", "<input", "useState")):
        errors.append("Frontend missing interactive UI")

    stub_markers = ("minimal app component", "satisfy the build", "placeholder app")
    if any(m in ui_src.lower() for m in stub_markers):
        errors.append("Frontend contains stub text")

    css = files.get("frontend/src/styles/app.css", "")
    if len(css.splitlines()) < 25 and "Akkio Premium SaaS" not in css and "Akkio SaaS" not in css:
        errors.append("styles/app.css too minimal (< 25 lines)")

    app_lower = app_src.lower()
    premium_markers = ("app-container", "app-header", "app-title", "card", "form-row", "btn-primary", "list-item")
    if app_src and sum(1 for m in premium_markers if m in app_src) < 3:
        errors.append(
            "App.jsx missing premium layout classes (need app-container, app-header, card, form-row, btn btn-primary, list-item)"
        )
    if app_src and '<input' in app_lower and 'className="input"' not in app_src and "className='input'" not in app_src:
        if 'className={`input' not in app_src:
            errors.append('Inputs must use className="input"')
    if app_src and '<button' in app_lower and 'btn' not in app_src:
        errors.append('Buttons must use className="btn btn-primary" or btn-ghost')

    if kind == "static_quiz":
        combined = ui_src.lower()
        if "question" not in combined:
            errors.append("Quiz missing questions in UI")
        if not any(k in combined for k in ('type="radio"', "role=\"radio\"", "radio", "option-row", "options")):
            errors.append("Quiz missing option selection UI")
        if "score" not in combined and "result" not in combined:
            errors.append("Quiz missing results/score screen")
        return errors

    if frontend_only:
        return errors

    routes = files.get("backend/routes.py", "")
    if not routes.strip() or "APIRouter" not in routes:
        errors.append("backend/routes.py missing APIRouter")
    elif prefix not in routes and f'"{prefix}"' not in routes and f"'{prefix}'" not in routes:
        if kind not in ("static", "custom"):
            errors.append(f"backend/routes.py missing API prefix {prefix}")

    models = files.get("backend/models.py", "")
    if "class " not in models and kind in ("quiz", "crud"):
        errors.append("backend/models.py missing SQLAlchemy models")

    schemas = files.get("backend/schemas.py", "")
    if "BaseModel" not in schemas and kind in ("quiz", "crud"):
        errors.append("backend/schemas.py missing Pydantic schemas")

    if "apiFetch" not in ui_src and "fetch(" not in ui_src and kind in ("quiz", "crud", "content", "form", "dashboard"):
        errors.append("Frontend does not call the API")

    if "apiFetch" in ui_src and re.search(
        r"apiFetch\([^,]+,\s*['\"](GET|POST|PUT|PATCH|DELETE)['\"]",
        ui_src,
        re.I,
    ):
        errors.append("apiFetch uses wrong signature (use options object, not method string)")

    if kind == "quiz":
        combined = (models + schemas + ui_src).lower()
        if "question" not in combined:
            errors.append("Quiz missing questions")
        if not any(k in ui_src.lower() for k in ('type="radio"', "radio", "option")):
            errors.append("Quiz missing multiple-choice UI")

    return errors
