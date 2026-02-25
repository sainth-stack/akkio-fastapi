"""
Dynamic Code Generator Agent - Generates code based on PRD and implementation plan using LLM
"""
from typing import Dict, Any, AsyncGenerator
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage

VITE_REACT_PLUGIN = "@vitejs/plugin-react"

# Minimal CRA package.json - Node 20+, with Tailwind
_CRA_PACKAGE_JSON = {
    "name": "frontend",
    "version": "1.0.0",
    "private": True,
    "engines": {"node": ">=20"},
    "dependencies": {
        "react": "^18.2.0",
        "react-dom": "^18.2.0",
        "react-scripts": "5.0.1",
    },
    "scripts": {
        "start": "NODE_OPTIONS=--openssl-legacy-provider react-scripts start",
        "build": "NODE_OPTIONS=--openssl-legacy-provider react-scripts build",
    },
    "devDependencies": {
        "tailwindcss": "latest",
        "postcss": "latest",
        "autoprefixer": "latest"
    },
    "eslintConfig": {"extends": ["react-app"]},
    "browserslist": {
        "production": [">0.2%", "not dead", "not op_mini all"],
        "development": ["last 1 chrome version", "last 1 firefox version", "last 1 safari version"],
    },
}


def _normalize_frontend_package_json_to_cra(files: Dict[str, str]) -> None:
    """
    If generated frontend uses Vite or problematic deps (@dnd-kit, etc.), replace
    with minimal Create React App (react-scripts) package.json so the app runs.
    Remove vite.config and ensure CRA entry files (public/index.html, src/index.js) exist.
    """
    pkg_path = "frontend/package.json"
    if pkg_path not in files:
        return
    try:
        pkg = json.loads(files[pkg_path])
        scripts = pkg.get("scripts") or {}
        deps = pkg.get("dependencies") or {}
        dev_deps = pkg.get("devDependencies") or {}
        has_vite = "vite" in str(scripts).lower() or "vite" in str(dev_deps).lower()
        has_dnd = any(k.startswith("@dnd-kit") for k in list(deps.keys()) + list(dev_deps.keys()))
        if not has_vite and not has_dnd:
            return
        cra = dict(_CRA_PACKAGE_JSON)
        cra["name"] = pkg.get("name", "frontend")
        files[pkg_path] = json.dumps(cra, indent=2)
        for vpath in ("frontend/vite.config.js", "frontend/vite.config.ts"):
            if vpath in files:
                del files[vpath]
        # Ensure CRA entry points exist (react-scripts expects public/index.html, src/index.js)
        if "frontend/public/index.html" not in files and "frontend/index.html" in files:
            files["frontend/public/index.html"] = files["frontend/index.html"]
        if "frontend/public/index.html" not in files:
            files["frontend/public/index.html"] = (
                "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"/>"
                "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>"
                "<title>React App</title></head><body><noscript>You need to enable JavaScript.</noscript>"
                "<div id=\"root\"></div></body></html>"
            )
        if "frontend/src/index.js" not in files and "frontend/src/index.jsx" not in files:
            files["frontend/src/index.js"] = (
                "import React from 'react';\nimport ReactDOM from 'react-dom/client';\n"
                "import App from './App';\nimport './styles.css';\n"
                "const root = ReactDOM.createRoot(document.getElementById('root'));\n"
                "root.render(<React.StrictMode><App /></React.StrictMode>);"
            )
        # Ensure index.js exists (CRA entry point)
        for f in ("frontend/tailwind.config.js", "frontend/postcss.config.js"):
            if f in files:
                del files[f]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass


_DEFAULT_STYLES_CSS = """/* App styles - modern primitives, colors, layout */
:root {
  --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
  --color-bg: #f8fafc;
  --color-surface: #ffffff;
  --color-text: #0f172a;
  --color-text-muted: #64748b;
  --color-primary: #6366f1;
  --color-primary-hover: #4f46e5;
  --color-border: #e2e8f0;
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
}

* { box-sizing: border-box; -webkit-font-smoothing: antialiased; }
body { margin: 0; font-family: var(--font-sans); background: var(--color-bg); color: var(--color-text); line-height: 1.5; }

.app { min-height: 100vh; padding: 2rem 1rem; }
.app-container { max-width: 48rem; margin: 0 auto; }
.header { margin-bottom: 2.5rem; text-align: center; }
.app-title { font-size: 2.25rem; font-weight: 700; color: #1e293b; letter-spacing: -0.025em; margin: 0 0 0.5rem 0; }
.app-subtitle { color: var(--color-text-muted); font-size: 1.125rem; }

.card { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); box-shadow: var(--shadow-md); padding: 1.5rem; margin-bottom: 2rem; }

.form-row { display: flex; flex-direction: row; gap: 0.75rem; margin-bottom: 1.5rem; }
.input { flex: 1; padding: 0.625rem 1rem; border: 1px solid var(--color-border); border-radius: var(--radius-md); font-family: inherit; font-size: 1rem; transition: all 0.2s; box-shadow: var(--shadow-sm); }
.input:focus { outline: none; border-color: var(--color-primary); box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2); }

.btn { display: inline-flex; align-items: center; justify-content: center; padding: 0.625rem 1.25rem; border-radius: var(--radius-md); font-weight: 600; font-family: inherit; cursor: pointer; border: none; transition: all 0.2s; font-size: 0.875rem; }
.btn-primary { background: var(--color-primary); color: white; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); }
.btn-primary:hover:not(:disabled) { background: var(--color-primary-hover); transform: translateY(-1px); box-shadow: var(--shadow-md); }
.btn-primary:active { transform: translateY(0); }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

.list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.75rem; }
.list-item { display: flex; align-items: center; justify-content: space-between; padding: 1rem 1.25rem; background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); transition: all 0.2s; box-shadow: var(--shadow-sm); }
.list-item:hover { transform: translateX(4px); border-color: var(--color-primary); box-shadow: var(--shadow-md); }
.list-item strong { font-weight: 600; color: #1e293b; }
.list-item .muted { color: var(--color-text-muted); font-size: 0.875rem; }
"""

_TODO_APP_PREMIUM_CSS = """
/* Premium Todo Specific Styles */
.todo-card {
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.todo-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: white;
  border-radius: var(--radius-md);
  margin-bottom: 0.5rem;
  transition: all 0.2s ease;
  border: 1px solid var(--color-border);
}

.todo-item:hover {
  border-color: var(--color-primary);
  box-shadow: var(--shadow-md);
  transform: scale(1.01);
}

.todo-checkbox {
  width: 1.25rem;
  height: 1.25rem;
  border-radius: 50%;
  border: 2px solid var(--color-border);
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.todo-checkbox.checked {
  background: var(--color-primary);
  border-color: var(--color-primary);
}

.todo-checkbox.checked::after {
  content: '✓';
  color: white;
  font-size: 0.75rem;
}

.todo-text {
  flex: 1;
  font-size: 1rem;
  color: var(--color-text);
  transition: all 0.2s;
}

.todo-text.completed {
  text-decoration: line-through;
  color: var(--color-text-muted);
}

.todo-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 600;
  background: #f1f5f9;
  color: #475569;
}

.todo-badge-high { background: #fee2e2; color: #dc2626; }
.todo-badge-medium { background: #fef3c7; color: #d97706; }
.todo-badge-low { background: #dcfce7; color: #16a34a; }
"""


def _detect_todo_app(requirement: str) -> bool:
    """Check if the requirement refers to a todo list or task manager."""
    keywords = ["todo", "to-do", "task", "checklist", "inventory", "list app"]
    req_lower = requirement.lower()
    return any(kw in req_lower for kw in keywords)


def _ensure_node_compat(files: Dict[str, str]) -> None:
    """
    Ensure frontend works with Node 20+.
    - Add NODE_OPTIONS=--openssl-legacy-provider to react-scripts start/build
    """
    pkg_path = "frontend/package.json"
    if pkg_path in files:
        try:
            pkg = json.loads(files[pkg_path])
            pkg.setdefault("engines", {"node": ">=20"})
            scripts = pkg.get("scripts") or {}
            # Fix Node 20+ OpenSSL compatibility for react-scripts
            for key in ("start", "build"):
                if key in scripts and "react-scripts" in str(scripts[key]) and "NODE_OPTIONS" not in str(scripts[key]):
                    scripts[key] = f"NODE_OPTIONS=--openssl-legacy-provider {scripts[key]}"
            pkg["scripts"] = scripts
            files[pkg_path] = json.dumps(pkg, indent=2)
        except (json.JSONDecodeError, TypeError):
            pass


# Core backend packages for FastAPI + SQLite (no version numbers, no external DB)
_BACKEND_CORE_PACKAGES = ["fastapi", "uvicorn", "sqlalchemy", "pydantic"]

# PyPI packages to NEVER add - usually local modules in generated apps (services, etc.)
_BACKEND_BLOCKLIST = {"services", "motor"}


def _normalize_backend_requirements(files: Dict[str, str]) -> None:
    """
    Normalize backend/requirements.txt: use package names ONLY (no version pins)
    so pip install works flexibly across environments. Ensure core packages are present.
    """
    path = "backend/requirements.txt"
    if path not in files:
        files[path] = "\n".join(_BACKEND_CORE_PACKAGES) + "\n"
        return
    lines = []
    seen = set()
    for raw in files[path].strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            lines.append(line)
            continue
        # Strip version: package, package==x, package>=x, package~=x, package[extra] -> package
        pkg = re.split(r'[=<>~!\[]', line)[0].strip().lower()
        if pkg and pkg not in seen and pkg not in _BACKEND_BLOCKLIST:
            seen.add(pkg)
            lines.append(pkg)
    # Ensure core packages exist
    for pkg in _BACKEND_CORE_PACKAGES:
        if pkg not in seen:
            lines.append(pkg)
    files[path] = "\n".join(lines) + "\n"


def _ensure_sqlite_database_default(files: Dict[str, str]) -> None:
    """
    Ensure backend/database.py uses SQLite by default - no external credentials needed.
    Generated apps run end-to-end out of the box. Set DATABASE_URL env for PostgreSQL in production.
    """
    path = "backend/database.py"
    if path not in files:
        return
    content = files[path]
    # If postgresql is the default URL, replace with SQLite so app runs without setup
    if "postgresql" in content.lower() and 'getenv' in content.lower():
        content = re.sub(
            r'getenv\s*\(\s*["\']DATABASE_URL["\']\s*,\s*["\']postgresql[^"\']*["\']\s*\)',
            'getenv("DATABASE_URL", "sqlite:///./app.db")',
            content,
            flags=re.IGNORECASE,
        )
        # Ensure connect_args for SQLite (remove if it was postgres-only)
        if "check_same_thread" not in content and "sqlite" in content.lower():
            content = re.sub(
                r'(create_engine\s*\(\s*\w+)',
                r'\1,\n    connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {}',
                content,
                count=1,
            )
        files[path] = content


def _fix_sqlalchemy_uuid_imports(files: Dict[str, str]) -> None:
    """
    Fix incorrect UUID imports for SQLite compatibility. sqlite dialect has Uuid not UUID;
    postgresql UUID doesn't work with SQLite. Use sqlalchemy.types.Uuid which works everywhere.
    """
    for path in list(files.keys()):
        if not (path.startswith("backend/") and path.endswith(".py")):
            continue
        content = files[path]
        content = content.replace(
            "from sqlalchemy.dialects.sqlite import UUID",
            "from sqlalchemy.types import Uuid as UUID",
        )
        content = content.replace(
            "from sqlalchemy.dialects.postgresql import UUID",
            "from sqlalchemy.types import Uuid as UUID",
        )
        if content != files[path]:
            files[path] = content


def _fix_backend_pydantic_and_common(files: Dict[str, str]) -> None:
    """
    Fix common backend issues so generated apps run end-to-end without errors.
    - Pydantic v2: orm_mode -> from_attributes, .dict() -> .model_dump()
    - PostgreSQL-only types when using SQLite: JSONB -> JSON, etc.
    """
    for path in list(files.keys()):
        if not (path.startswith("backend/") and path.endswith(".py")):
            continue
        content = files[path]
        changed = False

        # Pydantic v2 compatibility
        if "orm_mode = True" in content:
            content = content.replace("orm_mode = True", "from_attributes = True")
            changed = True
        if "orm_mode=True" in content:
            content = content.replace("orm_mode=True", "from_attributes=True")
            changed = True

        # .dict() is deprecated in Pydantic v2 - use .model_dump()
        if ".dict()" in content and "model_dump" not in content:
            content = content.replace(".dict()", ".model_dump()")
            changed = True

        # PostgreSQL JSONB/ARRAY don't work with SQLite - use generic types
        if "from sqlalchemy.dialects.postgresql import JSONB" in content:
            content = content.replace(
                "from sqlalchemy.dialects.postgresql import JSONB",
                "from sqlalchemy import JSON",
            )
            content = re.sub(r"\bJSONB\b", "JSON", content)
            changed = True

        if changed:
            files[path] = content


def _ensure_cors_in_backend(files: Dict[str, str]) -> None:
    """
    Ensure generated backend main.py has CORS middleware allowing all origins.
    Required for frontend-backend communication when served from different origins.
    """
    main_path = "backend/main.py"
    if main_path not in files:
        return
    content = files[main_path]
    if "CORSMiddleware" in content:
        # CORS exists - ensure it allows all origins
        if 'allow_origins=["*"]' not in content:
            content = re.sub(
                r'allow_origins\s*=\s*\[[^\]]*\]',
                'allow_origins=["*"]',
                content
            )
        if 'allow_methods=["*"]' not in content:
            content = re.sub(
                r'allow_methods\s*=\s*\[[^\]]*\]',
                'allow_methods=["*"]',
                content
            )
        if 'allow_headers=["*"]' not in content:
            content = re.sub(
                r'allow_headers\s*=\s*\[[^\]]*\]',
                'allow_headers=["*"]',
                content
            )
        content = re.sub(
            r'allow_credentials\s*=\s*True',
            'allow_credentials=False',
            content
        )
        files[main_path] = content
        return
    # No CORS - inject after app = FastAPI()
    cors_block = '''
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
'''
    if "app = FastAPI()" in content:
        content = content.replace("app = FastAPI()", "app = FastAPI()" + cors_block, 1)
    elif "app=FastAPI()" in content:
        content = content.replace("app=FastAPI()", "app=FastAPI()" + cors_block, 1)
    else:
        return
    files[main_path] = content


def _ensure_database_tables_created(files: Dict[str, str]) -> None:
    """
    Ensure backend creates database tables on startup. Add create_all if missing.
    Tables are created automatically - no manual migration or install needed.
    Prevents 'no such table' OperationalError when hitting API.
    """
    main_path = "backend/main.py"
    if main_path not in files:
        return
    content = files[main_path]
    if "create_all" in content or "metadata.create_all" in content:
        return
    # Import all model modules so they register with Base.metadata
    model_imports = ["    try:\n        import models\n    except ImportError:\n        pass"]
    for path in files:
        if path.startswith("backend/models/") and path.endswith(".py") and "__init__" not in path:
            mod = path.replace("backend/models/", "").replace(".py", "")
            if mod:
                model_imports.append(f"    try:\n        from models import {mod}\n    except ImportError:\n        pass")

    import_block = "\n".join(model_imports)

    block = f'''
@app.on_event("startup")
def _ensure_tables():
    """Create database tables if they do not exist."""
    import database
{import_block}
    try:
        database.Base.metadata.create_all(bind=database.engine)
    except Exception as e:
        print(f"Warning: Could not create tables: {{e}}")
'''
    if "app = FastAPI()" in content:
        content = content.replace("app = FastAPI()", "app = FastAPI()" + block, 1)
        files[main_path] = content
    elif "app=FastAPI()" in content:
        content = content.replace("app=FastAPI()", "app=FastAPI()" + block, 1)
        files[main_path] = content


def _fix_backend_python_relative_imports(files: Dict[str, str]) -> None:
    """
    Backend is run as 'uvicorn main:app' from the backend directory, so all .py
    files are loaded as top-level modules, not as part of a package. Relative
    imports (from . import ...) then fail. Convert them to absolute imports
    in every backend/*.py file so the app starts.
    """
    for path in list(files.keys()):
        if not (path.startswith("backend/") and path.endswith(".py")):
            continue
        content = files[path]
        if "from . import " not in content and "from ." not in content:
            continue
        content = re.sub(r"\bfrom\s+\.\s+import\s+", "import ", content)
        content = re.sub(r"\bfrom\s+\.(\w+)\s+import\s+", r"from \1 import ", content)
        files[path] = content


def _fix_frontend_backend_url_undefined(files: Dict[str, str]) -> None:
    """
    Prevent 404 on /undefined/tasks/ - ensure API base URL is never undefined.
    LLM sometimes generates: const apiBase = process.env.REACT_APP_BACKEND_URL;
    When env is not set, apiBase is undefined, and fetch(`${apiBase}/tasks/`) produces
    request to /undefined/tasks/ (relative URL). Fix by ensuring proper fallback.
    """
    safe_backend_url_expr = (
        "(process.env.REACT_APP_BACKEND_URL || process.env.VITE_BACKEND_URL || "
        "(typeof window !== 'undefined' ? (window.__BACKEND_URL__ || window.location.protocol + '//' + window.location.hostname + ':5004') : 'http://localhost:5004')).trim()"
    )
    for path in list(files.keys()):
        if not path.startswith("frontend/") or path.split(".")[-1] not in ("js", "jsx", "ts", "tsx"):
            continue
        content = files[path]
        changed = False

        # Fix: const x = process.env.REACT_APP_BACKEND_URL (no fallback)
        for varname in ("backendUrl", "apiUrl", "apiBase", "BASE_URL", "backend_url", "API_URL"):
            # Match: const backendUrl = process.env.REACT_APP_BACKEND_URL or similar
            pattern = rf"(\bconst\s+{varname}\s*=\s*)(process\.env\.(?:REACT_APP_BACKEND_URL|VITE_BACKEND_URL)(?:\s*\|\|\s*process\.env\.(?:REACT_APP_BACKEND_URL|VITE_BACKEND_URL))?\s*(?:\|\|\s*['\"][^'\"]*['\"])?\s*)\.?(trim\(\))?"
            if re.search(pattern, content):
                # Already has some fallback - ensure it has default for dev
                pass
            # Simpler: replace any assignment that ends with just env var (no || fallback)
            old_pat = rf"(\bconst\s+{varname}\s*=\s*)process\.env\.REACT_APP_BACKEND_URL\s*;"
            if re.search(old_pat, content):
                content = re.sub(
                    old_pat,
                    rf"\1{safe_backend_url_expr};",
                    content,
                )
                changed = True
            old_pat2 = rf"(\bconst\s+{varname}\s*=\s*)process\.env\.VITE_BACKEND_URL\s*;"
            if re.search(old_pat2, content):
                content = re.sub(
                    old_pat2,
                    rf"\1{safe_backend_url_expr};",
                    content,
                )
                changed = True

        # Upgrade static localhost to dynamic URL (works when deployed on different host)
        static_pat = r"\(\s*process\.env\.(?:REACT_APP_BACKEND_URL|VITE_BACKEND_URL)\s*\|\|\s*process\.env\.(?:REACT_APP_BACKEND_URL|VITE_BACKEND_URL)\s*\|\|\s*['\"]http://localhost:5004['\"]\s*\)\s*\.trim\(\)"
        if re.search(static_pat, content):
            content = re.sub(static_pat, safe_backend_url_expr, content)
            changed = True
        static_pat2 = r"\(\s*process\.env\.(?:REACT_APP_BACKEND_URL|VITE_BACKEND_URL)\s*\|\|\s*['\"]http://localhost:5004['\"]\s*\)\s*\.trim\(\)"
        if re.search(static_pat2, content):
            content = re.sub(static_pat2, safe_backend_url_expr, content)
            changed = True
        # Fix: (process.env.X || '').trim() - replace empty fallback with dynamic
        new_content = re.sub(
            r"(\|\|\s*['\"]['\"]\s*)\s*\)\s*\.trim\(\)",
            "|| (typeof window !== 'undefined' ? (window.__BACKEND_URL__ || window.location.protocol + '//' + window.location.hostname + ':5004') : 'http://localhost:5004')).trim()",
            content,
        )
        if new_content != content:
            content = new_content
            changed = True

        # Fix generic: any var = process.env.REACT_APP_BACKEND_URL (single env, no fallback)
        new_content = re.sub(
            r"(\bconst\s+\w+\s*=\s*)process\.env\.REACT_APP_BACKEND_URL\s*(?!\s*\|\|)(?!\s*\.trim)\s*;",
            rf"\1{safe_backend_url_expr};",
            content,
        )
        if new_content != content:
            content = new_content
            changed = True
        new_content = re.sub(
            r"(\bconst\s+\w+\s*=\s*)process\.env\.VITE_BACKEND_URL\s*(?!\s*\|\|)(?!\s*\.trim)\s*;",
            rf"\1{safe_backend_url_expr};",
            content,
        )
        if new_content != content:
            content = new_content
            changed = True

        # Ensure fetch/axios never get undefined - replace ${var} with ${var || 'http://localhost:5004'}
        for var in ("backendUrl", "apiUrl", "apiBase", "baseUrl", "API_URL"):
            pat = re.escape(f"${{{var}}}") + r"(?!\s*\|\|)"  # Match ${var} not already followed by ||
            if re.search(pat, content):
                content = re.sub(
                    pat,
                    f"${{{var} || 'http://localhost:5004'}}",
                    content,
                )
                changed = True

        if changed:
            files[path] = content


def _fix_frontend_map_safety(files: Dict[str, str]) -> None:
    """
    Prevent "X.map is not a function" errors. Ensure every .map() on array-like variables
    has a fallback. Replace {var.map( with {(var || []).map( in JSX.
    """
    for path in list(files.keys()):
        if not path.startswith("frontend/") or path.split(".")[-1] not in ("js", "jsx", "ts", "tsx"):
            continue
        content = files[path]
        changed = False
        # Pattern: { ident.map( or {ident.map( - variable directly before .map
        # Avoid double-wrapping: (x || []).map should stay
        def _repl(match):
            pre, ident, dotmap = match.group(1), match.group(2), match.group(3)
            if ident in ("[]", ")") or "|| []" in match.group(0) or "?? []" in match.group(0):
                return match.group(0)
            return f"{pre}({ident} || []){dotmap}"
        new_content = re.sub(
            r"(\{)\s*(\w+)(\.map\s*\()",
            _repl,
            content,
        )
        if new_content != content:
            changed = True
            content = new_content
        if changed:
            files[path] = content


def _fix_backend_routes_import(files: Dict[str, str]) -> None:
    """
    Fix ImportError: cannot import name 'todo_router' from 'routes'.
    - If routes.py doesn't exist: remove the bad import and include_router calls so app starts.
    - If routes.py exists but exports don't match (e.g. has 'router' but main wants 'todo_router'):
      fix main.py to import what routes actually exports (e.g. 'router') and use that.
    """
    main_path = "backend/main.py"
    routes_path = "backend/routes.py"
    if main_path not in files:
        return
    content = files[main_path]
    if "from routes import" not in content:
        return
    import_match = re.search(r"from\s+routes\s+import\s+([^\n]+)", content)
    if not import_match:
        return
    imported = [x.strip().split(" as ")[0] for x in import_match.group(1).split(",")]
    routes_content = files.get(routes_path, "")
    if not routes_content:
        # No routes.py - remove the bad import and include_router so app at least starts
        content = re.sub(r"\s*from\s+routes\s+import\s+[^\n]+\n", "\n", content)
        content = re.sub(r"\s*app\.include_router\s*\(\s*[^)]+\s*\)\s*\n", "\n", content)
        files[main_path] = content
        return
    # routes.py exists - check what it exports
    router_defs = re.findall(r"^(\w+)\s*=\s*APIRouter\s*\([^)]*\)", routes_content, re.MULTILINE)
    missing = [x for x in imported if x not in router_defs]
    if not missing and router_defs:
        return
    # main expects todo_router/user_router but routes has 'router' - fix main to use router
    if router_defs and set(imported) != set(router_defs):
        content = re.sub(
            r"from\s+routes\s+import\s+[^\n]+",
            f"from routes import {', '.join(router_defs)}",
            content,
        )
        # Replace all app.include_router(x) with app.include_router(router) for the first def
        for old_name in imported:
            content = re.sub(
                rf"app\.include_router\s*\(\s*{re.escape(old_name)}\s*\)",
                f"app.include_router({router_defs[0]})",
                content,
            )
        # Remove duplicate include_router calls when we collapsed multiple to one
        seen_routers = set()
        def dedupe_router(match):
            router_name = match.group(2)
            if router_name in seen_routers:
                return ""
            seen_routers.add(router_name)
            return match.group(0)
        content = re.sub(
            r"(\s*app\.include_router\s*\(\s*(\w+)\s*\)\s*\n)",
            dedupe_router,
            content,
        )
        files[main_path] = content


def _ensure_vite_react_plugin_in_package_json(files: Dict[str, str]) -> None:
    """
    If the project has a Vite frontend (vite.config.js/ts), ensure frontend/package.json
    includes @vitejs/plugin-react in devDependencies. The LLM sometimes omits it even when
    vite.config uses it, which causes 'Cannot find module @vitejs/plugin-react' at runtime.
    Mutates files in place so generated output is always runnable without manual installs.
    """
    has_vite_config = (
        "frontend/vite.config.js" in files or "frontend/vite.config.ts" in files
    )
    pkg_path = "frontend/package.json"
    if not has_vite_config or pkg_path not in files:
        return
    try:
        pkg = json.loads(files[pkg_path])
        dev_deps = pkg.get("devDependencies") or {}
        if dev_deps.get(VITE_REACT_PLUGIN):
            return
        dev_deps[VITE_REACT_PLUGIN] = "^4.2.1"
        pkg["devDependencies"] = dev_deps
        files[pkg_path] = json.dumps(pkg, indent=2)
    except (json.JSONDecodeError, TypeError):
        pass


def _validate_and_fix_backend_imports(files: Dict[str, str]) -> None:
    """
    Fix backend import errors:
    1. If main.py imports 'services' but services/ is a directory, create services/__init__.py
    2. Remove imports of non-existent modules
    """
    main_path = "backend/main.py"
    if main_path not in files:
        return
    
    content = files[main_path]
    
    # Check if services is imported
    if "import services" in content or "from services import" in content:
        # Check if services is a directory (has services/*.py files)
        has_services_dir = any(p.startswith("backend/services/") and p.endswith(".py") and "__init__" not in p for p in files)
        has_services_file = "backend/services.py" in files
        has_services_init = "backend/services/__init__.py" in files
        
        if has_services_dir and not has_services_file and not has_services_init:
            # Create __init__.py to make it a proper module
            service_files = [p for p in files if p.startswith("backend/services/") and p.endswith(".py") and "__init__" not in p]
            exports = []
            for sf in service_files:
                # Extract class names from service files
                service_content = files[sf]
                class_names = re.findall(r"^class (\w+Service):", service_content, re.MULTILINE)
                module_name = sf.replace("backend/services/", "").replace(".py", "")
                for cls in class_names:
                    exports.append(f"from .{module_name} import {cls}")
            
            if exports:
                files["backend/services/__init__.py"] = "\n".join(exports) + "\n"


def _ensure_complete_styles_css(files: Dict[str, str]) -> None:
    """
    Ensure styles.css has all required classes. If generated CSS is too minimal,
    replace with the comprehensive default template.
    """
    css_path = "frontend/src/styles.css"
    if css_path not in files:
        files[css_path] = _DEFAULT_STYLES_CSS
        return
    
    css_content = files[css_path]
    required_classes = [".app", ".btn", ".input", ".form-row", ".list-item"]
    missing = [cls for cls in required_classes if cls not in css_content]
    
    # If missing critical classes or CSS is too short (less than 80 lines), use default
    line_count = len(css_content.strip().split('\n'))
    if missing or line_count < 80:
        files[css_path] = _DEFAULT_STYLES_CSS

    # Special case: If it looks like a todo app (has todo classes in JSX), ensure premium todo CSS is present
    has_todo_markup = any("todo-" in content for path, content in files.items() if path.endswith((".js", ".jsx", ".tsx")))
    if has_todo_markup and ".todo-card" not in css_content:
        files[css_path] += _TODO_APP_PREMIUM_CSS


async def generate_code_from_plan(
    requirement: str,
    prd: str,
    plan: list,
    architecture: Dict[str, Any],
    llm,
    uiux: str = ""
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generates actual code based on PRD and implementation plan using LLM.
    When a matching template exists in app_builder/templates, uses template code modules instead.
    
    Args:
        requirement: Original user requirement
        prd: Complete PRD document
        plan: Implementation plan with steps
        architecture: Architecture decisions
        llm: Initialized LLM instance
        
    Yields:
        Dictionary events with file generation progress
    """
    # Check for template - use template code modules when available
    try:
        from app_builder.services.template_service import detect_template, get_template_code_files
        template_name = detect_template(requirement)
        if template_name:
            template_files = get_template_code_files(template_name)
            if template_files:
                yield {"event": "generation_start", "message": f"Using template: {template_name}"}
                from app_builder.agents.code_update_agent import tailor_template_to_requirement
                tailored = await tailor_template_to_requirement(template_files, requirement, template_name)
                for path, content in tailored.items():
                    yield {"event": "file_generated", "file": path, "content": content}
                _normalize_frontend_package_json_to_cra(tailored)
                _ensure_complete_styles_css(tailored)
                yield {"event": "generation_complete", "data": tailored}
                return
    except ImportError:
        pass

    # Extract key information from plan
    plan_summary = "\n".join([
        f"{i+1}. {step.get('title', step) if isinstance(step, dict) else step}"
        for i, step in enumerate(plan[:20])  # Increased context
    ])
    
    # Extract database schema
    db_schema = architecture.get('database_schema', {})
    tables = db_schema.get('tables', [])
    
    # Parse architecture details
    project_structure = architecture.get("project_structure", {})
    raw_backend_files = project_structure.get("backend", [])
    raw_frontend_files = project_structure.get("frontend", [])
    
    # Ensure all files have the correct directory prefix
    backend_files = []
    for f in raw_backend_files:
        if f.startswith("backend/"):
            backend_files.append(f)
        else:
            backend_files.append(f"backend/{f}")
            
    frontend_files = []
    for f in raw_frontend_files:
        if f.startswith("frontend/"):
            frontend_files.append(f)
        else:
            frontend_files.append(f"frontend/{f}")
    
    backend_framework = architecture.get('backend_structure', {}).get('framework', 'Unknown')
    frontend_framework = architecture.get('frontend_structure', {}).get('framework', 'Unknown')
    
    # Format architecture summary for context
    arch_summary = json.dumps(architecture, indent=2)
    
    # Construct the system prompt
    system_prompt = f"""You are an expert full-stack developer. You must generate a complete, working application based on the requirements, architecture, and UI/UX design.

**UI/UX DESIGN IS THE PRIMARY DRIVER – CODE MUST FOLLOW THE UI/UX SPECIFICATION:**
- All layouts, colors, typography, and component styling MUST be derived from the UI/UX Design Specification below.
- If a UI/UX spec is provided: extract Primary, Secondary, Background, Text colors and apply them in `src/styles.css` using CSS variables.
- Build professional interfaces: generous whitespace, subtle shadows, clean typography (Inter font), hover states.

**PRODUCTION READY CODE – GENERATE HIGH-QUALITY APPLICATIONS:**
1. **Visual Excellence**:
   - Use Tailwind CSS for ALL styling.
   - Font: Inter (add Google Fonts link in index.html).
   - Whitespace: padding 1-2rem, gap 0.75rem. Avoid cramped UIs.
2. **Interactive**: Add :hover states in styles.css. Use transition for smooth interactions.
3. **No Placeholders**: Write the FULL code. No `# implementation here`.
4. **Design Fidelity**: Follow UI/UX colors and fonts in styles.css.

**STRICT FILE GENERATION RULES:**
5. **Dependencies**: Minimal. React: `react`, `react-dom`, `react-scripts`. Add `tailwindcss`, `postcss`, `autoprefixer` for styling.
6. **package.json scripts**: MUST use `NODE_OPTIONS=--openssl-legacy-provider` for Node 20+ compatibility:
   "start": "NODE_OPTIONS=--openssl-legacy-provider react-scripts start"
7. **Tailwind CSS**: MANDATORY. Always generate `tailwind.config.js` and `postcss.config.js`. Use Tailwind utility classes in JSX.
8. **frontend/src/index.js**: MUST import `./styles.css` before App.
9. **Backend**: Use absolute imports. SQLite + SQLAlchemy. MUST add CORS: `app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])` so frontend can call API from any origin.

**UI/UX Design Specification:**
{uiux if uiux else "Standard modern UI/UX design (CLEAN, MODERN, USER-FRIENDLY)."}

{f'''**PREMIUM TODO APP INSTRUCTIONS**:
- This is a TODO/TASK app. USE the premium CSS classes: .todo-card, .todo-item, .todo-checkbox, .todo-text, .todo-badge.
- Layout: Header with title, Input area at top inside a .todo-card, Filter tabs (All, Active, Completed), List of .todo-item below.
- Interactions: Checkbox click toggles .checked class, .todo-text gets .completed class. Show priority with .todo-badge-high/medium/low.
''' if _detect_todo_app(requirement) else ""}

**CRITICAL STYLE INSTRUCTIONS:**
1. **Tailwind CSS MANDATORY**:
   - You MUST generate `tailwind.config.js` and `postcss.config.js`.
   - In `frontend/src/index.css`, include:
     @tailwind base;
     @tailwind components;
     @tailwind utilities;
   - Add Inter font: <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" /> in index.html.
   - **REQUIRED CSS classes** (all must be present): .app, .app-container, .app-title, .app-subtitle, .btn, .btn-primary, .input, .form-row, .list, .list-item
   - Each class MUST have: proper padding/margins, borders, border-radius, hover states with transitions, focus states for inputs
   - Minimum 100 lines of CSS - use comprehensive styling, never generate minimal CSS
   - Use CSS classes in JSX with className.

2. **UI/UX COMPLIANCE**:
   - Extract Primary, Secondary, Background colors from UI/UX spec and put in :root in styles.css.
   - Layout: If UI/UX says "Sidebar + Top Nav" -> build layout components. Use flexbox/grid in styles.css.

3. **ROBUSTNESS**:
   - Ensure the app is "npm install" ready. All deps used in code must be in `package.json`.

4. **FRONTEND MUST WORK END-TO-END EVEN IF BACKEND IS DOWN — USE LOCALSTORAGE**:
   - Frontend MUST be a FULLY FUNCTIONAL app even when backend never starts. Use localStorage as the data layer.
   - **On component mount**: Load data from localStorage FIRST (`JSON.parse(localStorage.getItem('appName_items')) || []`), THEN try API fetch.
   - **On every state change (add/edit/delete)**: Save to localStorage immediately: `localStorage.setItem('appName_items', JSON.stringify(updatedItems))`.
   - **API calls are OPTIONAL enhancements**: Wrap ALL fetch() calls in try/catch. On success, sync localStorage. On failure, silently continue with localStorage data.
   - **CRUD operations must work 100% with localStorage alone**:
     * CREATE: Push new item (with `id: Date.now()`) to state + localStorage. Try API in background.
     * READ: Load from localStorage. Try API, merge if available.
     * UPDATE: Update state + localStorage. Try API in background.
     * DELETE: Remove from state + localStorage. Try API in background.
   - **Pattern to use in EVERY component that manages data**:
     Initialize state from localStorage: const [items, setItems] = useState(() => JSON.parse(localStorage.getItem(STORAGE_KEY)) || []);
     Sync to localStorage on change: useEffect(() => localStorage.setItem(STORAGE_KEY, JSON.stringify(items)), [items]);
     Optional API fetch on mount: useEffect(() => fetch(url).then(r => r.json()).then(d => Array.isArray(d) && setItems(d)).catch(() => {{}}), []);
   - API base URL: Use dynamic URL for local + deployment: `const backendUrl = (process.env.REACT_APP_BACKEND_URL || process.env.VITE_BACKEND_URL || (typeof window !== 'undefined' ? (window.__BACKEND_URL__ || window.location.protocol + '//' + window.location.hostname + ':5004') : 'http://localhost:5004')).trim();`
   - NEVER leave backendUrl undefined. Guard API calls: if (!backendUrl) return;

5. **FRONTEND ARRAY SAFETY - PREVENT "X.map is not a function"**:
   - ALWAYS use safe patterns for .map(): use (items || []).map(...) or items?.map(...) - NEVER bare variable.map().
   - Initialize list state with useState([]) - never useState() or useState(undefined).
   - When setting from API: setItems(Array.isArray(data) ? data : []). Never setItems(data) directly.
   - In JSX: use {{(todos || []).map(...)}} or {{items?.map(...) ?? null}} - add fallback for every .map() on arrays/objects.


**CRITICAL - APP MUST RUN END-TO-END WITH ZERO MANUAL FIXES:**
- Use SQLite + SQLAlchemy. No external database. DATABASE_URL defaults to sqlite:///./app.db
- **Backend**: SQLAlchemy engine, SessionLocal, Base. models.Base.metadata.create_all() on startup.
- **Pydantic v2**: model_config = ConfigDict(from_attributes=True), .model_dump()
- **Backend structure**:
  - `backend/models.py`: SQLAlchemy model classes (Column, Integer, String, etc).
  - `backend/schemas.py`: Pydantic BaseModel classes.
  - `backend/database.py`: create_engine, SessionLocal, Base, get_db() generator.
  - `backend/main.py`: FastAPI app, Depends(get_db), create_all on startup.
- requirements.txt: fastapi, uvicorn, sqlalchemy, pydantic. No version numbers.

**Target Architecture:**
- Backend Framework: {backend_framework}
- Frontend Framework: {frontend_framework}
- Database: SQLite (SQLAlchemy) – runs out of the box.

**Expected Backend Files:**
{json.dumps(backend_files, indent=2)}

**Expected Frontend Files:**
{json.dumps(frontend_files, indent=2)}

**Output Format:**
- Use the `FILE: <path>` format for every file.
- Followed by the code block.

**Example:**
FILE: backend/main.py
```python
from fastapi import FastAPI
...
```

**Architecture Context:**
{arch_summary}

**Dependencies:**
- **backend/requirements.txt**: fastapi, uvicorn, sqlalchemy, pydantic. No version numbers.
- **backend/database.py**: SQLAlchemy create_engine, SessionLocal, Base. SQLite by default.
- **backend/models.py**: SQLAlchemy declarative models with Column, Integer, String, etc.
- **Frontend**: Minimal deps only: react, react-dom, react-scripts. Node 20+. Add "engines": {{"node": ">=20"}} to package.json. Use Tailwind CSS. Scripts MUST include NODE_OPTIONS=--openssl-legacy-provider.
- **Frontend State**: Use ONLY React useState and useEffect for state management. Do NOT use Zustand, Redux, MobX, or any external state library. Keep all state in App.js or pass via props. Use localStorage for persistence.
- In README.md: `pip install -r requirements.txt` and `npm install`.
"""

    user_prompt = f"""Generate complete application code for this SPECIFIC requirement:

## REQUIREMENT:
{requirement}

## PRD (Key Points):
{prd[:3000]}

## Implementation Steps:
{plan_summary}

## Architecture & Database Schema:
{arch_summary}

DATABASE TABLES TO IMPLEMENT:
{json.dumps(tables, indent=2) if tables else 'Use entities from requirement'}

IMPORTANT:
- Generate code for the ACTUAL requirement.
- STRICTLY follow the file paths in "project_structure" – generate only those files; do not add extra components or files.
- If the architecture says "NestJS", generate "NestJS" code. If it says "FastAPI", generate "FastAPI" code.
- Ensure all imports match the file structure.
- Prefer a single App file with all UI when the requirement is simple; create separate components only when required.

Start generating strictly using the "FILE: <path>" format.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    yield {
        "event": "generation_start",
        "message": "Starting code generation based on PRD and plan..."
    }
    
    # Stream the code generation
    full_response = ""
    file_starts = []  # List of (start_index, file_path)
    files_yielded = 0
    files_generated = {}  # Initialize to fix NameError

    
    chunk_count = 0
    
    async for chunk in llm.astream(messages):
        if hasattr(chunk, 'content'):
            content = chunk.content
            if not content:
                continue
                
            # append content
            prev_len = len(full_response)
            full_response += content
            chunk_count += 1
            
            # Progress update removed to prevent continuous messaging

            # Search for "FILE:" markers in the newly added content 
            # (including a bit of overlap context to catch split markers)
            search_start = max(0, prev_len - 10)
            
            # Simple check: Does the new content contain "FILE:"? 
            # We scan the full_response from search_start to find new markers
            import re
            # Regex to find "FILE: <path>" at start of line or string
            # We use a simple find loop for robustness
            
            lines = full_response[search_start:].split('\n')
            
            # This is slightly inefficient but safe: check if we found a new FILE: line
            # We track known file starts to avoid duplicates
            
            # Better approach: Scan full_response for all markers, compare with known ones
            # For performance, we can just scan the tail, but "FILE:" is rare enough.
            
            # Let's iterate lines in full text for simplicity and correctness over micro-optimization
            all_lines = full_response.split('\n')
            current_file_starts = []
            
            for i, line in enumerate(all_lines):
                if line.strip().startswith("FILE:"):
                    path = line.replace("FILE:", "").strip()
                    current_file_starts.append((i, path))
            
            # Check if we have a NEW completed file
            # A file is completed if we have a marker AFTER it
            if len(current_file_starts) > files_yielded + 1:
                # We have at least one completed file that hasn't been yielded
                # The file at files_yielded is now complete because files_yielded+1 exists
                
                # Extract content for the file at 'files_yielded'
                start_line_idx, file_path_to_yield = current_file_starts[files_yielded]
                end_line_idx, _ = current_file_starts[files_yielded + 1]
                
                # Content is lines between start and end
                file_lines = all_lines[start_line_idx+1 : end_line_idx]
                file_content = "\n".join(file_lines).strip()
                
                # Clean up markdown code blocks
                if "```" in file_content:
                    parts = file_content.split("```")
                    if len(parts) >= 2:
                        code = parts[1]
                        if code.strip() and "\n" in code: # typical ```python\n code...
                             code = "\n".join(code.split("\n")[1:])
                        file_content = code.strip()
                    else:
                        file_content = file_content.replace("```", "").strip()
                
                files_yielded += 1

                # Skip yielding directories
                if file_path_to_yield.endswith("/") or file_path_to_yield.endswith("\\"):
                    continue

                files_generated[file_path_to_yield] = file_content
                yield {
                    "event": "file_generated",
                    "file": file_path_to_yield,
                    "content": file_content,
                    "message": f"Generated {file_path_to_yield}"
                }
                 

    # Handled yielded files above loop. 
    # Now verify remaining file (the last one)
    all_lines = full_response.split('\n')
    current_file_starts = []
    for i, line in enumerate(all_lines):
        if line.strip().startswith("FILE:"):
            path = line.replace("FILE:", "").strip()
            current_file_starts.append((i, path))
            
    if len(current_file_starts) > files_yielded:
        # Yield the last file
        start_line_idx, file_path_to_yield = current_file_starts[files_yielded]
        # Content is distinct from start to end of string
        file_lines = all_lines[start_line_idx+1:]
        file_content = "\n".join(file_lines).strip()
        
        # Clean up markdown
        if "```" in file_content:
            parts = file_content.split("```")
            if len(parts) >= 2:
                code = parts[1]
                if code.strip() and "\n" in code:
                     code = "\n".join(code.split("\n")[1:])
                file_content = code.strip()
            # Handle case where closing ``` is missing (end of stream)
            elif len(parts) == 1 and parts[0].strip():
                 # Maybe header ``` is there but no footer
                 pass 
        
        file_content = file_content.replace("```", "").strip() # fallback cleanup
        
        files_generated[file_path_to_yield] = file_content
        yield {
            "event": "file_generated",
            "file": file_path_to_yield,
            "content": file_content,
            "message": f"Generated {file_path_to_yield}"
        }

    # Prefer CRA (react-scripts) over Vite: if generated frontend has Vite or @dnd-kit, replace with minimal runnable CRA
    _normalize_frontend_package_json_to_cra(files_generated)
    # Ensure Node 20+ compat
    _ensure_node_compat(files_generated)
    # Fix "X.map is not a function" - add (var || []) fallback for all .map() in JSX
    _fix_frontend_map_safety(files_generated)
    # Fix "cannot import todo_router from routes" - inline routes into main.py
    _fix_backend_routes_import(files_generated)
    # Ensure Vite + React projects have @vitejs/plugin-react in package.json (only if we didn't replace with CRA)
    _ensure_vite_react_plugin_in_package_json(files_generated)
    # Normalize backend requirements: no version numbers, core packages present
    _normalize_backend_requirements(files_generated)
    # Ensure database.py uses SQLite by default (no external credentials - runs out of the box)
    _ensure_sqlite_database_default(files_generated)
    # Fix all backend .py relative imports so uvicorn main:app works (no parent package).
    _fix_backend_python_relative_imports(files_generated)
    # Fix SQLAlchemy UUID import (sqlite dialect has Uuid, not UUID - use types.Uuid)
    _fix_sqlalchemy_uuid_imports(files_generated)
    # Fix Pydantic v2, JSONB->JSON for SQLite compatibility
    _fix_backend_pydantic_and_common(files_generated)
    # Ensure database tables are created on startup (prevents "no such table" error)
    _ensure_database_tables_created(files_generated)
    # Ensure CORS allows all origins for generated backend
    _ensure_cors_in_backend(files_generated)
    # Prevent 404 on /undefined/tasks/ - ensure API URL is never undefined
    _fix_frontend_backend_url_undefined(files_generated)
    # Fix backend import errors (services/ directory without __init__.py)
    _validate_and_fix_backend_imports(files_generated)
    # Ensure styles.css is complete with all required classes
    _ensure_complete_styles_css(files_generated)

    yield {
        "event": "generation_complete",
        "data": files_generated,
        "message": f"Generated {len(files_generated)} files"
    }
