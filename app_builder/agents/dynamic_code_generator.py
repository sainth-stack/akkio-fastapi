"""
Dynamic Code Generator Agent - Generates code based on PRD and implementation plan using LLM
"""
from typing import Dict, Any, AsyncGenerator
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage

VITE_REACT_PLUGIN = "@vitejs/plugin-react"

# Minimal CRA package.json - Node 20+, no Tailwind
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
        # Ensure styles.css exists (mandatory for styling)
        if "frontend/src/styles.css" not in files:
            files["frontend/src/styles.css"] = _DEFAULT_STYLES_CSS
        # Remove Tailwind config files
        for f in ("frontend/tailwind.config.js", "frontend/postcss.config.js"):
            if f in files:
                del files[f]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass


_DEFAULT_STYLES_CSS = """/* App styles - fonts, colors, layout */
:root {
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --color-bg: #f8fafc;
  --color-surface: #ffffff;
  --color-text: #1e293b;
  --color-text-muted: #64748b;
  --color-primary: #4f46e5;
  --color-primary-hover: #4338ca;
  --color-border: #e2e8f0;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
  --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1);
}

* { box-sizing: border-box; }
body { margin: 0; font-family: var(--font-sans); background: var(--color-bg); color: var(--color-text); -webkit-font-smoothing: antialiased; }

.app { min-height: 100vh; padding: 2rem; }
.app-container { max-width: 42rem; margin: 0 auto; }
.app-title { font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem; }
.app-subtitle { color: var(--color-text-muted); font-size: 0.875rem; margin-bottom: 1.5rem; }
.form-row { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 2rem; }
.input { flex: 1; min-width: 120px; padding: 0.5rem 1rem; border: 1px solid var(--color-border); border-radius: 0.5rem; font-family: inherit; }
.input:focus { outline: none; border-color: var(--color-primary); box-shadow: 0 0 0 2px rgba(79,70,229,0.2); }
.btn { padding: 0.5rem 1rem; border-radius: 0.5rem; font-weight: 500; font-family: inherit; cursor: pointer; border: none; transition: background 0.2s; }
.btn-primary { background: var(--color-primary); color: white; }
.btn-primary:hover:not(:disabled) { background: var(--color-primary-hover); }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
.list { list-style: none; padding: 0; margin: 0; }
.list-item { padding: 1rem; background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 0.75rem; margin-bottom: 0.75rem; box-shadow: var(--shadow-sm); }
.list-item:hover { box-shadow: var(--shadow-md); }
.list-item strong { font-weight: 500; }
.list-item .muted { color: var(--color-text-muted); margin-left: 0.5rem; }
"""


def _ensure_frontend_styles_and_node_compat(files: Dict[str, str]) -> None:
    """
    Ensure frontend uses styles.css (not Tailwind), works with Node 20+.
    - Add NODE_OPTIONS=--openssl-legacy-provider to react-scripts start/build
    - Ensure styles.css exists and is imported
    - Remove Tailwind packages
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
            # Remove Tailwind and other heavy deps
            deps = pkg.get("dependencies") or {}
            for k in list(deps.keys()):
                if k in ("tailwindcss", "postcss", "autoprefixer", "framer-motion", "lucide-react", "clsx", "tailwind-merge"):
                    del deps[k]
            pkg["dependencies"] = deps
            dev_deps = pkg.get("devDependencies") or {}
            for k in list(dev_deps.keys()):
                if k in ("tailwindcss", "postcss", "autoprefixer"):
                    del dev_deps[k]
            pkg["devDependencies"] = dev_deps
            files[pkg_path] = json.dumps(pkg, indent=2)
        except (json.JSONDecodeError, TypeError):
            pass
    # Ensure styles.css exists
    if "frontend/src/styles.css" not in files:
        files["frontend/src/styles.css"] = _DEFAULT_STYLES_CSS
    # Ensure index.js imports styles.css
    for idx_path in ("frontend/src/index.js", "frontend/src/index.jsx"):
        if idx_path in files:
            content = files[idx_path]
            if "styles.css" not in content:
                content = content.replace("import './index.css'", "import './styles.css'")
                content = content.replace('import "./index.css"', 'import "./styles.css"')
                content = content.replace("import './App.css'", "import './styles.css'")
                if "styles.css" not in content:
                    content = content.replace("import App from", "import './styles.css';\nimport App from", 1)
                files[idx_path] = content
    # Remove Tailwind config files
    for f in ("frontend/tailwind.config.js", "frontend/postcss.config.js"):
        if f in files:
            del files[f]


# Core backend packages for FastAPI + SQLite (no version numbers, no external DB)
_BACKEND_CORE_PACKAGES = ["fastapi", "uvicorn", "sqlalchemy", "pydantic"]

# PyPI packages to NEVER add - usually local modules in generated apps (services, etc.)
_BACKEND_BLOCKLIST = {"services"}


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
        # Strip version: package, package==x, package>=x, package[extra] -> package
        pkg = line.split("==")[0].split(">=")[0].split("[")[0].strip().lower()
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
        "(process.env.REACT_APP_BACKEND_URL || process.env.VITE_BACKEND_URL || 'http://localhost:5001').trim()"
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

        # Fix: (process.env.REACT_APP_BACKEND_URL || process.env.VITE_BACKEND_URL || '').trim()
        # Replace || '' with || 'http://localhost:5001' for dev default
        if "process.env.REACT_APP_BACKEND_URL" in content or "process.env.VITE_BACKEND_URL" in content:
            # Ensure we have a default so it's never undefined
            content = re.sub(
                r"(\|\|\s*['\"]['\"]\s*)\s*\)\s*\.trim\(\)",
                "|| 'http://localhost:5001').trim()",
                content,
            )
            if "|| 'http://localhost:5001'" in content or "|| \"http://localhost:5001\"" in content:
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

        # Ensure fetch/axios never get undefined - replace ${var} with ${var || 'http://localhost:5001'}
        for var in ("backendUrl", "apiUrl", "apiBase", "baseUrl", "API_URL"):
            pat = re.escape(f"${{{var}}}") + r"(?!\s*\|\|)"  # Match ${var} not already followed by ||
            if re.search(pat, content):
                content = re.sub(
                    pat,
                    f"${{{var} || 'http://localhost:5001'}}",
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
    
    Args:
        requirement: Original user requirement
        prd: Complete PRD document
        plan: Implementation plan with steps
        architecture: Architecture decisions
        llm: Initialized LLM instance
        
    Yields:
        Dictionary events with file generation progress
    """
    
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
   - Use `src/styles.css` for ALL styling. NO Tailwind. Define :root variables (--color-primary, --font-sans, etc).
   - Font: Inter (add Google Fonts link in index.html). Use CSS variables for colors.
   - Whitespace: padding 1-2rem, gap 0.75rem. Avoid cramped UIs.
2. **Interactive**: Add :hover states in styles.css. Use transition for smooth interactions.
3. **No Placeholders**: Write the FULL code. No `# implementation here`.
4. **Design Fidelity**: Follow UI/UX colors and fonts in styles.css.

**STRICT FILE GENERATION RULES:**
5. **Dependencies**: MINIMAL. React: `react`, `react-dom`, `react-scripts` only. NO Tailwind, Vite, framer-motion, lucide-react.
6. **package.json scripts**: MUST use `NODE_OPTIONS=--openssl-legacy-provider` for Node 20+ compatibility:
   "start": "NODE_OPTIONS=--openssl-legacy-provider react-scripts start"
7. **frontend/src/styles.css**: MANDATORY. Define :root variables, body, .app, .btn, .input, .list-item, etc.
8. **frontend/src/index.js**: MUST import `./styles.css` before App.
9. **Backend**: Use absolute imports. SQLite only. MUST add CORS: `app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])` so frontend can call API from any origin.

**UI/UX Design Specification:**
{uiux if uiux else "Standard modern UI/UX design (CLEAN, MODERN, USER-FRIENDLY)."}

**CRITICAL STYLE INSTRUCTIONS:**
1. **styles.css MANDATORY** - NO Tailwind:
   - You MUST generate `frontend/src/styles.css` with :root variables (--font-sans, --color-primary, --color-bg, etc).
   - Add Inter font: <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" /> in index.html.
   - Use CSS classes (.app, .btn, .input, .app-title, .list-item, etc) in styles.css. Reference them in JSX with className.

2. **UI/UX COMPLIANCE**:
   - Extract Primary, Secondary, Background colors from UI/UX spec and put in :root in styles.css.
   - Layout: If UI/UX says "Sidebar + Top Nav" -> build layout components. Use flexbox/grid in styles.css.

3. **ROBUSTNESS**:
   - Ensure the app is "npm install" ready. All deps used in code must be in `package.json`.

4. **FRONTEND WORKS WITH OR WITHOUT BACKEND**:
   - Frontend MUST run and display UI even when backend is not configured or not running.
   - Use useState for local state. Initialize with empty array/defaults. NEVER throw or crash when API fails.
   - When fetching: wrap in try/catch, on error set empty data or keep previous state. Show UI with local state only.
   - API base URL: ALWAYS use `const backendUrl = (process.env.REACT_APP_BACKEND_URL || process.env.VITE_BACKEND_URL || 'http://localhost:5001').trim();` - NEVER leave it undefined (causes 404 on /undefined/tasks/). Guard API calls: if (!backendUrl) return; before fetch when optional.

5. **FRONTEND ARRAY SAFETY - PREVENT "X.map is not a function"**:
   - ALWAYS use safe patterns for .map(): use (items || []).map(...) or items?.map(...) - NEVER bare variable.map().
   - Initialize list state with useState([]) - never useState() or useState(undefined).
   - When setting from API: setItems(Array.isArray(data) ? data : []). Never setItems(data) directly.
   - In JSX: use {{(todos || []).map(...)}} or {{items?.map(...) ?? null}} - add fallback for every .map() on arrays/objects.


**CRITICAL - APP MUST RUN END-TO-END WITH ZERO MANUAL FIXES:**
- Use SQLite only (sqlite:///./app.db). No PostgreSQL, no external DB.
- **Backend**: MUST call `database.Base.metadata.create_all(bind=database.engine)` at startup (or @app.on_event("startup")). Import all model modules first. Tables must be created automatically - no manual migration.
- For id columns: use `Integer` (preferred) or `from sqlalchemy.types import Uuid as UUID` – NEVER `from sqlalchemy.dialects.sqlite import UUID` or `from sqlalchemy.dialects.postgresql import UUID`.
- Use `from_attributes = True` in Pydantic models (not orm_mode). Use `.model_dump()` not `.dict()`.
- Use `JSON` from sqlalchemy, NOT `JSONB` (PostgreSQL-specific).
- NEVER add "services" to requirements.txt – it is a local module.
- **Backend routes**: Define ALL FastAPI endpoints directly in backend/main.py. NEVER use `from routes import todo_router, user_router` or create a separate routes.py. Do NOT split into router modules - keep all @app.get, @app.post, etc. in main.py to avoid ImportError.

**Target Architecture:**
- Backend Framework: {backend_framework}
- Frontend Framework: {frontend_framework}
- Database: SQLite (sqlite:///./app.db) – runs without any setup.

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
- **backend/requirements.txt**: Minimal only: fastapi, uvicorn, sqlalchemy, pydantic. No version numbers. Python 3.9+. Do NOT add unnecessary packages.
- **backend/database.py**: MUST use SQLite by default (sqlite:///./app.db). No external database setup - app runs out of the box. Include connect_args={{"check_same_thread": False}} for SQLite.
- **backend/models.py**: For UUID columns use `from sqlalchemy.types import Uuid as UUID` or `String(36)`. NEVER use `from sqlalchemy.dialects.sqlite import UUID` (SQLite has no UUID - use types.Uuid).
- **Frontend**: Minimal deps only: react, react-dom, react-scripts. Node 20+. Add "engines": {{"node": ">=20"}} to package.json. NO Tailwind. Use styles.css. Scripts MUST include NODE_OPTIONS=--openssl-legacy-provider.
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
    # Ensure styles.css, Node 20+ compat, no Tailwind
    _ensure_frontend_styles_and_node_compat(files_generated)
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

    yield {
        "event": "generation_complete",
        "data": files_generated,
        "message": f"Generated {len(files_generated)} files"
    }
