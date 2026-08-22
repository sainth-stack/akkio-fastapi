"""
Single base scaffold loader — React (Vite) + FastAPI.
All generated apps start from base-vite-fastapi; LLM customizes allowlisted files only.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, FrozenSet, Optional

logger = logging.getLogger("app_builder")

BASE_TEMPLATE_NAME = "base-vite-fastapi"
TEMPLATES_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
BASE_DIR = os.path.join(TEMPLATES_BASE, BASE_TEMPLATE_NAME)

LLM_ALLOWLIST_PREFIXES = (
    "frontend/src/App.jsx",
    "frontend/src/App.js",
    "frontend/src/components/",
    "frontend/src/styles/app.css",
    "frontend/src/styles.css",
    "backend/models.py",
    "backend/schemas.py",
    "backend/routes.py",
)

FROZEN_PATHS: FrozenSet[str] = frozenset({
    "frontend/package.json",
    "frontend/vite.config.js",
    "frontend/index.html",
    "frontend/src/main.jsx",
    "frontend/src/api/client.js",
    "frontend/src/styles/base.css",
    "backend/main.py",
    "backend/database.py",
    "backend/requirements.txt",
})

_ENTITY_PRIORITY = ("tasks", "items", "todos", "lists", "entries", "records")
_SKIP_TABLES = frozenset({"users", "user", "sessions", "session", "auth", "reminders", "task_tags", "tags"})

_STUB_MARKERS = (
    "minimal app component",
    "satisfy the build",
    "placeholder app",
    "todo: implement",
)


def is_frozen_path(path: str) -> bool:
    norm = path.replace("\\", "/").lstrip("/")
    return norm in FROZEN_PATHS


def is_allowlisted_path(path: str) -> bool:
    norm = path.replace("\\", "/").lstrip("/")
    if is_frozen_path(norm):
        return False
    return any(norm == p or norm.startswith(p) for p in LLM_ALLOWLIST_PREFIXES)


def _read_file(base_path: str, rel_path: str) -> Optional[str]:
    full = os.path.join(base_path, rel_path)
    if not os.path.isfile(full):
        return None
    try:
        with open(full, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _collect_base_files() -> Dict[str, str]:
    if not os.path.isdir(BASE_DIR):
        logger.error("[scaffold] base template missing: %s", BASE_DIR)
        return {}
    files: Dict[str, str] = {}
    exclude_dirs = {"__pycache__", "node_modules", ".git", "dist", "build"}
    for root, dirs, filenames in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        rel_root = os.path.relpath(root, BASE_DIR)
        for fn in filenames:
            if fn.startswith(".") or fn.endswith(".pyc"):
                continue
            if fn == "base.json":
                continue
            rel_path = os.path.join(rel_root, fn) if rel_root != "." else fn
            rel_path = rel_path.replace("\\", "/")
            if not (rel_path.startswith("backend/") or rel_path.startswith("frontend/")):
                continue
            content = _read_file(BASE_DIR, rel_path)
            if content is not None:
                files[rel_path] = content
    return files


def load_base_config() -> Dict[str, Any]:
    raw = _read_file(BASE_DIR, "base.json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def get_base_scaffold_files() -> Dict[str, str]:
    files = _collect_base_files()
    logger.info("[scaffold] loaded base-vite-fastapi | files=%d", len(files))
    return files


def _normalize_path(path: str) -> str:
    norm = path.replace("\\", "/").lstrip("/")
    if norm == "frontend/src/App.js":
        return "frontend/src/App.jsx"
    if norm == "frontend/src/styles.css":
        return "frontend/src/styles/app.css"
    return norm


def is_acceptable_llm_content(path: str, content: str) -> bool:
    """Reject empty, stub, or incomplete LLM output so base scaffold is preserved."""
    if not content or not str(content).strip():
        return False

    stripped = str(content).strip()
    lower = stripped.lower()

    for marker in _STUB_MARKERS:
        if marker in lower:
            return False

    if path.endswith(".py"):
        if len(stripped.splitlines()) < 8:
            return False
        if path.endswith("routes.py") and "APIRouter" not in stripped:
            return False
        if path.endswith("models.py") and "class " not in stripped:
            return False
        if path.endswith("schemas.py") and "BaseModel" not in stripped:
            return False

    if path.endswith(("App.jsx", "App.js")):
        if len(stripped) < 250:
            return False
        if "apiFetch" not in stripped and "fetch(" not in stripped:
            return False
        if not any(k in stripped for k in ("onSubmit", "onClick", "<input", "<button")):
            return False

    if path.endswith(("app.css", "styles.css")):
        if len(stripped.splitlines()) < 30:
            return False

    return True


def merge_llm_into_base(base_files: Dict[str, str], llm_files: Dict[str, str]) -> Dict[str, str]:
    merged = dict(base_files)
    for path, content in llm_files.items():
        norm = _normalize_path(path)
        if is_frozen_path(norm):
            continue
        if not is_allowlisted_path(norm):
            continue
        if not is_acceptable_llm_content(norm, content):
            logger.info("[scaffold] rejected low-quality LLM output for %s — keeping base", norm)
            continue
        merged[norm] = content
    return merged


def _pick_primary_table(tables: list) -> str:
    names: list[str] = []
    for table in tables:
        if isinstance(table, dict):
            name = (table.get("table_name") or table.get("name") or "").lower()
            if name:
                names.append(name)

    for preferred in _ENTITY_PRIORITY:
        if preferred in names:
            return preferred

    for name in names:
        if name not in _SKIP_TABLES:
            return name

    return names[0] if names else "items"


def _table_by_name(tables: list, table_name: str) -> Any:
    for table in tables:
        if isinstance(table, dict):
            name = (table.get("table_name") or table.get("name") or "").lower()
            if name == table_name:
                return table
    return tables[0] if tables else {}


def normalize_architecture_for_codegen(
    architecture: Dict[str, Any] | None,
    api_contract: Any = None,
) -> Dict[str, Any]:
    """Merge standalone api_contract into architecture for codegen."""
    arch: Dict[str, Any] = dict(architecture or {})

    contract = arch.get("api_contract")
    if not contract and api_contract:
        if isinstance(api_contract, str):
            try:
                contract = json.loads(api_contract)
            except json.JSONDecodeError:
                contract = {}
        elif isinstance(api_contract, dict):
            contract = api_contract
    if contract and not arch.get("api_contract"):
        arch["api_contract"] = contract

    return arch


def extract_api_contract(architecture: Dict[str, Any] | None) -> Dict[str, Any]:
    arch = architecture or {}
    tables = (arch.get("database_schema") or {}).get("tables") or []
    table_name = _pick_primary_table(tables)
    primary_table = _table_by_name(tables, table_name)
    entity = table_name.rstrip("s") if table_name.endswith("s") else table_name

    api_contract = arch.get("api_contract") or {}
    if isinstance(api_contract, str):
        try:
            api_contract = json.loads(api_contract)
        except json.JSONDecodeError:
            api_contract = {}

    routes = api_contract.get("routes") or api_contract.get("endpoints") or []
    prefix = f"/{table_name}"

    if isinstance(api_contract.get("endpoints"), dict):
        for path in api_contract["endpoints"]:
            m = re.match(r"^/?([\w-]+)", str(path).lstrip("/"))
            if m and m.group(1) not in _SKIP_TABLES:
                prefix = f"/{m.group(1)}"
                break

    if routes and isinstance(routes, list) and isinstance(routes[0], dict):
        path = routes[0].get("path") or routes[0].get("url") or ""
        m = re.match(r"^/?([\w-]+)", str(path).lstrip("/"))
        if m and m.group(1) not in _SKIP_TABLES:
            prefix = f"/{m.group(1)}"

    return {
        "entity": entity,
        "table_name": table_name,
        "api_prefix": prefix,
        "fields": _fields_from_table(primary_table),
    }


def _fields_from_table(table: Any) -> list:
    if not isinstance(table, dict):
        return ["id", "title", "completed"]
    cols = table.get("columns") or table.get("fields") or []
    names = []
    for c in cols:
        if isinstance(c, dict):
            names.append(c.get("name", ""))
        elif isinstance(c, str):
            names.append(c)
    return [n for n in names if n] or ["id", "title", "completed"]


def adapt_scaffold_to_contract(files: Dict[str, str], contract: Dict[str, Any]) -> Dict[str, str]:
    """Rename base /items scaffold to match architecture contract (e.g. /tasks)."""
    prefix = contract.get("api_prefix", "/items")
    table_name = contract.get("table_name", "items")
    entity = contract.get("entity", "item")

    if prefix == "/items" and table_name == "items":
        return files

    adapted = dict(files)
    replacements = [
        ("/items", prefix),
        ('"items"', f'"{table_name}"'),
        ("'items'", f"'{table_name}'"),
        ("__tablename__ = \"items\"", f'__tablename__ = "{table_name}"'),
        ("__tablename__ = 'items'", f"__tablename__ = '{table_name}'"),
        ("ItemOut", f"{entity.title()}Out"),
        ("ItemCreate", f"{entity.title()}Create"),
        ("ItemUpdate", f"{entity.title()}Update"),
        ("class Item", f"class {entity.title()}"),
        ("Item)", f"{entity.title()})"),
        ("Item,", f"{entity.title()},"),
        ("Item.", f"{entity.title()}."),
        ("from models import Item", f"from models import {entity.title()}"),
    ]

    for path in list(adapted.keys()):
        if not (
            path.endswith(("App.jsx", "App.js", "routes.py", "models.py", "schemas.py"))
            or path.endswith("app.css")
        ):
            continue
        content = adapted[path]
        for old, new in replacements:
            content = content.replace(old, new)
        adapted[path] = content

    return adapted


def validate_generated_contract(files: Dict[str, str], contract: Dict[str, Any]) -> list[str]:
    errors: list[str] = []
    prefix = contract.get("api_prefix", "/items")
    routes = files.get("backend/routes.py", "")
    if not routes.strip():
        errors.append("backend/routes.py is empty")
    elif prefix not in routes and f'"{prefix}"' not in routes and f"'{prefix}'" not in routes:
        errors.append(f"backend/routes.py missing API prefix {prefix}")

    models = files.get("backend/models.py", "")
    if not models.strip() or "class " not in models:
        errors.append("backend/models.py is empty or missing models")

    schemas = files.get("backend/schemas.py", "")
    if not schemas.strip() or "BaseModel" not in schemas:
        errors.append("backend/schemas.py is empty or missing schemas")

    app_paths = [p for p in files if p.replace("\\", "/").endswith(("App.jsx", "App.js"))]
    for app_path in app_paths:
        app_src = files.get(app_path, "")
        if "apiFetch" not in app_src and "fetch(" not in app_src:
            errors.append(f"{app_path} does not call the API")
        if app_src and not any(k in app_src for k in ("button", "onSubmit", "onClick", "<input")):
            errors.append(f"{app_path} missing interactive UI")
        if "minimal app component" in app_src.lower() or "satisfy the build" in app_src.lower():
            errors.append(f"{app_path} is a stub placeholder")

    css = files.get("frontend/src/styles/app.css") or files.get("frontend/src/styles.css", "")
    if len(css.splitlines()) < 40:
        errors.append("frontend styles too minimal (< 40 lines)")

    return errors


def detect_template(requirement: str) -> Optional[str]:
    return BASE_TEMPLATE_NAME


def get_template_code_files(template_name: str | None = None) -> Optional[Dict[str, str]]:
    return get_base_scaffold_files()


def load_template(template_name: str | None = None) -> Optional[Dict[str, Any]]:
    return load_base_config()
