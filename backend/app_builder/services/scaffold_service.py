"""
Single base scaffold loader — generic React (Vite) + FastAPI infrastructure shell.
Domain code is 100% LLM-generated from App Spec.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, FrozenSet, Optional, Tuple

from app_builder.services.app_spec_service import (
    build_app_spec,
    get_codegen_allowlist,
    is_path_allowlisted,
    validate_against_app_spec,
)

logger = logging.getLogger("app_builder")

BASE_TEMPLATE_NAME = "base-vite-fastapi"
TEMPLATES_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
BASE_DIR = os.path.join(TEMPLATES_BASE, BASE_TEMPLATE_NAME)

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

_STUB_MARKERS = (
    "minimal app component",
    "satisfy the build",
    "placeholder app",
    "waiting for application ui from code generation",
    "generic vite + react shell",
)


def is_frozen_path(path: str) -> bool:
    norm = path.replace("\\", "/").lstrip("/")
    return norm in FROZEN_PATHS


def is_allowlisted_path(path: str, allowlist: Optional[Tuple[str, ...]] = None) -> bool:
    norm = path.replace("\\", "/").lstrip("/")
    if is_frozen_path(norm):
        return False
    prefixes = allowlist or get_codegen_allowlist(build_app_spec("", {}))
    return is_path_allowlisted(norm, prefixes)


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
    logger.info("[scaffold] loaded generic base-vite-fastapi | files=%d", len(files))
    return files


def _normalize_path(path: str) -> str:
    norm = path.replace("\\", "/").lstrip("/")
    if norm == "frontend/src/App.js":
        return "frontend/src/App.jsx"
    if norm == "frontend/src/styles.css":
        return "frontend/src/styles/app.css"
    return norm


def is_acceptable_llm_content(path: str, content: str, app_spec: Optional[Dict[str, Any]] = None) -> bool:
    """Reject empty, stub, or shell content."""
    if not content or not str(content).strip():
        return False

    spec = app_spec or {}
    frontend_only = spec.get("frontend_only", False)
    stripped = str(content).strip()
    lower = stripped.lower()

    for marker in _STUB_MARKERS:
        if marker in lower:
            return False

    if path.endswith(".py") and not frontend_only:
        if path.endswith("routes.py"):
            if len(stripped.splitlines()) < 8 or "APIRouter" not in stripped:
                return False
        elif path.endswith("models.py"):
            if "class " not in stripped or len(stripped.splitlines()) < 5:
                return False
        elif path.endswith("schemas.py"):
            if "BaseModel" not in stripped or len(stripped.splitlines()) < 5:
                return False

    if path.endswith(("App.jsx", "App.js")):
        if len(stripped) < 200:
            return False
        if "shell-notice" in lower:
            return False

    if path.startswith("frontend/src/components/") and path.endswith((".jsx", ".js")):
        if len(stripped) < 60:
            return False

    if path.endswith(("app.css", "styles.css")):
        if len(stripped.splitlines()) < 20:
            return False

    return True


def merge_llm_into_base(
    base_files: Dict[str, str],
    llm_files: Dict[str, str],
    allowlist: Optional[Tuple[str, ...]] = None,
    app_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    merged = dict(base_files)
    prefixes = allowlist or get_codegen_allowlist(app_spec or build_app_spec("", {}))
    for path, content in llm_files.items():
        norm = _normalize_path(path)
        if is_frozen_path(norm):
            continue
        if not is_path_allowlisted(norm, prefixes):
            continue
        if not is_acceptable_llm_content(norm, content, app_spec):
            logger.info("[scaffold] rejected LLM output for %s — keeping base shell", norm)
            continue
        merged[norm] = content
    return merged


def normalize_architecture_for_codegen(
    architecture: Optional[Dict[str, Any]],
    api_contract: Any = None,
) -> Dict[str, Any]:
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


def extract_api_contract(architecture: Optional[Dict[str, Any]], requirement: str = "", prd: str = "") -> Dict[str, Any]:
    """Backward-compatible wrapper — prefer build_app_spec()."""
    spec = build_app_spec(requirement, architecture, prd)
    return {
        "entity": spec.get("primary_entity"),
        "table_name": spec.get("primary_table"),
        "api_prefix": spec.get("api_prefix"),
        "fields": spec.get("fields", []),
        "app_kind": spec.get("app_kind"),
    }


def validate_generated_contract(
    files: Dict[str, str],
    contract: Dict[str, Any],
    requirement: str = "",
    architecture: Optional[Dict[str, Any]] = None,
    prd: str = "",
) -> list[str]:
    spec = build_app_spec(requirement, architecture or {}, prd)
    if contract.get("api_prefix"):
        spec["api_prefix"] = contract["api_prefix"]
    if contract.get("table_name"):
        spec["primary_table"] = contract["table_name"]
    return validate_against_app_spec(files, spec)


def detect_template(requirement: str) -> Optional[str]:
    return BASE_TEMPLATE_NAME


def get_template_code_files(template_name: Optional[str] = None) -> Optional[Dict[str, str]]:
    return get_base_scaffold_files()


def load_template(template_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    return load_base_config()


# ---------------------------------------------------------------------------
# Track-aware scaffold loader
# ---------------------------------------------------------------------------

_FRONTEND_ONLY_TEMPLATE_NAME = "base-frontend-vite-mui"
_FRONTEND_ONLY_DIR = os.path.join(TEMPLATES_BASE, _FRONTEND_ONLY_TEMPLATE_NAME)


def get_scaffold_files_by_track(track: str) -> Dict[str, str]:
    """Load the appropriate scaffold template based on the generation track.

    Args:
        track: "legacy" loads base-vite-fastapi (existing behaviour).
               "frontend_only" loads base-frontend-vite-mui (new Lovable-style track).
               Any other value falls back to "legacy".

    Returns:
        Dict mapping relative file paths to file contents.
    """
    if track == "frontend_only":
        if not os.path.isdir(_FRONTEND_ONLY_DIR):
            logger.error(
                "[scaffold] frontend_only template missing at %s — falling back to legacy",
                _FRONTEND_ONLY_DIR,
            )
            return get_base_scaffold_files()

        files: Dict[str, str] = {}
        exclude_dirs = {"__pycache__", "node_modules", ".git", "dist", "build"}
        for root, dirs, filenames in os.walk(_FRONTEND_ONLY_DIR):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            rel_root = os.path.relpath(root, _FRONTEND_ONLY_DIR)
            for fn in filenames:
                if fn.endswith(".pyc") or fn == "base.json":
                    continue
                rel_path = os.path.join(rel_root, fn) if rel_root != "." else fn
                rel_path = rel_path.replace("\\", "/")
                if not rel_path.startswith("frontend/"):
                    continue
                full = os.path.join(root, fn)
                try:
                    with open(full, "r", encoding="utf-8") as fh:
                        files[rel_path] = fh.read()
                except OSError:
                    pass
        logger.info("[scaffold] loaded frontend_only base-frontend-vite-mui | files=%d", len(files))
        return files

    # Default: legacy fullstack scaffold
    return get_base_scaffold_files()
