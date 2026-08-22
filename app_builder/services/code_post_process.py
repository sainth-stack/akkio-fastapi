"""Shared post-processing for generated application files (codegen + LangGraph)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app_builder.agents import dynamic_code_generator as dcg
from app_builder.services.scaffold_service import (
    adapt_scaffold_to_contract,
    extract_api_contract,
    get_base_scaffold_files,
    is_frozen_path,
    merge_llm_into_base,
    validate_generated_contract,
)


def _is_vite_project(files: Dict[str, str]) -> bool:
    pkg = files.get("frontend/package.json", "")
    if "vite" in pkg.lower():
        return True
    return "frontend/vite.config.js" in files or "frontend/vite.config.ts" in files


def post_process_generated_files(
    files: Dict[str, str],
    architecture: Dict[str, Any] | None = None,
    template_name: Optional[str] = None,
    uiux: str = "",
) -> Dict[str, str]:
    """Apply normalization/fixups. Vite projects skip CRA conversion."""
    if not files:
        return files

    base = get_base_scaffold_files()
    files = merge_llm_into_base(base, files)
    contract = extract_api_contract(architecture)
    files = adapt_scaffold_to_contract(files, contract)

    if not _is_vite_project(files):
        dcg._normalize_frontend_package_json_to_cra(files)

    dcg._ensure_node_compat(files)
    dcg._fix_frontend_map_safety(files)
    dcg._fix_backend_routes_import(files)
    if not _is_vite_project(files):
        dcg._ensure_vite_react_plugin_in_package_json(files)
    dcg._normalize_backend_requirements(files)
    dcg._ensure_sqlite_database_default(files)
    dcg._fix_backend_python_relative_imports(files)
    dcg._fix_sqlalchemy_uuid_imports(files)
    dcg._fix_backend_pydantic_and_common(files)
    dcg._ensure_database_tables_created(files)
    dcg._ensure_cors_in_backend(files)
    dcg._fix_frontend_backend_url_undefined(files)
    dcg._validate_and_fix_backend_imports(files)
    dcg._ensure_complete_styles_css(files, uiux=uiux)

    # Strip LLM overrides of frozen scaffold files
    for path in list(files.keys()):
        if is_frozen_path(path):
            if path in base:
                files[path] = base[path]

    # Remove CRA-only artifacts from Vite projects
    if _is_vite_project(files):
        for junk in (
            "frontend/craco.config.js",
            "frontend/public/index.html",
            "frontend/src/index.js",
        ):
            files.pop(junk, None)

    return files


def should_use_template(requirement: str, prd: str, architecture: dict | None) -> bool:
    """Always use the single Vite+FastAPI base scaffold."""
    return True


def validate_codegen_output(
    files: Dict[str, str],
    architecture: Dict[str, Any] | None = None,
) -> list[str]:
    contract = extract_api_contract(architecture)
    return validate_generated_contract(files, contract)


def ensure_valid_codegen_output(
    files: Dict[str, str],
    architecture: Dict[str, Any] | None = None,
    uiux: str = "",
) -> tuple[Dict[str, str], list[str]]:
    """
    Validate generated files. On failure, fall back to base scaffold adapted to contract.
    Returns (files, errors) — errors empty when output is valid.
    """
    errors = validate_codegen_output(files, architecture)
    if not errors:
        return files, []

    contract = extract_api_contract(architecture)
    logger = __import__("logging").getLogger("app_builder")
    logger.warning("[codegen] validation failed (%s) — using base scaffold fallback", "; ".join(errors[:3]))

    fallback = adapt_scaffold_to_contract(dict(get_base_scaffold_files()), contract)
    fallback = post_process_generated_files(fallback, architecture, uiux=uiux)
    fallback_errors = validate_codegen_output(fallback, architecture)
    if fallback_errors:
        return files, errors + fallback_errors
    return fallback, []
