"""Shared post-processing for generated application files (codegen + LangGraph)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app_builder.agents import dynamic_code_generator as dcg
from api.app_creator.cra_npm_patch import ensure_cra_build_env_file, patch_package_json_in_files


def post_process_generated_files(
    files: Dict[str, str],
    architecture: Dict[str, Any] | None = None,
    template_name: Optional[str] = None,
    uiux: str = "",
) -> Dict[str, str]:
    """Apply normalization/fixups so disk matches validated DB content."""
    if not files:
        return files
    dcg._normalize_frontend_package_json_to_cra(files)
    dcg._ensure_node_compat(files)
    dcg._fix_frontend_map_safety(files)
    dcg._fix_backend_routes_import(files)
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
    patch_package_json_in_files(files)
    ensure_cra_build_env_file(files)
    return files


def should_use_template(requirement: str, prd: str, architecture: dict | None) -> bool:
    """Skip stock templates when the user completed planning (PRD or architecture)."""
    if (prd or "").strip():
        return False
    if isinstance(architecture, dict) and len(architecture) > 0:
        return False
    try:
        from app_builder.services.template_service import detect_template

        return detect_template(requirement or "") is not None
    except Exception:
        return False
