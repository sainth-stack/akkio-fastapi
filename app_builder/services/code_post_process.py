"""Shared post-processing for generated application files (codegen + LangGraph)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from app_builder.agents import dynamic_code_generator as dcg
from app_builder.services.app_spec_service import (
    build_app_spec,
    get_codegen_allowlist,
    validate_against_app_spec,
)
from app_builder.services.scaffold_service import (
    get_base_scaffold_files,
    is_frozen_path,
    merge_llm_into_base,
)
from app_builder.services.app_generators import apply_deterministic_fallback, generate_app_css

logger = logging.getLogger("app_builder")


def _is_vite_project(files: Dict[str, str]) -> bool:
    pkg = files.get("frontend/package.json", "")
    if "vite" in pkg.lower():
        return True
    return "frontend/vite.config.js" in files or "frontend/vite.config.ts" in files


def apply_design_tokens(files: Dict[str, str], design_tokens: Dict[str, Any] | None) -> None:
    """Inject styling-agent CSS into generated project files."""
    if not design_tokens or not isinstance(design_tokens, dict):
        return
    token_data = design_tokens.get("design_tokens") if isinstance(design_tokens.get("design_tokens"), dict) else design_tokens
    css = design_tokens.get("app_css")
    if not css or len(str(css).strip()) < 40:
        primary = (token_data.get("colors") or {}).get("primary", "#4f46e5")
        css = generate_app_css(str(primary))
    for css_path in ("frontend/src/styles/app.css", "frontend/src/styles.css"):
        if css_path in files or css_path.endswith("app.css"):
            files[css_path if css_path in files else "frontend/src/styles/app.css"] = css
            break
    else:
        files["frontend/src/styles/app.css"] = css


def post_process_generated_files(
    files: Dict[str, str],
    architecture: Dict[str, Any] | None = None,
    template_name: Optional[str] = None,
    uiux: str = "",
    requirement: str = "",
    prd: str = "",
    app_spec: Optional[Dict[str, Any]] = None,
    design_tokens: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Apply normalization/fixups. Merges LLM output into generic base shell."""
    spec = app_spec or build_app_spec(requirement, architecture, prd, uiux)
    allowlist = get_codegen_allowlist(spec)
    base = get_base_scaffold_files()
    files = dict(files or {})
    files = merge_llm_into_base(base, files, allowlist=allowlist, app_spec=spec)

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
    apply_design_tokens(files, design_tokens)
    dcg._ensure_complete_styles_css(files, uiux=uiux)

    for path in list(files.keys()):
        if is_frozen_path(path) and path in base:
            files[path] = base[path]

    if _is_vite_project(files):
        for junk in (
            "frontend/craco.config.js",
            "frontend/public/index.html",
            "frontend/src/index.js",
        ):
            files.pop(junk, None)

    return files


def should_use_template(requirement: str, prd: str, architecture: dict | None) -> bool:
    return True


def validate_codegen_output(
    files: Dict[str, str],
    architecture: Dict[str, Any] | None = None,
    requirement: str = "",
    prd: str = "",
    app_spec: Optional[Dict[str, Any]] = None,
) -> list[str]:
    spec = app_spec or build_app_spec(requirement, architecture, prd)
    return validate_against_app_spec(files, spec)


def ensure_valid_codegen_output(
    files: Dict[str, str],
    architecture: Dict[str, Any] | None = None,
    uiux: str = "",
    requirement: str = "",
    prd: str = "",
    app_spec: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, str], list[str]]:
    """
    Validate generated files. Apply deterministic fallback for known app kinds when LLM fails.
    """
    spec = app_spec or build_app_spec(requirement, architecture, prd, uiux)
    errors = validate_against_app_spec(files, spec)
    if not errors:
        return files, []

    kind = spec.get("app_kind", "custom")
    supported = (
        "static_quiz", "static", "quiz", "crud", "content", "form", "dashboard", "custom",
    )
    if errors and (kind in supported or spec.get("frontend_only")):
        logger.warning(
            "[codegen] LLM validation failed (%s) — applying deterministic %s fallback",
            "; ".join(errors[:3]),
            kind,
        )
        fallback_files = apply_deterministic_fallback(dict(files), spec, uiux=uiux, prd=prd)
        fallback_files = post_process_generated_files(
            fallback_files,
            architecture,
            uiux=uiux,
            requirement=requirement,
            prd=prd,
            app_spec=spec,
        )
        fb_errors = validate_against_app_spec(fallback_files, spec)
        if not fb_errors:
            return fallback_files, []

    logger.warning("[codegen] validation failed: %s", "; ".join(errors[:5]))
    return files, errors
