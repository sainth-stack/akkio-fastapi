"""
Template Service — single base scaffold (React Vite + FastAPI).
Legacy multi-template routing removed; all apps use base-vite-fastapi.
"""
from app_builder.services.scaffold_service import (  # noqa: F401
    BASE_TEMPLATE_NAME,
    detect_template,
    extract_api_contract,
    get_base_scaffold_files,
    get_template_code_files,
    is_allowlisted_path,
    is_frozen_path,
    load_base_config,
    load_template,
    merge_llm_into_base,
    validate_generated_contract,
)

TEMPLATE_MAPPING = {BASE_TEMPLATE_NAME: {"template_path": BASE_TEMPLATE_NAME, "config": "base.json"}}
