"""
Project config extraction and persistence for single-backend architecture.
Builds config from BuilderState and writes to project_config.json.
"""
import json
import logging
import os
from typing import Dict, Any, List, Optional

from .runtime_paths import get_projects_dir, resolve_project_root

logger = logging.getLogger("app_builder")


def _extract_tables_from_contract(api_contract: Dict[str, Any]) -> List[str]:
    """Extract collection/table names from API contract paths (e.g. GET /users -> users)."""
    tables = set()
    endpoints = api_contract.get("endpoints") or {}
    for path_spec in endpoints:
        # path_spec is like "GET /users", "POST /todos", "GET /todos/{id}"
        parts = path_spec.split()
        if len(parts) >= 2:
            path = parts[1].strip("/")
            # Get first segment (e.g. "users" from "users" or "users/123")
            segment = path.split("/")[0]
            if segment and not segment.isdigit() and segment != "api":
                tables.add(segment)
    return sorted(tables)


def _extract_fields_from_contract(api_contract: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract field names per collection from API contract response schemas."""
    fields: Dict[str, List[str]] = {}
    endpoints = api_contract.get("endpoints") or {}
    for path_spec, spec in endpoints.items():
        parts = path_spec.split()
        if len(parts) < 2:
            continue
        path = parts[1].strip("/")
        segment = path.split("/")[0]
        if not segment or segment.isdigit() or segment == "api":
            continue
        response = spec.get("response") or {}
        if isinstance(response, dict):
            field_names = [k for k in response.keys() if k != "id"]
            if field_names and (segment not in fields or len(field_names) > len(fields.get(segment, []))):
                fields[segment] = field_names
    return fields


def _infer_handler_from_path(path: str) -> str:
    """Infer handler type from endpoint path for dynamic custom endpoints."""
    p = path.lower().replace("-", "_")
    if "translate" in p:
        return "llm_translate"
    if "generate" in p or "ideas" in p:
        return "llm_content"
    return "llm_content"


def _extract_custom_endpoints(
    api_contract: Dict[str, Any],
    tables: List[str],
    template_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Extract custom (non-CRUD) endpoints for any app. Returns { path: handler }.
    - From template: ideas-generator -> generate-ideas, generate; language-translator -> translate
    - From api_contract: POST /path where path not in tables -> infer handler from path name
    """
    custom: Dict[str, str] = {}
    tables_set = set(tables)

    # 1. Template-specific endpoints
    if template_name == "ideas-generator":
        custom["generate-ideas"] = "llm_content"
        custom["generate"] = "llm_content"
    elif template_name == "language-translator":
        custom["translate"] = "llm_translate"

    # 2. From api_contract: POST endpoints that are not CRUD collections
    endpoints = api_contract.get("endpoints") or {}
    for path_spec in endpoints:
        parts = path_spec.split()
        if len(parts) < 2 or parts[0].upper() != "POST":
            continue
        path = parts[1].strip("/")
        segment = path.split("/")[0]
        if not segment or segment.isdigit() or segment == "api":
            continue
        if segment in tables_set:
            continue  # CRUD collection, skip
        if segment not in custom:
            custom[segment] = _infer_handler_from_path(segment)

    return custom


def _extract_llm_prompt(structured_requirement: Dict[str, Any], architecture: Dict[str, Any]) -> Optional[str]:
    """Extract LLM system prompt for GenAI-style apps."""
    desc = structured_requirement.get("description") or structured_requirement.get("project_name") or ""
    # If it's an LLM/GenAI app (ideas generator, translator, etc.), use structured requirement
    gen_type = structured_requirement.get("gen_type") or ""
    if gen_type or "llm" in str(architecture).lower() or "gen" in str(architecture).lower():
        return f"You are a helpful assistant. {desc}"
    return None


def build_project_config(
    project_name: str,
    structured_requirement: Dict[str, Any],
    architecture: Dict[str, Any],
    api_contract: Dict[str, Any],
    db_schema: Any,
    template_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build project config from BuilderState components."""
    schema_text = ""
    if isinstance(db_schema, dict):
        schema_text = db_schema.get("schema") or str(db_schema)
    else:
        schema_text = str(db_schema) if db_schema else ""

    tables = _extract_tables_from_contract(api_contract)
    fields = _extract_fields_from_contract(api_contract)
    # Fallback: if no tables from contract, try entities from structured requirement
    if not tables and structured_requirement:
        entities = structured_requirement.get("entities") or []
        if isinstance(entities, list):
            for e in entities:
                if isinstance(e, dict) and e.get("name"):
                    tables.append(e["name"])
                elif isinstance(e, str):
                    tables.append(e)
        tables = sorted(set(tables))

    llm_prompt = _extract_llm_prompt(structured_requirement, architecture or {})

    # Custom endpoints: { path: handler } - dynamic for all app types (ideas-generator, translator, LLM-generated)
    custom_endpoints = _extract_custom_endpoints(api_contract, tables, template_name)

    config = {
        "projectId": project_name,
        "tables": tables,
        "fields": fields,
        "api_contract": api_contract,
        "db_schema": schema_text,
        "llmPrompt": llm_prompt,
        "custom_endpoints": custom_endpoints,
    }
    return config


def persist_project_config(project_name: str, config: Dict[str, Any]) -> str:
    """Write project_config.json to project root. Uses resolve_project_root for consistency."""
    project_root = resolve_project_root(project_name)
    config_path = os.path.join(project_root, "project_config.json")
    try:
        os.makedirs(project_root, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info("[project_config] persisted to %s", config_path)
        return config_path
    except Exception as e:
        logger.exception("[project_config] failed to persist: %s", e)
        raise


def _infer_config_from_project(project_root: str, project_name: str) -> Optional[Dict[str, Any]]:
    """Infer minimal project_config when missing. Detects ideas-generator/translator from frontend."""
    app_js_paths = [
        os.path.join(project_root, "frontend", "src", "App.js"),
        os.path.join(project_root, "frontend", "App.js"),
    ]
    content = ""
    for p in app_js_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                break
            except Exception:
                pass
    custom_endpoints: Dict[str, str] = {}
    if "/generate-ideas" in content:
        custom_endpoints = {"generate-ideas": "llm_content", "generate": "llm_content"}
    elif "/translate" in content:
        custom_endpoints = {"translate": "llm_translate"}
    config = {
        "projectId": project_name,
        "tables": [],
        "fields": {},
        "api_contract": {},
        "db_schema": "",
        "llmPrompt": None,
        "custom_endpoints": custom_endpoints,
    }
    return config


def get_project_config(project_name: str) -> Optional[Dict[str, Any]]:
    """Load project config from project_config.json. Uses resolve_project_root. Auto-creates if missing."""
    project_root = resolve_project_root(project_name)
    config_path = os.path.join(project_root, "project_config.json")
    try:
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # Project dir exists but config missing (legacy apps) - infer and persist
        if os.path.isdir(project_root):
            config = _infer_config_from_project(project_root, project_name)
            if config:
                persist_project_config(project_name, config)
                return config
    except Exception as e:
        logger.warning("[project_config] failed to load %s: %s", config_path, e)
    return None
