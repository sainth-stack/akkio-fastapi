"""
Initialize per-project SQLite DB during app generation.
Creates data/app.db and tables so backend works immediately on first run.
"""
import logging
import os
import sqlite3
from typing import Any, Dict, List

from .runtime_paths import resolve_project_root

logger = logging.getLogger("app_builder")


def _extract_fields_from_contract(api_contract: Dict[str, Any]) -> Dict[str, List[str]]:
    """Derive field names per collection from api_contract."""
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


def _build_schema_from_config(config: Dict[str, Any]) -> str:
    """Generate CREATE TABLE SQL from config."""
    tables = config.get("tables") or []
    fields = config.get("fields") or {}
    if not fields and config.get("api_contract"):
        fields = _extract_fields_from_contract(config["api_contract"])
    statements = []
    for table in tables:
        cols = fields.get(table, [])
        col_defs = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
        for col in cols:
            col_defs.append(f'"{col}" TEXT')
        statements.append(f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(col_defs)})')
    return "; ".join(statements)


def ensure_project_db_initialized(project_name: str, config: Dict[str, Any]) -> None:
    """Create data/app.db and tables for project. Called during validation step."""
    if not config.get("tables"):
        return  # LLM-only apps (ideas-generator, translator) have no tables
    project_root = resolve_project_root(project_name)
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, "app.db")
    schema_sql = config.get("db_schema") or ""
    if not schema_sql or "CREATE TABLE" not in schema_sql.upper():
        schema_sql = _build_schema_from_config(config)
    if not schema_sql.strip():
        return
    try:
        conn = sqlite3.connect(db_path)
        for stmt in (s.strip() for s in schema_sql.split(";") if s.strip()):
            if stmt.upper().startswith("CREATE"):
                try:
                    conn.execute(stmt)
                    conn.commit()
                except sqlite3.OperationalError as e:
                    if "already exists" not in str(e).lower():
                        logger.warning("[db_init] schema stmt failed: %s", e)
        conn.close()
        logger.info("[db_init] initialized DB for project=%s", project_name)
    except Exception as e:
        logger.warning("[db_init] failed for project=%s: %s", project_name, e)
