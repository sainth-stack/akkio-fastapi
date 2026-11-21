"""
Ensure MongoDB-backed dynamic app storage is ready (no local SQLite).

Called during codegen validation when project config defines tables.
"""
import logging
from typing import Any, Dict

from db.dynamic_app_store import ensure_indexes

logger = logging.getLogger("app_builder")


def ensure_project_db_initialized(project_name: str, config: Dict[str, Any]) -> None:
    """Register indexes for dynamic CRUD; data lives in MongoDB `dynamic_app_rows`."""
    if not config.get("tables"):
        return
    try:
        ensure_indexes()
        logger.info("[db_init] MongoDB dynamic storage ready for project=%s", project_name)
    except Exception as e:
        logger.warning("[db_init] MongoDB init failed for project=%s: %s", project_name, e)
