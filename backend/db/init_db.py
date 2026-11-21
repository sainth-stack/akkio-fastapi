"""Database schema initialization — called once at application startup."""

from __future__ import annotations

import logging

from db import MongoDatabase
from db.dynamic_app_store import ensure_indexes as ensure_dynamic_app_indexes

logger = logging.getLogger(__name__)


def init_schemas() -> None:
    MongoDatabase._ensure_pool()
    db = MongoDatabase()
    try:
        db.ensure_connection()
        db.ensure_indexes()
        ensure_dynamic_app_indexes()
        db.create_table()
        db.ensure_training_tables()
        db.create_llm_settings_table()
    finally:
        db.close()

    from api.auth.store import auth_store

    auth_store.init_schema()

    from db.app_builder import get_app_builder_db

    get_app_builder_db().init_schema()

    logger.info("Database schemas initialized (MongoDB)")


def shutdown_db() -> None:
    MongoDatabase.close_all_connections()
    logger.info("Database connection closed")
