"""MongoDB database layer for Akkio platform."""

from __future__ import annotations

from .app_builder import AppBuilderStore, get_app_builder_db
from .mongo_mixins import LLMSettingsMixin, MultiModelMixin, TabularDataMixin
from .mongodb import (
    MONGODB_DB_NAME,
    MONGODB_URI,
    MongoPool,
    close_client,
    get_database,
    ping_database,
    validate_db_config,
)

# Legacy names kept so imports keep working during migration.
PostgresDatabase = None  # set after class definition
PGHOST = ""
PGDATABASE = MONGODB_DB_NAME
PGUSER = ""
PGPASSWORD = ""


class MongoDatabase(MongoPool, MultiModelMixin, TabularDataMixin, LLMSettingsMixin):
    """Facade used by routes; MongoDB-backed."""


PostgresDatabase = MongoDatabase

__all__ = [
    "AppBuilderStore",
    "MongoDatabase",
    "PostgresDatabase",
    "MONGODB_URI",
    "MONGODB_DB_NAME",
    "PGDATABASE",
    "PGHOST",
    "PGPASSWORD",
    "PGUSER",
    "close_client",
    "get_app_builder_db",
    "get_database",
    "ping_database",
    "validate_db_config",
]
