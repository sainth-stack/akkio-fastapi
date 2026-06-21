"""Postgres database layer — split by domain."""

from __future__ import annotations

from .app_builder import AppBuilderStore, get_app_builder_db
from .llm_settings import LLMSettingsMixin
from .multi_model import MultiModelMixin
from .postgres import PGDATABASE, PGHOST, PGPASSWORD, PGUSER, PostgresPool
from .tabular import TabularDataMixin


class PostgresDatabase(PostgresPool, MultiModelMixin, TabularDataMixin, LLMSettingsMixin):
    """Facade used by routes; combines connection pool and domain mixins."""


__all__ = [
    "AppBuilderStore",
    "PGDATABASE",
    "PGHOST",
    "PGPASSWORD",
    "PGUSER",
    "PostgresDatabase",
    "get_app_builder_db",
]
