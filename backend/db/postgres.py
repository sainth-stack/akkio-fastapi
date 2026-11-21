"""
Legacy import path — Akkio platform uses MongoDB only.

Some older deployments import PostgresDatabase from this module. It now resolves to
MongoDatabase (see db/__init__.py). Do not add Postgres env vars.
"""
from __future__ import annotations

from db.mongodb import (
    MONGODB_DB_NAME as PGDATABASE,
    MongoPool as PostgresPool,
    close_client,
    get_database,
    ping_database,
    validate_db_config,
)

PGHOST = ""
PGUSER = ""
PGPASSWORD = ""


def __getattr__(name: str):
    if name == "PostgresDatabase":
        from db import MongoDatabase

        return MongoDatabase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PGDATABASE",
    "PGHOST",
    "PGPASSWORD",
    "PGUSER",
    "PostgresDatabase",
    "PostgresPool",
    "close_client",
    "get_database",
    "ping_database",
    "validate_db_config",
]
