"""MongoDB connection pool for Akkio platform (auth, app builder, usage)."""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote_plus

from fastapi import HTTPException
from pymongo import ASCENDING, MongoClient, ReturnDocument
from pymongo.database import Database

MONGODB_URI = os.getenv("MONGODB_URI") or ""
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME") or "akkio"

_client: Optional[MongoClient] = None
_client_lock = threading.Lock()


def validate_db_config() -> None:
    if not MONGODB_URI.strip():
        raise RuntimeError(
            "Missing MONGODB_URI. Set mongodb+srv://... in .env (see .env.example)."
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_client() -> MongoClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                validate_db_config()
                _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=20000)
    return _client


def get_database(name: Optional[str] = None) -> Database:
    return get_client()[name or MONGODB_DB_NAME]


def get_collection(name: str):
    return get_database()[name]


def next_seq(counter_name: str) -> int:
    coll = get_collection("counters")
    doc = coll.find_one_and_update(
        {"_id": counter_name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return int(doc["seq"])


def ping_database() -> dict[str, Any]:
    client = get_client()
    client.admin.command("ping")
    db = get_database()
    return {
        "status": "active",
        "database": db.name,
        "collections": db.list_collection_names()[:20],
    }


def close_client() -> None:
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


class MongoPool:
    """Drop-in-ish replacement for PostgresPool connection lifecycle."""

    _indexes_ready = False
    _index_lock = threading.Lock()

    def __init__(self) -> None:
        self.connection = self

    @staticmethod
    def _clean_name(name: str) -> str:
        try:
            return (name or "").split(".")[0]
        except Exception:
            return name

    @classmethod
    def _ensure_pool(cls) -> None:
        get_client()

    @classmethod
    def get_pool_status(cls) -> dict[str, Any]:
        try:
            return ping_database()
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def ensure_connection(self) -> None:
        self._ensure_pool()

    def close(self) -> None:
        return None

    @contextmanager
    def get_connection(self):
        self.ensure_connection()
        yield self

    @classmethod
    def close_all_connections(cls) -> None:
        close_client()

    def ensure_indexes(self) -> None:
        if self.__class__._indexes_ready:
            return
        with self.__class__._index_lock:
            if self.__class__._indexes_ready:
                return
            users = get_collection("auth_users")
            users.create_index([("email", ASCENDING), ("app", ASCENDING)], unique=True)
            roles = get_collection("auth_roles")
            roles.create_index([("name", ASCENDING)], unique=True)
            get_collection("auth_revoked_tokens").create_index([("jti", ASCENDING)], unique=True)
            get_collection("llm_settings").create_index([("email", ASCENDING)], unique=True)
            get_collection("user_usage").create_index([("email", ASCENDING)], unique=True)
            get_collection("multi_model_sessions").create_index([("session_id", ASCENDING)], unique=True)
            get_collection("dynamic_app_rows").create_index(
                [("project_id", ASCENDING), ("table", ASCENDING), ("id", ASCENDING)],
                unique=True,
            )
            self.__class__._indexes_ready = True

    @classmethod
    def build_uri_from_parts(
        cls,
        user: str,
        password: str,
        host: str,
        *,
        db: str = "",
        options: str = "retryWrites=true&w=majority",
    ) -> str:
        creds = f"{quote_plus(user)}:{quote_plus(password)}"
        base = host if host.startswith("mongodb") else f"mongodb+srv://{creds}@{host}"
        if base.startswith("mongodb+srv://") and "@" not in base.split("mongodb+srv://", 1)[1][:1]:
            base = f"mongodb+srv://{creds}@{host.lstrip('/')}"
        if db and "?" in base:
            return f"{base}/{db}?{options.split('?')[-1]}"
        if db:
            return f"{base}/{db}?{options}" if options else f"{base}/{db}"
        return f"{base}?{options}" if "?" not in base else base
