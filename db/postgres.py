from __future__ import annotations

import os
import threading
from contextlib import contextmanager

import psycopg2
from fastapi import HTTPException
from psycopg2.pool import SimpleConnectionPool


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Set Postgres connection vars in .env (see .env.example)."
        )
    return value


PGHOST = os.getenv("PGHOST") or ""
PGDATABASE = os.getenv("PGDATABASE") or ""
PGUSER = os.getenv("PGUSER") or ""
PGPASSWORD = os.getenv("PGPASSWORD") or ""


def validate_db_config() -> None:
    """Fail fast when Postgres env is not configured."""
    _require_env("PGHOST")
    _require_env("PGDATABASE")
    _require_env("PGUSER")
    _require_env("PGPASSWORD")
    global PGHOST, PGDATABASE, PGUSER, PGPASSWORD
    PGHOST = os.environ["PGHOST"]
    PGDATABASE = os.environ["PGDATABASE"]
    PGUSER = os.environ["PGUSER"]
    PGPASSWORD = os.environ["PGPASSWORD"]


class PostgresPool:
    _pool: SimpleConnectionPool | None = None
    _pool_lock = threading.Lock()

    def __init__(self):
        self.connection = None
        self._tables_initialized = False

    @staticmethod
    def _clean_name(name: str) -> str:
        try:
            return (name or "").split(".")[0]
        except Exception:
            return name

    @classmethod
    def _ensure_pool(cls):
        if cls._pool is None:
            with cls._pool_lock:
                if cls._pool is None:
                    validate_db_config()
                    try:
                        cls._pool = SimpleConnectionPool(
                            minconn=2,
                            maxconn=20,
                            database=PGDATABASE,
                            user=PGUSER,
                            password=PGPASSWORD,
                            host=PGHOST,
                            port=int(os.getenv("PGPORT", "5432")),
                        )
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"DB Pool Init Error: {e}")

    def ensure_connection(self):
        try:
            if self.connection is None or self.connection.closed:
                self.__class__._ensure_pool()
                self.connection = self.__class__._pool.getconn()
                self.connection.autocommit = True
        except Exception as e:
            print(f"Error ensuring connection: {e}")
            raise

    def close(self):
        try:
            if (
                self.connection is not None
                and not self.connection.closed
                and self.__class__._pool is not None
            ):
                self.__class__._pool.putconn(self.connection)
        except Exception as e:
            print(f"Warning: Error returning connection to pool: {e}")
        finally:
            self.connection = None

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            self.__class__._ensure_pool()
            conn = self.__class__._pool.getconn()
            if conn is None:
                raise HTTPException(
                    status_code=503,
                    detail="Database connection pool exhausted. Please try again.",
                )
            conn.autocommit = True
            yield conn
        finally:
            if conn is not None and self.__class__._pool is not None:
                try:
                    self.__class__._pool.putconn(conn)
                except Exception as e:
                    print(f"Error returning connection to pool: {e}")

    @classmethod
    def get_pool_status(cls):
        if cls._pool is None:
            return {"status": "not_initialized"}
        return {
            "status": "active",
            "min_connections": 2,
            "max_connections": 20,
        }

    @classmethod
    def close_all_connections(cls):
        if cls._pool is not None:
            with cls._pool_lock:
                try:
                    cls._pool.closeall()
                    cls._pool = None
                except Exception as e:
                    print(f"Error closing connection pool: {e}")
