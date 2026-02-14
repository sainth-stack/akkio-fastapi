import psycopg2
import pandas as pd
import pickle
import base64
import psycopg2.errors
from psycopg2.pool import SimpleConnectionPool
from datetime import datetime
from fastapi import HTTPException
import io
import json
import pickle
import threading
import time
from contextlib import contextmanager

class PostgresDatabase:
    _pool: SimpleConnectionPool | None = None
    _pool_lock = threading.Lock()

    def __init__(self):
        self.connection = None
        self._tables_initialized = False
    
    @staticmethod
    def _clean_name(name: str) -> str:
        try:
            return (name or "").split('.')[0]
        except Exception:
            return name

    @classmethod
    def _ensure_pool(cls):
        if cls._pool is None:
            with cls._pool_lock:
                if cls._pool is None:
                    try:
                        # Initialize connection pool with increased size for better concurrency
                        cls._pool = SimpleConnectionPool(
                            minconn=2,
                            maxconn=20,
                            database=PGDATABASE,
                            user=PGUSER,
                            password=PGPASSWORD,
                            host=PGHOST,
                            port=5432
                        )
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"DB Pool Init Error: {e}")

    def create_connection(self, user, password, database, host, port=5432, return_tables_info: bool = False):
        try:
            # Prefer the shared pool even when explicitly asked to connect
            if self.__class__._pool is None:
                self.__class__._pool = SimpleConnectionPool(
                    minconn=1,
                    maxconn=10,
                    database=database,
                    user=user,
                    password=password,
                    host=host,
                    port=port
                )
            self.connection = self.__class__._pool.getconn()
            self.connection.autocommit = True
            if return_tables_info:
                return self.get_tables_info()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB Connection Error: {e}")

    def ensure_connection(self):
        try:
            if self.connection is None or self.connection.closed:
                # Get a connection from the pool without noisy logs
                self.__class__._ensure_pool()
                self.connection = self.__class__._pool.getconn()
                self.connection.autocommit = True
        except Exception as e:
            print(f"Error ensuring connection: {e}")
            raise
    
    def close(self):
        """Return the connection to the pool."""
        try:
            if self.connection is not None and not self.connection.closed and self.__class__._pool is not None:
                self.__class__._pool.putconn(self.connection)
        except Exception as e:
            print(f"Warning: Error returning connection to pool: {e}")
        finally:
            self.connection = None

    @contextmanager
    def get_connection(self):
        """Context manager for safe database connection handling."""
        conn = None
        try:
            self.__class__._ensure_pool()
            conn = self.__class__._pool.getconn()
            if conn is None:
                raise HTTPException(status_code=503, detail="Database connection pool exhausted. Please try again.")
            conn.autocommit = True
            yield conn
        except Exception as e:
            print(f"Database connection error: {e}")
            raise
        finally:
            if conn is not None and self.__class__._pool is not None:
                try:
                    self.__class__._pool.putconn(conn)
                except Exception as e:
                    print(f"Error returning connection to pool: {e}")

    @classmethod
    def get_pool_status(cls):
        """Get current connection pool status for monitoring."""
        if cls._pool is None:
            return {"status": "not_initialized"}
        try:
            # Note: SimpleConnectionPool doesn't expose direct metrics, 
            # but we can provide basic info
            return {
                "status": "active",
                "min_connections": 2,
                "max_connections": 20,
                "note": "Pool is healthy and operational"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @classmethod
    def close_all_connections(cls):
        """Safely close all connections in the pool (for shutdown/cleanup)."""
        if cls._pool is not None:
            with cls._pool_lock:
                try:
                    cls._pool.closeall()
                    cls._pool = None
                    print("All database connections closed successfully")
                except Exception as e:
                    print(f"Error closing connection pool: {e}")

    def create_table(self):
        """Creates akio_data_fastapi with no unique constraints on email, allowing duplicates."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS akio_data_fastapi(
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) NOT NULL,
                    name VARCHAR(255),
                    lastupdate TIMESTAMP,
                    datecreated TIMESTAMP,
                    fileobj BYTEA,
                    rawfile BYTEA,
                    type VARCHAR(50),
                    subtype VARCHAR(50)
                )
            """)
            # Backfill columns for older installs
            try:
                cursor.execute("ALTER TABLE akio_data_fastapi ADD COLUMN IF NOT EXISTS type VARCHAR(50)")
                cursor.execute("ALTER TABLE akio_data_fastapi ADD COLUMN IF NOT EXISTS subtype VARCHAR(50)")
                cursor.execute("ALTER TABLE akio_data_fastapi ADD COLUMN IF NOT EXISTS rawfile BYTEA")
            except Exception:
                pass

    def create_training_tables(self):
        """Create tables for training jobs and trained models."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_jobs (
                    id SERIAL PRIMARY KEY,
                    job_id UUID NOT NULL,
                    email VARCHAR(255) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    target VARCHAR(255),
                    status VARCHAR(32) NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    message TEXT,
                    model_type VARCHAR(32),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(email, name, target, model_type)
                )
            """)
            
            # Multi-model training sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS multi_model_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    model_name VARCHAR(255) NOT NULL,
                    user_email VARCHAR(255) NOT NULL,
                    system_prompt TEXT NOT NULL,
                    status VARCHAR(32) NOT NULL DEFAULT 'pending',
                    progress INTEGER NOT NULL DEFAULT 0,
                    stage TEXT,
                    temperature FLOAT DEFAULT 0.0,
                    workflow TEXT,
                    output_format TEXT,
                    published BOOLEAN DEFAULT FALSE,
                    public_id VARCHAR(255),
                    published_at TIMESTAMP,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    completed_at TIMESTAMP,
                    UNIQUE(user_email, model_name)
                )
            """)
            
            # Multi-model files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS multi_model_files (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    file_name VARCHAR(500) NOT NULL,
                    file_type VARCHAR(50) NOT NULL,
                    storage_path TEXT,
                    vector_collection_id VARCHAR(255),
                    db_table_name VARCHAR(255),
                    processed BOOLEAN DEFAULT FALSE,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    FOREIGN KEY (session_id) REFERENCES multi_model_sessions(session_id) ON DELETE CASCADE
                )
            """)

            # Migrations for multi-model tables
            try:
                cursor.execute("ALTER TABLE multi_model_sessions ADD COLUMN IF NOT EXISTS temperature FLOAT DEFAULT 0.0")
                cursor.execute("ALTER TABLE multi_model_sessions ADD COLUMN IF NOT EXISTS workflow TEXT")
                cursor.execute("ALTER TABLE multi_model_sessions ADD COLUMN IF NOT EXISTS output_format TEXT")
                cursor.execute("ALTER TABLE multi_model_sessions ADD COLUMN IF NOT EXISTS published BOOLEAN DEFAULT FALSE")
                cursor.execute("ALTER TABLE multi_model_sessions ADD COLUMN IF NOT EXISTS public_id VARCHAR(255)")
                cursor.execute("ALTER TABLE multi_model_sessions ADD COLUMN IF NOT EXISTS published_at TIMESTAMP")
                cursor.execute("ALTER TABLE multi_model_files ADD COLUMN IF NOT EXISTS error_message TEXT")
            except Exception:
                pass

            # Ensure uniqueness of public_id when present (nullable unique index)
            try:
                cursor.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_indexes WHERE indexname = 'uq_multi_model_sessions_public_id'
                        ) THEN
                            CREATE UNIQUE INDEX uq_multi_model_sessions_public_id
                            ON multi_model_sessions(public_id)
                            WHERE public_id IS NOT NULL;
                        END IF;
                    END$$;
                """)
            except Exception:
                pass
            # App Builder created apps (store all section content for list/edit)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS app_builder_apps (
                    id SERIAL PRIMARY KEY,
                    user_email VARCHAR(255) NOT NULL,
                    app_name VARCHAR(255) NOT NULL,
                    prompt TEXT NOT NULL,
                    project_name VARCHAR(255) NOT NULL,
                    prd TEXT,
                    generated_uiux TEXT,
                    plan JSONB,
                    architecture JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_app_builder_apps_user_email ON app_builder_apps(user_email)")
                cursor.execute("ALTER TABLE app_builder_apps ADD COLUMN IF NOT EXISTS generated_uiux TEXT")
            except Exception:
                pass
            # App Builder codegen sessions: store multi-agent output + generated code JSON (same as documents)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS app_builder_codegen_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    project_name VARCHAR(255) NOT NULL,
                    requirement TEXT,
                    prd TEXT,
                    plan JSONB,
                    architecture JSONB,
                    generated_code_json JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_codegen_sessions_session_id ON app_builder_codegen_sessions(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_codegen_sessions_project_name ON app_builder_codegen_sessions(project_name)")
            except Exception:
                pass
            # App Builder deployments: store deployed app URLs and status
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS app_builder_deployments (
                    id SERIAL PRIMARY KEY,
                    app_id INTEGER REFERENCES app_builder_apps(id) ON DELETE CASCADE,
                    project_name VARCHAR(255) NOT NULL,
                    frontend_url VARCHAR(512),
                    backend_url VARCHAR(512),
                    frontend_port INTEGER,
                    backend_port INTEGER,
                    deployment_status VARCHAR(50) DEFAULT 'pending',
                    error_message TEXT,
                    deployed_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_deployments_app_id ON app_builder_deployments(app_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_deployments_project_name ON app_builder_deployments(project_name)")
            except Exception:
                pass
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trained_models (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    target VARCHAR(255),
                    model_type VARCHAR(32),
                    framework VARCHAR(64),
                    metrics JSONB,
                    model_obj BYTEA,
                    trained_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(email, name, target, model_type)
                )
            """)
            # Migrations for older installs: add columns/indexes if missing
            try:
                # Ensure target column exists
                cursor.execute("ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS target VARCHAR(255)")
                cursor.execute("ALTER TABLE trained_models ADD COLUMN IF NOT EXISTS target VARCHAR(255)")
                # Drop legacy unique constraints on (email, name) if they exist to avoid conflicts
                cursor.execute("ALTER TABLE training_jobs DROP CONSTRAINT IF EXISTS training_jobs_email_name_key")
                cursor.execute("ALTER TABLE trained_models DROP CONSTRAINT IF EXISTS trained_models_email_name_key")
                # Drop any legacy unique indexes on (email, name)
                cursor.execute("DROP INDEX IF EXISTS uq_training_jobs_email_name")
                cursor.execute("DROP INDEX IF EXISTS uq_trained_models_email_name")
                # Drop old unique indexes on (email,name,target)
                cursor.execute("DROP INDEX IF EXISTS uq_training_jobs_email_name_target")
                cursor.execute("DROP INDEX IF EXISTS uq_trained_models_email_name_target")
                # Create unique indexes if not exists
                cursor.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_indexes WHERE indexname = 'uq_training_jobs_email_name_target_modeltype'
                        ) THEN
                            CREATE UNIQUE INDEX uq_training_jobs_email_name_target_modeltype
                            ON training_jobs(email, name, target, model_type);
                        END IF;
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_indexes WHERE indexname = 'uq_trained_models_email_name_target_modeltype'
                        ) THEN
                            CREATE UNIQUE INDEX uq_trained_models_email_name_target_modeltype
                            ON trained_models(email, name, target, model_type);
                        END IF;
                    END$$;
                """)
                # Performance indexes for quick latest status lookups
                cursor.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_indexes WHERE indexname = 'idx_training_jobs_email_name_updated'
                        ) THEN
                            CREATE INDEX idx_training_jobs_email_name_updated
                            ON training_jobs(email, name, updated_at DESC);
                        END IF;
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_indexes WHERE indexname = 'idx_training_jobs_email_name_target_modeltype_updated'
                        ) THEN
                            CREATE INDEX idx_training_jobs_email_name_target_modeltype_updated
                            ON training_jobs(email, name, target, model_type, updated_at DESC);
                        END IF;
                    END$$;
                """)
            except Exception:
                pass
            # Dataset schema storage (LLM-detected columns)
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS dataset_schemas (
                        id SERIAL PRIMARY KEY,
                        email VARCHAR(255) NOT NULL,
                        name VARCHAR(255) NOT NULL,
                        predictable_columns JSONB,
                        forecastable_columns JSONB,
                        columns JSONB,
                        total_columns INTEGER,
                        detected_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(email, name)
                    )
                """)
                cursor.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_indexes WHERE indexname = 'idx_dataset_schemas_email_name'
                        ) THEN
                            CREATE INDEX idx_dataset_schemas_email_name
                            ON dataset_schemas(email, name);
                        END IF;
                    END$$;
                """)
            except Exception:
                pass
        self._tables_initialized = True

    def ensure_training_tables(self):
        self.ensure_connection()
        if not self._tables_initialized:
            try:
                self.create_training_tables()
            except Exception as exc:
                print(f"Error creating training tables: {exc}")
                raise

    def create_reports_table(self):
        """Creates reports_fastapi table with url, title, and description fields."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports_fastapi (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255),
                    url TEXT,
                    title VARCHAR(500),
                    description TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

    def create_llm_settings_table(self):
        """Creates llm_settings table for storing user-specific API keys and models."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_settings (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    provider VARCHAR(50) DEFAULT 'openai',
                    api_key TEXT,
                    model_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            # Add provider column if it doesn't exist (for existing tables)
            try:
                cursor.execute("""
                    ALTER TABLE llm_settings 
                    ADD COLUMN IF NOT EXISTS provider VARCHAR(50) DEFAULT 'openai'
                """)
            except Exception:
                pass

    def _ensure_type_columns(self):
        """Ensure type/subtype/rawfile columns exist on akio_data_fastapi."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            try:
                cursor.execute("ALTER TABLE akio_data_fastapi ADD COLUMN IF NOT EXISTS type VARCHAR(50)")
                cursor.execute("ALTER TABLE akio_data_fastapi ADD COLUMN IF NOT EXISTS subtype VARCHAR(50)")
                cursor.execute("ALTER TABLE akio_data_fastapi ADD COLUMN IF NOT EXISTS rawfile BYTEA")
            except Exception:
                pass

    def insert_or_update(self, email, data, tb_name, data_type: str | None = None, data_subtype: str | None = None, raw_bytes: bytes | None = None):
        """Insert or update a row in akio_data_fastapi. Allows multiple rows per email."""
        self.ensure_connection()
        # Make sure optional columns exist
        self._ensure_type_columns()
        tb_name_clean = self._clean_name(tb_name)
        blob_data = pickle.dumps(data)
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT id FROM akio_data_fastapi WHERE email = %s AND name = %s", (email, tb_name_clean))
            existing = cursor.fetchone()
            if existing:
                try:
                    cursor.execute("""
                        UPDATE akio_data_fastapi
                        SET lastupdate = %s, fileobj = %s, rawfile = COALESCE(%s, rawfile), type = COALESCE(%s, type), subtype = COALESCE(%s, subtype)
                        WHERE email = %s AND name = %s
                    """, (datetime.now(), psycopg2.Binary(blob_data), psycopg2.Binary(raw_bytes) if raw_bytes is not None else None, data_type, data_subtype, email, tb_name_clean))
                except Exception:
                    # Fallback for older schema
                    cursor.execute("""
                        UPDATE akio_data_fastapi
                        SET lastupdate = %s, fileobj = %s
                        WHERE email = %s AND name = %s
                    """, (datetime.now(), psycopg2.Binary(blob_data), email, tb_name_clean))
                return "updated"
            else:
                try:
                    cursor.execute("""
                        INSERT INTO akio_data_fastapi (email, name, lastupdate, datecreated, fileobj, rawfile, type, subtype)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (email, tb_name_clean, datetime.now(), datetime.now(), psycopg2.Binary(blob_data), psycopg2.Binary(raw_bytes) if raw_bytes is not None else None, data_type, data_subtype))
                except Exception:
                    # Fallback for older schema
                    cursor.execute("""
                        INSERT INTO akio_data_fastapi (email, name, lastupdate, datecreated, fileobj)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (email, tb_name_clean, datetime.now(), datetime.now(), psycopg2.Binary(blob_data)))
                return "inserted"

    # -------- Training jobs and models helpers --------
    def upsert_training_job(self, email: str, name: str, job_id: str, status: str, progress: int = 0, message: str = None, model_type: str | None = None, target: str | None = None):
        self.ensure_training_tables()
        name = self._clean_name(name)
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO training_jobs (job_id, email, name, target, status, progress, message, model_type, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (email, name, target, model_type) DO UPDATE SET
                    job_id = EXCLUDED.job_id,
                    status = EXCLUDED.status,
                    progress = EXCLUDED.progress,
                    message = EXCLUDED.message,
                    model_type = EXCLUDED.model_type,
                    updated_at = NOW()
            """, (job_id, email, name, target, status, progress, message, model_type))

    def update_training_progress(self, email: str, name: str, progress: int, message: str = None, target: str | None = None, model_type: str | None = None):
        self.ensure_training_tables()
        name = self._clean_name(name)
        with self.connection.cursor() as cursor:
            cursor.execute("""
                UPDATE training_jobs
                SET progress = %s, message = %s, updated_at = NOW()
                WHERE email = %s AND name = %s AND (target IS NOT DISTINCT FROM %s) AND (model_type IS NOT DISTINCT FROM %s)
            """, (progress, message, email, name, target, model_type))

    def complete_training_job(self, email: str, name: str, message: str = None, model_type: str | None = None, target: str | None = None):
        self.ensure_training_tables()
        name = self._clean_name(name)
        with self.connection.cursor() as cursor:
            if model_type is not None:
                cursor.execute("""
                    UPDATE training_jobs
                    SET status = 'completed', progress = 100, message = %s, model_type = %s, updated_at = NOW()
                    WHERE email = %s AND name = %s AND (target IS NOT DISTINCT FROM %s) AND (model_type IS NOT DISTINCT FROM %s)
                """, (message, model_type, email, name, target, model_type))
            else:
                cursor.execute("""
                    UPDATE training_jobs
                    SET status = 'completed', progress = 100, message = %s, updated_at = NOW()
                    WHERE email = %s AND name = %s AND (target IS NOT DISTINCT FROM %s) AND (model_type IS NOT DISTINCT FROM %s)
                """, (message, email, name, target, model_type))

    def fail_training_job(self, email: str, name: str, message: str = None, target: str | None = None, model_type: str | None = None):
        self.ensure_training_tables()
        name = self._clean_name(name)
        with self.connection.cursor() as cursor:
            cursor.execute("""
                UPDATE training_jobs
                SET status = 'failed', message = %s, updated_at = NOW()
                WHERE email = %s AND name = %s AND (target IS NOT DISTINCT FROM %s) AND (model_type IS NOT DISTINCT FROM %s)
            """, (message, email, name, target, model_type))

    def get_training_status(self, email: str, name: str, target: str | None = None, model_type: str | None = None):
        self.ensure_training_tables()
        name = self._clean_name(name)
        with self.connection.cursor() as cursor:
            if target is None and model_type is None:
                cursor.execute("""
                    SELECT job_id, status, progress, COALESCE(message, ''), COALESCE(model_type, ''), created_at, updated_at, target
                    FROM training_jobs
                    WHERE email = %s AND name = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (email, name))
            else:
                cursor.execute("""
                    SELECT job_id, status, progress, COALESCE(message, ''), COALESCE(model_type, ''), created_at, updated_at, target
                    FROM training_jobs
                    WHERE email = %s AND name = %s AND (target IS NOT DISTINCT FROM %s) AND (model_type IS NOT DISTINCT FROM %s)
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (email, name, target, model_type))
            row = cursor.fetchone()
            if not row:
                return None
            keys = ["job_id", "status", "progress", "message", "model_type", "created_at", "updated_at", "target"]
            return dict(zip(keys, row))

    def save_trained_model(self, email: str, name: str, model_type: str, framework: str, metrics: dict, model_bytes: bytes, target: str | None = None):
        self.ensure_training_tables()
        name = self._clean_name(name)
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO trained_models (email, name, target, model_type, framework, metrics, model_obj, trained_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, NOW(), NOW())
                ON CONFLICT (email, name, target, model_type) DO UPDATE SET
                    model_type = EXCLUDED.model_type,
                    framework = EXCLUDED.framework,
                    metrics = EXCLUDED.metrics,
                    model_obj = EXCLUDED.model_obj,
                    updated_at = NOW()
            """, (email, name, target, model_type, framework, json.dumps(metrics or {}), psycopg2.Binary(model_bytes)))

    def get_existing_model(self, email: str, name: str, target: str | None = None, model_type: str | None = None):
        self.ensure_training_tables()
        name = self._clean_name(name)
        with self.connection.cursor() as cursor:
            if target is None and model_type is None:
                cursor.execute("""
                    SELECT model_type, framework, metrics, model_obj, trained_at, updated_at, target
                    FROM trained_models
                    WHERE email = %s AND name = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (email, name))
            else:
                cursor.execute("""
                    SELECT model_type, framework, metrics, model_obj, trained_at, updated_at, target
                    FROM trained_models
                    WHERE email = %s AND name = %s AND (target IS NOT DISTINCT FROM %s) AND (model_type IS NOT DISTINCT FROM %s)
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (email, name, target, model_type))
            row = cursor.fetchone()
            if not row:
                return None
            keys = ["model_type", "framework", "metrics", "model_obj", "trained_at", "updated_at", "target"]
            result = dict(zip(keys, row))
            try:
                result["metrics"] = json.loads(result["metrics"]) if isinstance(result["metrics"], str) else result["metrics"]
            except Exception:
                pass
            return result

    def get_existing_model_info(self, email: str, name: str, target: str | None = None):
        """Lightweight fetch: get trained model metadata WITHOUT model_obj bytes."""
        self.ensure_training_tables()
        name = self._clean_name(name)
        with self.connection.cursor() as cursor:
            if target is None:
                cursor.execute("""
                    SELECT model_type, framework, metrics, trained_at, updated_at, target
                    FROM trained_models
                    WHERE email = %s AND name = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (email, name))
            else:
                cursor.execute("""
                    SELECT model_type, framework, metrics, trained_at, updated_at, target
                    FROM trained_models
                    WHERE email = %s AND name = %s AND (target IS NOT DISTINCT FROM %s)
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (email, name, target))
            row = cursor.fetchone()
            if not row:
                return None
            keys = ["model_type", "framework", "metrics", "trained_at", "updated_at", "target"]
            result = dict(zip(keys, row))
            try:
                result["metrics"] = json.loads(result["metrics"]) if isinstance(result["metrics"], str) else result["metrics"]
            except Exception:
                pass
            return result

    def get_trained_and_pending_targets(self, email: str, name: str):
        """
        Return two arrays for fast UI tracking:
        - trained: targets present in trained_models (distinct), excluding NULL
        - pending: targets with latest job status in ('queued','running'); these are removed from trained
        """
        self.ensure_training_tables()
        name = self._clean_name(name)
        trained = []
        pending = []
        with self.connection.cursor() as cursor:
            # trained targets
            cursor.execute("""
                SELECT DISTINCT target
                FROM trained_models
                WHERE email = %s AND name = %s AND target IS NOT NULL
            """, (email, name))
            trained = [r[0] for r in cursor.fetchall() if r and r[0] is not None]
            # latest job status per target
            cursor.execute("""
                SELECT DISTINCT ON (target) target, status
                FROM training_jobs
                WHERE email = %s AND name = %s AND target IS NOT NULL AND model_type IS NOT NULL
                ORDER BY target, updated_at DESC
            """, (email, name))
            rows = cursor.fetchall()
            # Build pending set from latest queued/running
            pending_set = set()
            ordered_pending = []
            for t, st in rows:
                if st in ("queued", "running"):
                    if t not in pending_set:
                        pending_set.add(t)
                        ordered_pending.append(t)
            # Exclude pending targets from trained list
            filtered_trained = [t for t in trained if t not in pending_set]
            return {"trained": filtered_trained, "pending": ordered_pending}

    def get_overall_training_summary(self, email: str, name: str):
        """
        Compute an overall status summary across targets:
        - total_targets, completed, running, queued, failed
        - overall_progress: average of latest per-target progress
        """
        self.ensure_training_tables()
        name = self._clean_name(name)
        with self.connection.cursor() as cursor:
            # For each (target, model_type), get the latest row
            cursor.execute("""
                SELECT DISTINCT ON (target, model_type) target, model_type, status, progress
                FROM training_jobs
                WHERE email = %s AND name = %s AND target IS NOT NULL AND model_type IS NOT NULL
                ORDER BY target, model_type, updated_at DESC
            """, (email, name))
            rows = cursor.fetchall()
            if not rows:
                return {"total_targets": 0, "completed": 0, "running": 0, "queued": 0, "failed": 0, "overall_progress": 0}
            total = len(rows)
            counts = {"completed": 0, "running": 0, "queued": 0, "failed": 0}
            progresses = []
            for _, _, st, prog in rows:
                st_norm = (st or "").lower()
                if st_norm in counts:
                    counts[st_norm] += 1
                elif st_norm == "failed":
                    counts["failed"] += 1
                elif st_norm == "completed":
                    counts["completed"] += 1
                elif st_norm == "running":
                    counts["running"] += 1
                elif st_norm == "queued":
                    counts["queued"] += 1
                # progress could be None
                try:
                    progresses.append(int(prog or 0))
                except Exception:
                    progresses.append(0)
            overall = int(round(sum(progresses) / max(1, total)))
            return {
                "total_targets": total,
                **counts,
                "overall_progress": overall
            }

    # -------- Dataset schema (LLM-driven) --------
    def save_dataset_schema(self, email: str, name: str, predictable_columns: list, forecastable_columns: list, columns: list, total_columns: int | None = None):
        self.ensure_training_tables()
        name = self._clean_name(name)
        total_columns = int(total_columns if total_columns is not None else len(columns or []))
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO dataset_schemas (email, name, predictable_columns, forecastable_columns, columns, total_columns, detected_at)
                VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, NOW())
                ON CONFLICT (email, name) DO UPDATE SET
                    predictable_columns = EXCLUDED.predictable_columns,
                    forecastable_columns = EXCLUDED.forecastable_columns,
                    columns = EXCLUDED.columns,
                    total_columns = EXCLUDED.total_columns,
                    detected_at = NOW()
            """, (email, name, json.dumps(predictable_columns or []), json.dumps(forecastable_columns or []), json.dumps(columns or []), total_columns))

    def get_dataset_schema(self, email: str, name: str):
        self.ensure_training_tables()
        name = self._clean_name(name)
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT predictable_columns, forecastable_columns, columns, total_columns, detected_at
                FROM dataset_schemas
                WHERE email = %s AND name = %s
            """, (email, name))
            row = cursor.fetchone()
            if not row:
                return None
            keys = ["predictable_columns", "forecastable_columns", "columns", "total_columns", "detected_at"]
            result = dict(zip(keys, row))
            try:
                result["predictable_columns"] = json.loads(result["predictable_columns"]) if isinstance(result["predictable_columns"], str) else result["predictable_columns"]
            except Exception:
                pass
            try:
                result["forecastable_columns"] = json.loads(result["forecastable_columns"]) if isinstance(result["forecastable_columns"], str) else result["forecastable_columns"]
            except Exception:
                pass
            try:
                result["columns"] = json.loads(result["columns"]) if isinstance(result["columns"], str) else result["columns"]
            except Exception:
                pass
            return result

    def read(self, table_name):
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, email, name, lastupdate, datecreated, fileobj, rawfile "
                "FROM akio_data_fastapi WHERE name = %s LIMIT 1",
                (table_name,)
            )
            row = cursor.fetchone()

        if not row:
            return None

        cols = ['id', 'email', 'name', 'lastupdate', 'datecreated', 'fileobj', 'rawfile']
        return pd.DataFrame([row], columns=cols)

    def get_tables_info(self,table_name):
        """Returns table info, gracefully handles missing table."""
        df = self.read(table_name)
        if df.empty:
            return {}
        df['lastupdate'] = df['lastupdate'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
        df['datecreated'] = df['datecreated'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
        return df.iloc[:, :-1].to_dict(orient="records")


    def get_user_tables(self, user):
        """Get user tables with proper connection management."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT name FROM akio_data_fastapi WHERE email = %s", (user,))
                    rows = cursor.fetchall()
                    return [row[0] for row in rows]
        except Exception as e:
            print(f"Error getting user tables for {user}: {e}")
            raise

    def get_user_items(self, email: str):
        """
        Return list of dicts with name, type, subtype for a given user.
        Gracefully handles missing columns by returning None for missing fields.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    try:
                        cursor.execute("SELECT name, type, subtype FROM akio_data_fastapi WHERE email = %s", (email,))
                        rows = cursor.fetchall()
                        results = []
                        for row in rows:
                            # row may contain only name if columns missing; handle lengths
                            if len(row) == 3:
                                results.append({"name": row[0], "type": row[1], "subtype": row[2]})
                            elif len(row) == 1:
                                results.append({"name": row[0], "type": None, "subtype": None})
                            else:
                                # unexpected shape
                                results.append({"name": row[0] if row else None, "type": None, "subtype": None})
                        return results
                    except Exception:
                        # Fallback if columns missing
                        cursor.execute("SELECT name FROM akio_data_fastapi WHERE email = %s", (email,))
                        rows = cursor.fetchall()
                        return [{"name": row[0], "type": None, "subtype": None} for row in rows]
        except Exception as e:
            print(f"Error getting user items for {email}: {e}")
            raise

    def get_table_data(self, table_name):
        """Safely retrieve table data with proper connection management."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    clean_name = self._clean_name(table_name)
                    cursor.execute(
                        "SELECT fileobj FROM akio_data_fastapi WHERE name = %s LIMIT 1",
                        (clean_name,)
                    )
                    row = cursor.fetchone()
                    
                    if not row or row[0] is None:
                        return pd.DataFrame()
                    
                    bytes_data = row[0]
                    table_data = pickle.loads(bytes(bytes_data))
                    if isinstance(table_data, pd.DataFrame):
                        return table_data
                    return pd.DataFrame(table_data)
        except Exception as e:
            print(f"Error getting table data for {table_name}: {e}")
            raise

    def get_raw_file(self, table_name) -> bytes | None:
        """Return rawfile bytes for a given name if available."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    clean_name = self._clean_name(table_name)
                    cursor.execute(
                        "SELECT rawfile FROM akio_data_fastapi WHERE name = %s LIMIT 1",
                        (clean_name,)
                    )
                    row = cursor.fetchone()
                    if not row or row[0] is None:
                        return None
                    return bytes(row[0])
        except Exception as e:
            print(f"Error getting raw file for {table_name}: {e}")
            return None

    def delete_tables_data(self, email, table_names):
        """Delete tables with proper connection management."""
        if not table_names:
            return "No table names provided"
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    placeholders = ','.join(['%s'] * len(table_names))
                    cursor.execute(
                        f"DELETE FROM akio_data_fastapi WHERE email = %s AND name IN ({placeholders})",
                        [email] + table_names
                    )
                    return f"{cursor.rowcount} records deleted"
        except Exception as e:
            print(f"Error deleting tables for {email}: {e}")
            raise

    def delete_all_tables_data(self, email):
        """Delete all tables for user with proper connection management."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM akio_data_fastapi WHERE email = %s", (email,))
                    return f"{cursor.rowcount} records deleted"
        except Exception as e:
            print(f"Error deleting all tables for {email}: {e}")
            raise

    
    def insert_report(self, email, url, title=None, description=None):
        """Insert report with proper connection management."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO reports_fastapi (email, url, title, description)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id, email, url, title, description, created_at
                    """, (email, url, title, description))
                    result = cursor.fetchone()
                    return dict(zip([d[0] for d in cursor.description], result))
        except Exception as e:
            print(f"Error inserting report for {email}: {e}")
            raise

    def update_report(self, email, url, title=None, description=None):
        """Update the report URL, title, and description for a given email."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE reports_fastapi
                        SET url = %s, title = %s, description = %s, updated_at = NOW()
                        WHERE email = %s
                        RETURNING id, email, url, title, description, updated_at
                    """, (url, title, description, email))
                    result = cursor.fetchone()
                    return dict(zip([d[0] for d in cursor.description], result)) if result else None
        except Exception as e:
            print(f"Error updating report for {email}: {e}")
            raise

    def get_report_by_email(self, email):
        """Get reports with proper connection management."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id, email, url, title, description, created_at, updated_at
                        FROM reports_fastapi
                        WHERE email = %s
                    """, (email,))
                    rows = cursor.fetchall()
                    return [dict(zip([d[0] for d in cursor.description], row)) for row in rows]
        except Exception as e:
            print(f"Error getting reports for {email}: {e}")
            raise

    def get_reports_by_ids_and_email(self, email, report_ids):
        """Get reports by IDs with proper connection management."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    query = """
                        SELECT id, email, url, title, description, created_at, updated_at
                        FROM reports_fastapi
                        WHERE email = %s
                        AND id = ANY(%s)
                    """
                    cursor.execute(query, (email, report_ids))
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            print(f"Error getting reports by IDs for {email}: {e}")
            raise

    def delete_report_by_email(self, email):
        """Delete reports with proper connection management."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM reports_fastapi WHERE email = %s", (email,))
                    return f"{cursor.rowcount} reports_fastapi deleted"
        except Exception as e:
            print(f"Error deleting reports for {email}: {e}")
            raise

    def delete_user_report_by_id(self, email, report_id):
        """Delete report by ID with proper connection management."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM reports_fastapi WHERE id = %s AND email = %s", (report_id, email))
                    return f"{cursor.rowcount} reports_fastapi deleted"
        except Exception as e:
            print(f"Error deleting report {report_id} for {email}: {e}")
            raise

    def delete_all_tables(self):
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS reports_fastapi")
            cursor.execute("DROP TABLE IF EXISTS akio_data_fastapi")
            print("All tables dropped successfully.")
            return "All tables dropped"

    # -------- Multi-Model Training Methods --------
    def create_multi_model_session(self, session_id: str, model_name: str, user_email: str, system_prompt: str, temperature: float = 0.0, workflow: str = None, output_format: str = None):
        """Create a new multi-model training session."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO multi_model_sessions (session_id, model_name, user_email, system_prompt, temperature, workflow, output_format, status, progress, stage, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', 0, 'Initializing', NOW(), NOW())
                ON CONFLICT (user_email, model_name) DO UPDATE SET
                    session_id = EXCLUDED.session_id,
                    system_prompt = EXCLUDED.system_prompt,
                    temperature = EXCLUDED.temperature,
                    workflow = EXCLUDED.workflow,
                    output_format = EXCLUDED.output_format,
                    status = 'pending',
                    progress = 0,
                    stage = 'Initializing',
                    error_message = NULL,
                    updated_at = NOW(),
                    completed_at = NULL
            """, (session_id, model_name, user_email, system_prompt, temperature, workflow, output_format))

    def add_multi_model_file(self, session_id: str, file_name: str, file_type: str, storage_path: str = None, 
                            vector_collection_id: str = None, db_table_name: str = None, error_message: str = None):
        """Add a file to a multi-model training session."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO multi_model_files (session_id, file_name, file_type, storage_path, vector_collection_id, db_table_name, error_message, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """, (session_id, file_name, file_type, storage_path, vector_collection_id, db_table_name, error_message))

    def update_multi_model_progress(self, session_id: str, progress: int, stage: str, status: str = None):
        """Update progress for a multi-model training session."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            if status:
                cursor.execute("""
                    UPDATE multi_model_sessions
                    SET progress = %s, stage = %s, status = %s, updated_at = NOW()
                    WHERE session_id = %s
                """, (progress, stage, status, session_id))
            else:
                cursor.execute("""
                    UPDATE multi_model_sessions
                    SET progress = %s, stage = %s, updated_at = NOW()
                    WHERE session_id = %s
                """, (progress, stage, session_id))

    def complete_multi_model_training(self, session_id: str):
        """Mark a multi-model training session as completed."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                UPDATE multi_model_sessions
                SET status = 'completed', progress = 100, stage = 'Training completed', updated_at = NOW(), completed_at = NOW()
                WHERE session_id = %s
            """, (session_id,))

    def fail_multi_model_training(self, session_id: str, error_message: str):
        """Mark a multi-model training session as failed."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                UPDATE multi_model_sessions
                SET status = 'failed', error_message = %s, updated_at = NOW()
                WHERE session_id = %s
            """, (error_message, session_id))

    def get_multi_model_session(self, session_id: str = None, user_email: str = None, model_name: str = None):
        """Get multi-model session details."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            if session_id:
                cursor.execute("""
                    SELECT session_id, model_name, user_email, system_prompt, status, progress, stage, temperature, workflow, output_format,
                           published, public_id, published_at,
                           error_message, created_at, updated_at, completed_at
                    FROM multi_model_sessions
                    WHERE session_id = %s
                """, (session_id,))
            elif user_email and model_name:
                cursor.execute("""
                    SELECT session_id, model_name, user_email, system_prompt, status, progress, stage, temperature, workflow, output_format,
                           published, public_id, published_at,
                           error_message, created_at, updated_at, completed_at
                    FROM multi_model_sessions
                    WHERE user_email = %s AND model_name = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (user_email, model_name))
            else:
                return None
                
            row = cursor.fetchone()
            if not row:
                return None
            keys = [
                "session_id",
                "model_name",
                "user_email",
                "system_prompt",
                "status",
                "progress",
                "stage",
                "temperature",
                "workflow",
                "output_format",
                "published",
                "public_id",
                "published_at",
                "error_message",
                "created_at",
                "updated_at",
                "completed_at",
            ]
            return dict(zip(keys, row))

    def get_multi_model_files(self, session_id: str):
        """Get all files for a multi-model session."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, session_id, file_name, file_type, storage_path, vector_collection_id, db_table_name, processed, error_message, created_at
                FROM multi_model_files
                WHERE session_id = %s
                ORDER BY created_at
            """, (session_id,))
            rows = cursor.fetchall()
            keys = ["id", "session_id", "file_name", "file_type", "storage_path", "vector_collection_id", "db_table_name", "processed", "error_message", "created_at"]
            return [dict(zip(keys, row)) for row in rows]

    def get_user_multi_models(self, user_email: str):
        """Get all multi-models for a user."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT session_id, model_name, system_prompt, status, progress, stage, temperature,
                       workflow, output_format,
                       published, public_id, published_at,
                       error_message,
                       created_at, updated_at, completed_at
                FROM multi_model_sessions
                WHERE user_email = %s
                ORDER BY updated_at DESC
            """, (user_email,))
            rows = cursor.fetchall()
            keys = [
                "session_id",
                "model_name",
                "system_prompt",
                "status",
                "progress",
                "stage",
                "temperature",
                "workflow",
                "output_format",
                "published",
                "public_id",
                "published_at",
                "error_message",
                "created_at",
                "updated_at",
                "completed_at",
            ]
            return [dict(zip(keys, row)) for row in rows]

    def get_multi_model_session_by_public_id(self, public_id: str):
        """Get a multi-model session by its public_id."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT session_id, model_name, user_email, system_prompt, status, progress, stage, temperature, workflow, output_format,
                       published, public_id, published_at,
                       error_message, created_at, updated_at, completed_at
                FROM multi_model_sessions
                WHERE public_id = %s
                """,
                (public_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            keys = [
                "session_id",
                "model_name",
                "user_email",
                "system_prompt",
                "status",
                "progress",
                "stage",
                "temperature",
                "workflow",
                "output_format",
                "published",
                "public_id",
                "published_at",
                "error_message",
                "created_at",
                "updated_at",
                "completed_at",
            ]
            return dict(zip(keys, row))

    def set_multi_model_published(self, session_id: str, user_email: str, published: bool, public_id: str = None):
        """
        Toggle publish state. If publishing and public_id is provided, store it.
        If publishing and public_id is NULL, keep existing public_id (if any).
        If unpublishing, keep public_id (so republish reuses same link) but clear published_at.
        """
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            if published:
                cursor.execute(
                    """
                    UPDATE multi_model_sessions
                    SET
                        published = TRUE,
                        public_id = COALESCE(%s, public_id),
                        published_at = COALESCE(published_at, NOW()),
                        updated_at = NOW()
                    WHERE session_id = %s AND user_email = %s
                    """,
                    (public_id, session_id, user_email),
                )
            else:
                cursor.execute(
                    """
                    UPDATE multi_model_sessions
                    SET
                        published = FALSE,
                        published_at = NULL,
                        updated_at = NOW()
                    WHERE session_id = %s AND user_email = %s
                    """,
                    (session_id, user_email),
                )
            return cursor.rowcount

    def update_multi_model_session_config(
        self,
        session_id: str,
        user_email: str,
        model_name: str = None,
        system_prompt: str = None,
        temperature: float = None,
        workflow: str = None,
        output_format: str = None,
    ):
        """Update a multi-model session's editable configuration fields."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE multi_model_sessions
                SET
                    model_name = COALESCE(%s, model_name),
                    system_prompt = COALESCE(%s, system_prompt),
                    temperature = COALESCE(%s, temperature),
                    workflow = COALESCE(%s, workflow),
                    output_format = COALESCE(%s, output_format),
                    updated_at = NOW()
                WHERE session_id = %s AND user_email = %s
                """,
                (model_name, system_prompt, temperature, workflow, output_format, session_id, user_email),
            )
            return cursor.rowcount

    def mark_file_processed(self, session_id: str, file_name: str):
        """Mark a file as processed."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                UPDATE multi_model_files
                SET processed = TRUE
                WHERE session_id = %s AND file_name = %s
            """, (session_id, file_name))

    def delete_multi_model_session(self, session_id: str):
        """Delete a multi-model session and all associated files."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            # Delete files first (foreign key constraint)
            cursor.execute("""
                DELETE FROM multi_model_files
                WHERE session_id = %s
            """, (session_id,))
            
            # Delete session
            cursor.execute("""
                DELETE FROM multi_model_sessions
                WHERE session_id = %s
            """, (session_id,))
            
            self.connection.commit()

    # -------- App Builder Apps --------
    def create_app_builder_app(self, user_email: str, app_name: str, prompt: str, project_name: str,
                                prd: str = None, generated_uiux: str = None, plan: list = None, architecture: dict = None):
        """Create a new app builder app record."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO app_builder_apps (user_email, app_name, prompt, project_name, prd, generated_uiux, plan, architecture, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, NOW(), NOW())
                RETURNING id, user_email, app_name, prompt, project_name, prd, generated_uiux, plan, architecture, created_at, updated_at
            """, (user_email, app_name, prompt, project_name, prd, generated_uiux, json.dumps(plan) if plan is not None else None,
                  json.dumps(architecture) if architecture is not None else None))
            row = cursor.fetchone()
            cols = [d[0] for d in cursor.description]
            return dict(zip(cols, row))

    def get_user_app_builder_apps(self, user_email: str):
        """Get all app builder apps for a user."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, user_email, app_name, prompt, project_name, prd, generated_uiux, plan, architecture, created_at, updated_at
                FROM app_builder_apps
                WHERE user_email = %s
                ORDER BY updated_at DESC
            """, (user_email,))
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
            result = []
            for row in rows:
                r = dict(zip(cols, row))
                if isinstance(r.get("plan"), str):
                    try:
                        r["plan"] = json.loads(r["plan"]) if r["plan"] else None
                    except Exception:
                        r["plan"] = None
                if isinstance(r.get("architecture"), str):
                    try:
                        r["architecture"] = json.loads(r["architecture"]) if r["architecture"] else None
                    except Exception:
                        r["architecture"] = None
                result.append(r)
            return result

    def get_app_builder_app(self, app_id: int, user_email: str = None):
        """Get a single app builder app by id (optionally scoped by user_email)."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            if user_email:
                cursor.execute("""
                    SELECT id, user_email, app_name, prompt, project_name, prd, generated_uiux, plan, architecture, created_at, updated_at
                    FROM app_builder_apps
                    WHERE id = %s AND user_email = %s
                """, (app_id, user_email))
            else:
                cursor.execute("""
                    SELECT id, user_email, app_name, prompt, project_name, prd, generated_uiux, plan, architecture, created_at, updated_at
                    FROM app_builder_apps
                    WHERE id = %s
                """, (app_id,))
            row = cursor.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cursor.description]
            r = dict(zip(cols, row))
            if isinstance(r.get("plan"), str):
                try:
                    r["plan"] = json.loads(r["plan"]) if r["plan"] else None
                except Exception:
                    r["plan"] = None
            if isinstance(r.get("architecture"), str):
                try:
                    r["architecture"] = json.loads(r["architecture"]) if r["architecture"] else None
                except Exception:
                    r["architecture"] = None
            return r

    def update_app_builder_app(self, app_id: int, user_email: str, app_name: str = None, prompt: str = None,
                                project_name: str = None, prd: str = None, generated_uiux: str = None,
                                plan: list = None, architecture: dict = None):
        """Update an app builder app."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                UPDATE app_builder_apps
                SET
                    app_name = COALESCE(%s, app_name),
                    prompt = COALESCE(%s, prompt),
                    project_name = COALESCE(%s, project_name),
                    prd = COALESCE(%s, prd),
                    generated_uiux = COALESCE(%s, generated_uiux),
                    plan = COALESCE(%s::jsonb, plan),
                    architecture = COALESCE(%s::jsonb, architecture),
                    updated_at = NOW()
                WHERE id = %s AND user_email = %s
            """, (app_name, prompt, project_name, prd, generated_uiux,
                  json.dumps(plan) if plan is not None else None,
                  json.dumps(architecture) if architecture is not None else None,
                  app_id, user_email))
            return cursor.rowcount

    def delete_app_builder_app(self, app_id: int, user_email: str):
        """Delete an app builder app."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("DELETE FROM app_builder_apps WHERE id = %s AND user_email = %s", (app_id, user_email))
            return cursor.rowcount

    # Deployment methods
    def create_deployment(self, app_id: int = None, project_name: str = None, frontend_url: str = None, 
                         backend_url: str = None, frontend_port: int = None, backend_port: int = None,
                         deployment_status: str = 'pending', error_message: str = None):
        """Create a new deployment record. app_id is optional."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO app_builder_deployments (app_id, project_name, frontend_url, backend_url, 
                                                     frontend_port, backend_port, deployment_status, 
                                                     error_message, deployed_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                RETURNING id, app_id, project_name, frontend_url, backend_url, frontend_port, backend_port, 
                         deployment_status, error_message, deployed_at, updated_at
            """, (app_id, project_name, frontend_url, backend_url, frontend_port, backend_port, 
                  deployment_status, error_message))
            row = cursor.fetchone()
            if row:
                keys = ["id", "app_id", "project_name", "frontend_url", "backend_url", "frontend_port", 
                       "backend_port", "deployment_status", "error_message", "deployed_at", "updated_at"]
                return dict(zip(keys, row))
            return None

    def update_deployment(self, deployment_id: int, frontend_url: str = None, backend_url: str = None,
                         deployment_status: str = None, error_message: str = None):
        """Update an existing deployment record."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            updates = []
            params = []
            if frontend_url is not None:
                updates.append("frontend_url = %s")
                params.append(frontend_url)
            if backend_url is not None:
                updates.append("backend_url = %s")
                params.append(backend_url)
            if deployment_status is not None:
                updates.append("deployment_status = %s")
                params.append(deployment_status)
            if error_message is not None:
                updates.append("error_message = %s")
                params.append(error_message)
            
            if updates:
                updates.append("updated_at = NOW()")
                params.append(deployment_id)
                cursor.execute(f"""
                    UPDATE app_builder_deployments
                    SET {', '.join(updates)}
                    WHERE id = %s
                """, params)
                return cursor.rowcount
            return 0

    def get_deployment_by_app_id(self, app_id: int):
        """Get the latest deployment for an app."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, app_id, project_name, frontend_url, backend_url, frontend_port, backend_port,
                       deployment_status, error_message, deployed_at, updated_at
                FROM app_builder_deployments
                WHERE app_id = %s
                ORDER BY deployed_at DESC
                LIMIT 1
            """, (app_id,))
            row = cursor.fetchone()
            if row:
                keys = ["id", "app_id", "project_name", "frontend_url", "backend_url", "frontend_port",
                       "backend_port", "deployment_status", "error_message", "deployed_at", "updated_at"]
                return dict(zip(keys, row))
            return None

    def get_deployment_by_project_name(self, project_name: str):
        """Get the latest deployment for a project."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, app_id, project_name, frontend_url, backend_url, frontend_port, backend_port,
                       deployment_status, error_message, deployed_at, updated_at
                FROM app_builder_deployments
                WHERE project_name = %s
                ORDER BY deployed_at DESC
                LIMIT 1
            """, (project_name,))
            row = cursor.fetchone()
            if row:
                keys = ["id", "app_id", "project_name", "frontend_url", "backend_url", "frontend_port",
                       "backend_port", "deployment_status", "error_message", "deployed_at", "updated_at"]
                return dict(zip(keys, row))
            return None

    def create_or_update_codegen_session(self, session_id: str, project_name: str, requirement: str = None,
                                          prd: str = None, plan: list = None, architecture: dict = None,
                                          generated_code_json: dict = None):
        """Create or update a codegen session with multi-agent data and generated code JSON (stored like documents)."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO app_builder_codegen_sessions
                (session_id, project_name, requirement, prd, plan, architecture, generated_code_json, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, NOW(), NOW())
                ON CONFLICT (session_id) DO UPDATE SET
                    project_name = COALESCE(EXCLUDED.project_name, app_builder_codegen_sessions.project_name),
                    requirement = COALESCE(EXCLUDED.requirement, app_builder_codegen_sessions.requirement),
                    prd = COALESCE(EXCLUDED.prd, app_builder_codegen_sessions.prd),
                    plan = COALESCE(EXCLUDED.plan, app_builder_codegen_sessions.plan),
                    architecture = COALESCE(EXCLUDED.architecture, app_builder_codegen_sessions.architecture),
                    generated_code_json = COALESCE(EXCLUDED.generated_code_json, app_builder_codegen_sessions.generated_code_json),
                    updated_at = NOW()
            """, (session_id, project_name, requirement, prd,
                  json.dumps(plan) if plan is not None else None,
                  json.dumps(architecture) if architecture is not None else None,
                  json.dumps(generated_code_json) if generated_code_json is not None else None))
            return cursor.rowcount

    def get_codegen_session(self, session_id: str):
        """Get a codegen session by session_id."""
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, session_id, project_name, requirement, prd, plan, architecture, generated_code_json, created_at, updated_at
                FROM app_builder_codegen_sessions
                WHERE session_id = %s
            """, (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cursor.description]
            r = dict(zip(cols, row))
            for key in ("plan", "architecture", "generated_code_json"):
                if isinstance(r.get(key), str):
                    try:
                        r[key] = json.loads(r[key]) if r[key] else None
                    except Exception:
                        r[key] = None
            return r

    def delete_table(self, table_name: str):
        """Delete a specific table from the database."""
        with self.connection.cursor() as cursor:
            cursor.execute(f"""
                DROP TABLE IF EXISTS {table_name} CASCADE
            """)
            self.connection.commit()

    # -------- LLM Settings Methods --------
    def save_llm_settings(self, email: str, provider: str = None, api_key: str = None, model_name: str = None):
        """Save or update LLM settings for a user."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO llm_settings (email, provider, api_key, model_name, created_at, updated_at)
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (email) DO UPDATE SET
                    provider = COALESCE(EXCLUDED.provider, llm_settings.provider),
                    api_key = COALESCE(EXCLUDED.api_key, llm_settings.api_key),
                    model_name = COALESCE(EXCLUDED.model_name, llm_settings.model_name),
                    updated_at = NOW()
            """, (email, provider, api_key, model_name))

    def get_llm_settings(self, email: str):
        """Get LLM settings for a user."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT email, provider, api_key, model_name, created_at, updated_at
                FROM llm_settings
                WHERE email = %s
            """, (email,))
            row = cursor.fetchone()
            if not row:
                return None
            keys = ["email", "provider", "api_key", "model_name", "created_at", "updated_at"]
            return dict(zip(keys, row))

    def delete_llm_settings(self, email: str):
        """Delete LLM settings for a user."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("DELETE FROM llm_settings WHERE email = %s", (email,))
            return f"{cursor.rowcount} LLM settings deleted"


# Sample config - replace with your actual database credentials
PGHOST = 'ep-yellow-recipe-a5fny139.us-east-2.aws.neon.tech'
PGDATABASE = 'test'
PGUSER = 'test_owner'
PGPASSWORD = 'tcWI7unQ6REA'

if __name__ == '__main__':
    pdd = PostgresDatabase()
    pdd.create_connection(PGUSER, PGPASSWORD, PGDATABASE, PGHOST)
    #pdd.delete_all_tables()      # Drops old tables (with all constraints)
    pdd.create_table()           # Creates akio_data_fastapi (no unique on email)
    pdd.create_reports_table()   # Creates reports_fastapi (no unique/FK on email)
    print("Tables recreated with no unique or foreign key constraints on email.")
