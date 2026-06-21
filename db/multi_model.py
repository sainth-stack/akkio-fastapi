from __future__ import annotations

import json
from typing import Any


_MULTI_MODEL_DDL = """
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
);

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
);
"""


class MultiModelMixin:
    def ensure_training_tables(self):
        self.ensure_connection()
        if self._tables_initialized:
            return
        with self.connection.cursor() as cursor:
            cursor.execute(_MULTI_MODEL_DDL)
            for stmt in (
                "ALTER TABLE multi_model_sessions ADD COLUMN IF NOT EXISTS temperature FLOAT DEFAULT 0.0",
                "ALTER TABLE multi_model_sessions ADD COLUMN IF NOT EXISTS workflow TEXT",
                "ALTER TABLE multi_model_sessions ADD COLUMN IF NOT EXISTS output_format TEXT",
                "ALTER TABLE multi_model_sessions ADD COLUMN IF NOT EXISTS published BOOLEAN DEFAULT FALSE",
                "ALTER TABLE multi_model_sessions ADD COLUMN IF NOT EXISTS public_id VARCHAR(255)",
                "ALTER TABLE multi_model_sessions ADD COLUMN IF NOT EXISTS published_at TIMESTAMP",
                "ALTER TABLE multi_model_files ADD COLUMN IF NOT EXISTS error_message TEXT",
            ):
                try:
                    cursor.execute(stmt)
                except Exception:
                    pass
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
        self._tables_initialized = True

    def create_multi_model_session(
        self,
        session_id: str,
        model_name: str,
        user_email: str,
        system_prompt: str,
        temperature: float = 0.0,
        workflow: str | None = None,
        output_format: str | None = None,
    ):
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO multi_model_sessions (
                    session_id, model_name, user_email, system_prompt, temperature,
                    workflow, output_format, status, progress, stage, created_at, updated_at
                )
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
                """,
                (session_id, model_name, user_email, system_prompt, temperature, workflow, output_format),
            )

    def add_multi_model_file(
        self,
        session_id: str,
        file_name: str,
        file_type: str,
        storage_path: str | None = None,
        vector_collection_id: str | None = None,
        db_table_name: str | None = None,
        error_message: str | None = None,
    ):
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO multi_model_files (
                    session_id, file_name, file_type, storage_path,
                    vector_collection_id, db_table_name, error_message, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                """,
                (session_id, file_name, file_type, storage_path, vector_collection_id, db_table_name, error_message),
            )

    def update_multi_model_progress(self, session_id: str, progress: int, stage: str, status: str | None = None):
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            if status:
                cursor.execute(
                    """
                    UPDATE multi_model_sessions
                    SET progress = %s, stage = %s, status = %s, updated_at = NOW()
                    WHERE session_id = %s
                    """,
                    (progress, stage, status, session_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE multi_model_sessions
                    SET progress = %s, stage = %s, updated_at = NOW()
                    WHERE session_id = %s
                    """,
                    (progress, stage, session_id),
                )

    def complete_multi_model_training(self, session_id: str):
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE multi_model_sessions
                SET status = 'completed', progress = 100, stage = 'Training completed',
                    updated_at = NOW(), completed_at = NOW()
                WHERE session_id = %s
                """,
                (session_id,),
            )

    def fail_multi_model_training(self, session_id: str, error_message: str):
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE multi_model_sessions
                SET status = 'failed', error_message = %s, updated_at = NOW()
                WHERE session_id = %s
                """,
                (error_message, session_id),
            )

    def _session_row_to_dict(self, row) -> dict:
        keys = [
            "session_id", "model_name", "user_email", "system_prompt", "status", "progress",
            "stage", "temperature", "workflow", "output_format", "published", "public_id",
            "published_at", "error_message", "created_at", "updated_at", "completed_at",
        ]
        return dict(zip(keys, row))

    def get_multi_model_session(
        self,
        session_id: str | None = None,
        user_email: str | None = None,
        model_name: str | None = None,
    ):
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            if session_id:
                cursor.execute(
                    """
                    SELECT session_id, model_name, user_email, system_prompt, status, progress, stage,
                           temperature, workflow, output_format, published, public_id, published_at,
                           error_message, created_at, updated_at, completed_at
                    FROM multi_model_sessions WHERE session_id = %s
                    """,
                    (session_id,),
                )
            elif user_email and model_name:
                cursor.execute(
                    """
                    SELECT session_id, model_name, user_email, system_prompt, status, progress, stage,
                           temperature, workflow, output_format, published, public_id, published_at,
                           error_message, created_at, updated_at, completed_at
                    FROM multi_model_sessions
                    WHERE user_email = %s AND model_name = %s
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (user_email, model_name),
                )
            else:
                return None
            row = cursor.fetchone()
            return self._session_row_to_dict(row) if row else None

    def get_multi_model_files(self, session_id: str):
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, session_id, file_name, file_type, storage_path, vector_collection_id,
                       db_table_name, processed, error_message, created_at
                FROM multi_model_files WHERE session_id = %s ORDER BY created_at
                """,
                (session_id,),
            )
            keys = [
                "id", "session_id", "file_name", "file_type", "storage_path",
                "vector_collection_id", "db_table_name", "processed", "error_message", "created_at",
            ]
            return [dict(zip(keys, row)) for row in cursor.fetchall()]

    def get_user_multi_models(self, user_email: str):
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT session_id, model_name, system_prompt, status, progress, stage, temperature,
                       workflow, output_format, published, public_id, published_at, error_message,
                       created_at, updated_at, completed_at
                FROM multi_model_sessions WHERE user_email = %s ORDER BY updated_at DESC
                """,
                (user_email,),
            )
            keys = [
                "session_id", "model_name", "system_prompt", "status", "progress", "stage",
                "temperature", "workflow", "output_format", "published", "public_id",
                "published_at", "error_message", "created_at", "updated_at", "completed_at",
            ]
            return [dict(zip(keys, row)) for row in cursor.fetchall()]

    def get_multi_model_session_by_public_id(self, public_id: str):
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT session_id, model_name, user_email, system_prompt, status, progress, stage,
                       temperature, workflow, output_format, published, public_id, published_at,
                       error_message, created_at, updated_at, completed_at
                FROM multi_model_sessions WHERE public_id = %s
                """,
                (public_id,),
            )
            row = cursor.fetchone()
            return self._session_row_to_dict(row) if row else None

    def set_multi_model_published(
        self, session_id: str, user_email: str, published: bool, public_id: str | None = None
    ):
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            if published:
                cursor.execute(
                    """
                    UPDATE multi_model_sessions
                    SET published = TRUE,
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
                    SET published = FALSE, published_at = NULL, updated_at = NOW()
                    WHERE session_id = %s AND user_email = %s
                    """,
                    (session_id, user_email),
                )
            return cursor.rowcount

    def update_multi_model_session_config(
        self,
        session_id: str,
        user_email: str,
        model_name: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        workflow: str | None = None,
        output_format: str | None = None,
    ):
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE multi_model_sessions
                SET model_name = COALESCE(%s, model_name),
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
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE multi_model_files SET processed = TRUE
                WHERE session_id = %s AND file_name = %s
                """,
                (session_id, file_name),
            )

    def delete_multi_model_session(self, session_id: str):
        self.ensure_training_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("DELETE FROM multi_model_files WHERE session_id = %s", (session_id,))
            cursor.execute("DELETE FROM multi_model_sessions WHERE session_id = %s", (session_id,))

    def delete_table(self, table_name: str):
        with self.connection.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
