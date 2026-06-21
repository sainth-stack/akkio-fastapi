"""
Postgres-backed App Builder storage (apps, codegen sessions, deployments).
Replaces legacy MongoDB app_builder_db — Postgres-only app store.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from psycopg2.extras import RealDictCursor, Json

from db.postgres import PostgresPool

_BUILDER_DDL = """
CREATE TABLE IF NOT EXISTS builder_apps (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    name VARCHAR(255) NOT NULL,
    project_name VARCHAR(255) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS app_builder_codegen_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    project_name VARCHAR(255) NOT NULL,
    requirement TEXT,
    prd TEXT,
    plan JSONB,
    architecture JSONB,
    api_contract TEXT,
    db_schema TEXT,
    generated_code_json JSONB,
    generated_files JSONB,
    app_id INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS app_builder_deployments (
    id SERIAL PRIMARY KEY,
    app_id INTEGER REFERENCES builder_apps(id) ON DELETE SET NULL,
    project_name VARCHAR(255) NOT NULL,
    frontend_url VARCHAR(512),
    backend_url VARCHAR(512),
    frontend_port INTEGER,
    backend_port INTEGER,
    deployment_status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    deployed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_builder_apps_user_id ON builder_apps (user_id);
CREATE INDEX IF NOT EXISTS idx_builder_apps_project_name ON builder_apps (project_name);
CREATE INDEX IF NOT EXISTS idx_codegen_sessions_session_id ON app_builder_codegen_sessions (session_id);
CREATE INDEX IF NOT EXISTS idx_deployments_app_id ON app_builder_deployments (app_id);
CREATE INDEX IF NOT EXISTS idx_deployments_project_name ON app_builder_deployments (project_name);
"""

_METADATA_FIELDS = (
    "prompt",
    "prd",
    "generated_uiux",
    "plan",
    "architecture",
    "api_contract",
    "db_schema",
    "agents_state",
    "generated_code_json",
    "generated_files",
    "user_email",
)


def _serialize_ts(value) -> Any:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _app_row_to_dict(row: dict, user_email: str | None = None) -> dict:
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    email = user_email or meta.get("user_email") or ""
    out = {
        "id": str(row["id"]),
        "user_email": email,
        "app_name": row["name"],
        "name": row["name"],
        "project_name": row["project_name"],
        "created_at": _serialize_ts(row.get("created_at")),
        "updated_at": _serialize_ts(row.get("updated_at")),
    }
    for key in _METADATA_FIELDS:
        if key in meta and key != "user_email":
            out[key] = meta[key]
    return out


class AppBuilderStore(PostgresPool):
    _schema_ready = False

    def init_schema(self) -> None:
        if self.__class__._schema_ready:
            return
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(_BUILDER_DDL)
        self.__class__._schema_ready = True

    def _resolve_user_id(self, user_email: str) -> int:
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id FROM auth.users WHERE email = %s AND app = 'akkio' LIMIT 1",
                    (user_email.lower(),),
                )
                row = cursor.fetchone()
                if row:
                    return row[0]
        raise ValueError(f"No auth user found for email: {user_email}")

    def _user_email_for_id(self, user_id: int) -> str | None:
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT email FROM auth.users WHERE id = %s", (user_id,))
                row = cursor.fetchone()
                return row[0] if row else None

    def create_app_builder_app(
        self,
        user_email: str,
        app_name: str,
        prompt: str,
        project_name: str,
        prd: str | None = None,
        generated_uiux: str | None = None,
        plan: list | None = None,
        architecture: dict | None = None,
        api_contract: str | None = None,
        db_schema: str | None = None,
        agents_state: dict | None = None,
        generated_code_json: dict | None = None,
        generated_files: dict | None = None,
        user_id: int | None = None,
    ) -> dict:
        self.init_schema()
        uid = user_id if user_id is not None else self._resolve_user_id(user_email)
        metadata = {
            "prompt": prompt,
            "user_email": user_email.lower(),
            "prd": prd,
            "generated_uiux": generated_uiux,
            "plan": plan,
            "architecture": architecture,
            "api_contract": api_contract,
            "db_schema": db_schema,
            "agents_state": agents_state,
            "generated_code_json": generated_code_json,
            "generated_files": generated_files,
        }
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    INSERT INTO builder_apps (user_id, name, project_name, metadata)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, user_id, name, project_name, metadata, created_at, updated_at
                    """,
                    (uid, app_name, project_name, Json(metadata)),
                )
                row = dict(cursor.fetchone())
        return _app_row_to_dict(row, user_email)

    def get_user_app_builder_apps(self, user_email: str, user_id: int | None = None) -> list:
        self.init_schema()
        uid = user_id if user_id is not None else self._resolve_user_id(user_email)
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT id, user_id, name, project_name, metadata, created_at, updated_at
                    FROM builder_apps WHERE user_id = %s ORDER BY updated_at DESC
                    """,
                    (uid,),
                )
                return [_app_row_to_dict(dict(r), user_email) for r in cursor.fetchall()]

    def get_app_by_project_name(self, project_name: str) -> dict | None:
        self.init_schema()
        if not project_name or not project_name.strip():
            return None
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT id, user_id, name, project_name, metadata, created_at, updated_at
                    FROM builder_apps WHERE project_name = %s
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (project_name.strip(),),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                row = dict(row)
                email = self._user_email_for_id(row["user_id"])
                return _app_row_to_dict(row, email)

    def get_app_builder_app(
        self, app_id, user_email: str | None = None, user_id: int | None = None
    ) -> dict | None:
        self.init_schema()
        try:
            aid = int(app_id)
        except (TypeError, ValueError):
            return None
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if user_id is not None:
                    cursor.execute(
                        """
                        SELECT id, user_id, name, project_name, metadata, created_at, updated_at
                        FROM builder_apps WHERE id = %s AND user_id = %s
                        """,
                        (aid, user_id),
                    )
                elif user_email:
                    uid = self._resolve_user_id(user_email)
                    cursor.execute(
                        """
                        SELECT id, user_id, name, project_name, metadata, created_at, updated_at
                        FROM builder_apps WHERE id = %s AND user_id = %s
                        """,
                        (aid, uid),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT id, user_id, name, project_name, metadata, created_at, updated_at
                        FROM builder_apps WHERE id = %s
                        """,
                        (aid,),
                    )
                row = cursor.fetchone()
                if not row:
                    return None
                row = dict(row)
                email = user_email or self._user_email_for_id(row["user_id"])
                return _app_row_to_dict(row, email)

    def update_app_builder_app(
        self,
        app_id,
        user_email: str,
        app_name: str | None = None,
        prompt: str | None = None,
        project_name: str | None = None,
        prd: str | None = None,
        generated_uiux: str | None = None,
        plan: list | None = None,
        architecture: dict | None = None,
        api_contract: str | None = None,
        db_schema: str | None = None,
        agents_state: dict | None = None,
        generated_code_json: dict | None = None,
        generated_files: dict | None = None,
        user_id: int | None = None,
    ) -> int:
        self.init_schema()
        existing = self.get_app_builder_app(app_id, user_email=user_email, user_id=user_id)
        if not existing:
            return 0
        meta = {k: existing.get(k) for k in _METADATA_FIELDS if k in existing}
        meta["user_email"] = user_email.lower()
        updates = {
            "app_name": app_name,
            "prompt": prompt,
            "project_name": project_name,
            "prd": prd,
            "generated_uiux": generated_uiux,
            "plan": plan,
            "architecture": architecture,
            "api_contract": api_contract,
            "db_schema": db_schema,
            "agents_state": agents_state,
            "generated_code_json": generated_code_json,
            "generated_files": generated_files,
        }
        key_map = {"app_name": "name"}
        name_val = app_name
        project_val = project_name
        for k, v in updates.items():
            if v is not None:
                field = key_map.get(k, k)
                if field == "name":
                    name_val = v
                elif field == "project_name":
                    project_val = v
                else:
                    meta[field] = v
        uid = user_id if user_id is not None else self._resolve_user_id(user_email)
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE builder_apps
                    SET name = COALESCE(%s, name),
                        project_name = COALESCE(%s, project_name),
                        metadata = %s,
                        updated_at = NOW()
                    WHERE id = %s AND user_id = %s
                    """,
                    (name_val, project_val, Json(meta), int(app_id), uid),
                )
                return cursor.rowcount

    def delete_app_builder_app(self, app_id, user_email: str, user_id: int | None = None) -> int:
        self.init_schema()
        uid = user_id if user_id is not None else self._resolve_user_id(user_email)
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM builder_apps WHERE id = %s AND user_id = %s",
                    (int(app_id), uid),
                )
                return cursor.rowcount

    def create_deployment(
        self,
        app_id=None,
        project_name: str | None = None,
        frontend_url: str | None = None,
        backend_url: str | None = None,
        frontend_port: int | None = None,
        backend_port: int | None = None,
        deployment_status: str = "pending",
        error_message: str | None = None,
    ) -> dict | None:
        self.init_schema()
        app_id_int = None
        if app_id is not None:
            try:
                app_id_int = int(app_id)
            except (TypeError, ValueError):
                app_id_int = None
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    INSERT INTO app_builder_deployments (
                        app_id, project_name, frontend_url, backend_url,
                        frontend_port, backend_port, deployment_status, error_message
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, app_id, project_name, frontend_url, backend_url,
                              frontend_port, backend_port, deployment_status, error_message,
                              deployed_at, updated_at
                    """,
                    (
                        app_id_int,
                        project_name,
                        frontend_url,
                        backend_url,
                        frontend_port,
                        backend_port,
                        deployment_status,
                        error_message,
                    ),
                )
                row = dict(cursor.fetchone())
        row["id"] = str(row["id"])
        if row.get("app_id") is not None:
            row["app_id"] = str(row["app_id"])
        row["deployed_at"] = _serialize_ts(row.get("deployed_at"))
        row["updated_at"] = _serialize_ts(row.get("updated_at"))
        return row

    def update_deployment(
        self,
        deployment_id,
        frontend_url: str | None = None,
        backend_url: str | None = None,
        deployment_status: str | None = None,
        error_message: str | None = None,
    ) -> int:
        self.init_schema()
        try:
            did = int(deployment_id)
        except (TypeError, ValueError):
            return 0
        fields = []
        values: list[Any] = []
        for col, val in (
            ("frontend_url", frontend_url),
            ("backend_url", backend_url),
            ("deployment_status", deployment_status),
            ("error_message", error_message),
        ):
            if val is not None:
                fields.append(f"{col} = %s")
                values.append(val)
        if not fields:
            return 0
        fields.append("updated_at = NOW()")
        values.append(did)
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"UPDATE app_builder_deployments SET {', '.join(fields)} WHERE id = %s",
                    tuple(values),
                )
                return cursor.rowcount

    def _deployment_row(self, row: dict) -> dict:
        row["id"] = str(row["id"])
        if row.get("app_id") is not None:
            row["app_id"] = str(row["app_id"])
        row["deployed_at"] = _serialize_ts(row.get("deployed_at"))
        row["updated_at"] = _serialize_ts(row.get("updated_at"))
        return row

    def get_deployment_by_app_id(self, app_id) -> dict | None:
        self.init_schema()
        try:
            aid = int(app_id)
        except (TypeError, ValueError):
            return None
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT id, app_id, project_name, frontend_url, backend_url,
                           frontend_port, backend_port, deployment_status, error_message,
                           deployed_at, updated_at
                    FROM app_builder_deployments
                    WHERE app_id = %s ORDER BY deployed_at DESC LIMIT 1
                    """,
                    (aid,),
                )
                row = cursor.fetchone()
                return self._deployment_row(dict(row)) if row else None

    def get_deployment_by_project_name(self, project_name: str) -> dict | None:
        self.init_schema()
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT id, app_id, project_name, frontend_url, backend_url,
                           frontend_port, backend_port, deployment_status, error_message,
                           deployed_at, updated_at
                    FROM app_builder_deployments
                    WHERE project_name = %s ORDER BY deployed_at DESC LIMIT 1
                    """,
                    (project_name,),
                )
                row = cursor.fetchone()
                return self._deployment_row(dict(row)) if row else None

    def get_deployment_by_id(self, deployment_id) -> dict | None:
        self.init_schema()
        try:
            did = int(deployment_id)
        except (TypeError, ValueError):
            return None
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT id, app_id, project_name, frontend_url, backend_url,
                           frontend_port, backend_port, deployment_status, error_message,
                           deployed_at, updated_at
                    FROM app_builder_deployments WHERE id = %s
                    """,
                    (did,),
                )
                row = cursor.fetchone()
                return self._deployment_row(dict(row)) if row else None

    def create_or_update_codegen_session(
        self,
        session_id: str,
        project_name: str,
        requirement: str | None = None,
        prd: str | None = None,
        plan: list | None = None,
        architecture: Any = None,
        api_contract: str | None = None,
        db_schema: str | None = None,
        generated_code_json: dict | None = None,
        generated_files: dict | None = None,
        app_id: str | None = None,
    ) -> int:
        self.init_schema()
        app_id_int = None
        if app_id:
            try:
                app_id_int = int(app_id)
            except (TypeError, ValueError):
                pass
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id FROM app_builder_codegen_sessions WHERE session_id = %s",
                    (session_id,),
                )
                if cursor.fetchone():
                    cursor.execute(
                        """
                        UPDATE app_builder_codegen_sessions SET
                            project_name = COALESCE(%s, project_name),
                            requirement = COALESCE(%s, requirement),
                            prd = COALESCE(%s, prd),
                            plan = COALESCE(%s, plan),
                            architecture = COALESCE(%s, architecture),
                            api_contract = COALESCE(%s, api_contract),
                            db_schema = COALESCE(%s, db_schema),
                            generated_code_json = COALESCE(%s, generated_code_json),
                            generated_files = COALESCE(%s, generated_files),
                            app_id = COALESCE(%s, app_id),
                            updated_at = NOW()
                        WHERE session_id = %s
                        """,
                        (
                            project_name,
                            requirement,
                            prd,
                            Json(plan) if plan is not None else None,
                            Json(architecture) if architecture is not None else None,
                            api_contract,
                            db_schema,
                            Json(generated_code_json) if generated_code_json is not None else None,
                            Json(generated_files) if generated_files is not None else None,
                            app_id_int,
                            session_id,
                        ),
                    )
                    return cursor.rowcount
                cursor.execute(
                    """
                    INSERT INTO app_builder_codegen_sessions (
                        session_id, project_name, requirement, prd, plan, architecture,
                        api_contract, db_schema, generated_code_json, generated_files, app_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        session_id,
                        project_name,
                        requirement,
                        prd,
                        Json(plan) if plan is not None else None,
                        Json(architecture) if architecture is not None else None,
                        api_contract,
                        db_schema,
                        Json(generated_code_json) if generated_code_json is not None else None,
                        Json(generated_files) if generated_files is not None else None,
                        app_id_int,
                    ),
                )
                return 1

    def get_codegen_session(self, session_id: str) -> dict | None:
        self.init_schema()
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT id, session_id, project_name, requirement, prd, plan, architecture,
                           api_contract, db_schema, generated_code_json, generated_files,
                           app_id, created_at, updated_at
                    FROM app_builder_codegen_sessions WHERE session_id = %s
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                out = dict(row)
                out["id"] = str(out["id"])
                if out.get("app_id") is not None:
                    out["app_id"] = str(out["app_id"])
                for k in ("created_at", "updated_at"):
                    out[k] = _serialize_ts(out.get(k))
                return out


_store: Optional[AppBuilderStore] = None


def get_app_builder_db() -> AppBuilderStore:
    global _store
    if _store is None:
        _store = AppBuilderStore()
    return _store
