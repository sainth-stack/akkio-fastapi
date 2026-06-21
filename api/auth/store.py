from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from psycopg2.extras import RealDictCursor

from db import PostgresDatabase

AUTH_SCHEMA = "auth"

_AUTH_SCHEMA_SQL = f"""
CREATE SCHEMA IF NOT EXISTS {AUTH_SCHEMA};

CREATE TABLE IF NOT EXISTS {AUTH_SCHEMA}.organizations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {AUTH_SCHEMA}.roles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    permissions TEXT[] NOT NULL DEFAULT '{{}}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {AUTH_SCHEMA}.users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255),
    username VARCHAR(255),
    app VARCHAR(50) NOT NULL DEFAULT 'akkio',
    organization_id INTEGER REFERENCES {AUTH_SCHEMA}.organizations(id) ON DELETE SET NULL,
    google_id VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (email, app)
);

CREATE TABLE IF NOT EXISTS {AUTH_SCHEMA}.user_roles (
    user_id INTEGER NOT NULL REFERENCES {AUTH_SCHEMA}.users(id) ON DELETE CASCADE,
    role_id INTEGER NOT NULL REFERENCES {AUTH_SCHEMA}.roles(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, role_id)
);

CREATE TABLE IF NOT EXISTS {AUTH_SCHEMA}.refresh_tokens (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES {AUTH_SCHEMA}.users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {AUTH_SCHEMA}.revoked_tokens (
    jti VARCHAR(64) PRIMARY KEY,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_auth_users_email_app ON {AUTH_SCHEMA}.users (email, app);
CREATE INDEX IF NOT EXISTS idx_auth_user_roles_user_id ON {AUTH_SCHEMA}.user_roles (user_id);
CREATE INDEX IF NOT EXISTS idx_auth_revoked_tokens_expires_at ON {AUTH_SCHEMA}.revoked_tokens (expires_at);
"""


def _serialize_org(row: dict | None) -> dict | None:
    if not row:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row.get("description"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _serialize_role(row: dict) -> dict:
    perms = row.get("permissions") or []
    if isinstance(perms, str):
        perms = [perms]
    return {
        "id": row["id"],
        "name": row["name"],
        "permissions": list(perms),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


class AuthStore:
    def init_schema(self) -> None:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(_AUTH_SCHEMA_SQL)

    def revoke_token(self, jti: str, expires_at: datetime) -> None:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO auth.revoked_tokens (jti, expires_at)
                    VALUES (%s, %s)
                    ON CONFLICT (jti) DO NOTHING
                    """,
                    (jti, expires_at),
                )

    def is_token_revoked(self, jti: str) -> bool:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1 FROM auth.revoked_tokens
                    WHERE jti = %s AND expires_at > NOW()
                    """,
                    (jti,),
                )
                return cursor.fetchone() is not None

    def get_user_permissions(self, user_id: int) -> set[str]:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT r.permissions
                    FROM auth.user_roles ur
                    JOIN auth.roles r ON r.id = ur.role_id
                    WHERE ur.user_id = %s
                    """,
                    (user_id,),
                )
                perms: set[str] = set()
                for (role_perms,) in cursor.fetchall():
                    if role_perms:
                        perms.update(role_perms)
                return perms

    def _fetch_roles_for_user(self, cursor, user_id: int) -> list[dict]:
        cursor.execute(
            """
            SELECT r.id, r.name, r.permissions, r.created_at, r.updated_at
            FROM auth.user_roles ur
            JOIN auth.roles r ON r.id = ur.role_id
            WHERE ur.user_id = %s
            ORDER BY r.id
            """,
            (user_id,),
        )
        return [_serialize_role(dict(row)) for row in cursor.fetchall()]

    def _fetch_user_row(self, cursor, user_id: int) -> dict | None:
        cursor.execute(
            """
            SELECT u.id, u.name, u.email, u.username, u.app, u.organization_id,
                   u.created_at, u.updated_at,
                   o.id AS org_id, o.name AS org_name, o.description AS org_description,
                   o.created_at AS org_created_at, o.updated_at AS org_updated_at
            FROM auth.users u
            LEFT JOIN auth.organizations o ON o.id = u.organization_id
            WHERE u.id = %s
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        data = dict(row)
        org = None
        if data.get("org_id") is not None:
            org = {
                "id": data["org_id"],
                "name": data["org_name"],
                "description": data.get("org_description"),
                "created_at": data["org_created_at"],
                "updated_at": data["org_updated_at"],
            }
        roles = self._fetch_roles_for_user(cursor, user_id)
        return {
            "id": data["id"],
            "name": data.get("name"),
            "email": data["email"],
            "username": data.get("username"),
            "app": data["app"],
            "organization": _serialize_org(org),
            "roles": roles,
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
        }

    def get_user_by_id(self, user_id: int) -> dict | None:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                return self._fetch_user_row(cursor, user_id)

    def get_user_by_email(self, email: str, app: str = "akkio") -> dict | None:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT id FROM auth.users WHERE email = %s AND app = %s",
                    (email.lower(), app),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return self._fetch_user_row(cursor, row["id"])

    def get_user_credentials(self, email: str, app: str = "akkio") -> dict | None:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT id, email, password_hash, app
                    FROM auth.users
                    WHERE email = %s AND app = %s
                    """,
                    (email.lower(), app),
                )
                row = cursor.fetchone()
                return dict(row) if row else None

    def set_user_roles(self, user_id: int, role_ids: list[int]) -> None:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM auth.user_roles WHERE user_id = %s", (user_id,))
                for role_id in role_ids:
                    cursor.execute(
                        """
                        INSERT INTO auth.user_roles (user_id, role_id)
                        VALUES (%s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (user_id, role_id),
                    )
                cursor.execute(
                    "UPDATE auth.users SET updated_at = NOW() WHERE id = %s",
                    (user_id,),
                )

    def create_user(
        self,
        *,
        name: str,
        email: str,
        password_hash: str | None,
        username: str | None,
        app: str,
        organization_id: int | None,
        role_ids: list[int],
        google_id: str | None = None,
    ) -> dict:
        db = PostgresDatabase()
        email = email.lower()
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                try:
                    cursor.execute(
                        """
                        INSERT INTO auth.users (name, email, password_hash, username, app, organization_id, google_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (name, email, password_hash, username, app, organization_id, google_id),
                    )
                except Exception as exc:
                    if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
                        raise HTTPException(status_code=400, detail="Email already registered for this app") from exc
                    raise
                user_id = cursor.fetchone()["id"]
                for role_id in role_ids:
                    cursor.execute(
                        "INSERT INTO auth.user_roles (user_id, role_id) VALUES (%s, %s)",
                        (user_id, role_id),
                    )
        user = self.get_user_by_id(user_id)
        assert user is not None
        return user

    def find_or_create_google_user(
        self,
        *,
        email: str,
        name: str,
        google_id: str,
        app: str = "akkio",
        default_organization_id: int | None = None,
        default_role_ids: list[int] | None = None,
    ) -> dict:
        db = PostgresDatabase()
        email = email.lower()
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT id FROM auth.users WHERE email = %s AND app = %s",
                    (email, app),
                )
                row = cursor.fetchone()
                if row:
                    cursor.execute(
                        """
                        UPDATE auth.users
                        SET google_id = COALESCE(google_id, %s),
                            name = COALESCE(NULLIF(name, ''), %s),
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (google_id, name, row["id"]),
                    )
                    return self._fetch_user_row(cursor, row["id"])

        username = name.replace(" ", "_") if name else email.split("@")[0]
        return self.create_user(
            name=name,
            email=email,
            password_hash=None,
            username=username,
            app=app,
            organization_id=default_organization_id,
            role_ids=default_role_ids or [],
            google_id=google_id,
        )

    # --- Organizations ---

    def list_organizations(self) -> list[dict]:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT id, name, description, created_at, updated_at
                    FROM auth.organizations
                    ORDER BY id
                    """
                )
                return [dict(row) for row in cursor.fetchall()]

    def get_organization(self, org_id: int) -> dict | None:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT id, name, description, created_at, updated_at
                    FROM auth.organizations WHERE id = %s
                    """,
                    (org_id,),
                )
                row = cursor.fetchone()
                return dict(row) if row else None

    def create_organization(self, name: str, description: str | None) -> dict:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    INSERT INTO auth.organizations (name, description)
                    VALUES (%s, %s)
                    RETURNING id, name, description, created_at, updated_at
                    """,
                    (name, description),
                )
                return dict(cursor.fetchone())

    def update_organization(self, org_id: int, updates: dict[str, Any]) -> dict | None:
        fields = []
        values: list[Any] = []
        for key in ("name", "description"):
            if key in updates and updates[key] is not None:
                fields.append(f"{key} = %s")
                values.append(updates[key])
        if not fields:
            return self.get_organization(org_id)
        fields.append("updated_at = NOW()")
        values.append(org_id)
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    f"""
                    UPDATE auth.organizations SET {", ".join(fields)}
                    WHERE id = %s
                    RETURNING id, name, description, created_at, updated_at
                    """,
                    tuple(values),
                )
                row = cursor.fetchone()
                return dict(row) if row else None

    def delete_organization(self, org_id: int) -> bool:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM auth.organizations WHERE id = %s", (org_id,))
                return cursor.rowcount > 0

    # --- Roles ---

    def list_roles(self) -> list[dict]:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT id, name, permissions, created_at, updated_at
                    FROM auth.roles ORDER BY id
                    """
                )
                return [_serialize_role(dict(row)) for row in cursor.fetchall()]

    def get_role(self, role_id: int) -> dict | None:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT id, name, permissions, created_at, updated_at
                    FROM auth.roles WHERE id = %s
                    """,
                    (role_id,),
                )
                row = cursor.fetchone()
                return _serialize_role(dict(row)) if row else None

    def create_role(self, name: str, permissions: list[str]) -> dict:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                try:
                    cursor.execute(
                        """
                        INSERT INTO auth.roles (name, permissions)
                        VALUES (%s, %s)
                        RETURNING id, name, permissions, created_at, updated_at
                        """,
                        (name, permissions),
                    )
                except Exception as exc:
                    if "unique" in str(exc).lower():
                        raise HTTPException(status_code=400, detail="Role name already exists") from exc
                    raise
                return _serialize_role(dict(cursor.fetchone()))

    def update_role(self, role_id: int, updates: dict[str, Any]) -> dict | None:
        fields = []
        values: list[Any] = []
        if "name" in updates and updates["name"] is not None:
            fields.append("name = %s")
            values.append(updates["name"])
        if "permissions" in updates and updates["permissions"] is not None:
            fields.append("permissions = %s")
            values.append(updates["permissions"])
        if not fields:
            return self.get_role(role_id)
        fields.append("updated_at = NOW()")
        values.append(role_id)
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    f"""
                    UPDATE auth.roles SET {", ".join(fields)}
                    WHERE id = %s
                    RETURNING id, name, permissions, created_at, updated_at
                    """,
                    tuple(values),
                )
                row = cursor.fetchone()
                return _serialize_role(dict(row)) if row else None

    def delete_role(self, role_id: int) -> bool:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM auth.roles WHERE id = %s", (role_id,))
                return cursor.rowcount > 0

    # --- Users (admin) ---

    def list_users(self) -> list[dict]:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT id FROM auth.users ORDER BY id")
                user_ids = [row["id"] for row in cursor.fetchall()]
                return [self._fetch_user_row(cursor, uid) for uid in user_ids]

    def update_user(self, user_id: int, updates: dict[str, Any]) -> dict | None:
        fields = []
        values: list[Any] = []
        column_map = {
            "name": "name",
            "email": "email",
            "username": "username",
            "app": "app",
            "password_hash": "password_hash",
            "organization_id": "organization_id",
        }
        for key, column in column_map.items():
            if key in updates and updates[key] is not None:
                value = updates[key]
                if key == "email":
                    value = str(value).lower()
                fields.append(f"{column} = %s")
                values.append(value)
        role_ids = updates.get("role_ids")
        if not fields and role_ids is None:
            return self.get_user_by_id(user_id)
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if fields:
                    fields.append("updated_at = NOW()")
                    values.append(user_id)
                    cursor.execute(
                        f"""
                        UPDATE auth.users SET {", ".join(fields)}
                        WHERE id = %s
                        RETURNING id
                        """,
                        tuple(values),
                    )
                    if cursor.fetchone() is None:
                        return None
                if role_ids is not None:
                    cursor.execute("DELETE FROM auth.user_roles WHERE user_id = %s", (user_id,))
                    for role_id in role_ids:
                        cursor.execute(
                            "INSERT INTO auth.user_roles (user_id, role_id) VALUES (%s, %s)",
                            (user_id, role_id),
                        )
                    cursor.execute(
                        "UPDATE auth.users SET updated_at = NOW() WHERE id = %s",
                        (user_id,),
                    )
        return self.get_user_by_id(user_id)

    def delete_user(self, user_id: int) -> bool:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM auth.users WHERE id = %s", (user_id,))
                return cursor.rowcount > 0

    def get_default_org_and_admin_role(self) -> tuple[int | None, int | None]:
        db = PostgresDatabase()
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id FROM auth.organizations ORDER BY id LIMIT 1")
                org_row = cursor.fetchone()
                cursor.execute("SELECT id FROM auth.roles WHERE name = 'Admin' LIMIT 1")
                role_row = cursor.fetchone()
                return (
                    org_row[0] if org_row else None,
                    role_row[0] if role_row else None,
                )


auth_store = AuthStore()
