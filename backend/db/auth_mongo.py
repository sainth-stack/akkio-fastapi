"""MongoDB-backed auth store (organizations, roles, users, tokens)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import HTTPException

from db.mongodb import get_collection, next_seq, utc_now


def _serialize_org(doc: Optional[dict]) -> Optional[dict]:
    if not doc:
        return None
    return {
        "id": doc["id"],
        "name": doc["name"],
        "description": doc.get("description"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def _serialize_role(doc: dict) -> dict:
    perms = doc.get("permissions") or []
    return {
        "id": doc["id"],
        "name": doc["name"],
        "permissions": list(perms),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


class AuthStore:
    def init_schema(self) -> None:
        from db.mongodb import MongoPool

        MongoPool().ensure_indexes()
        get_collection("auth_organizations").create_index("id")
        get_collection("auth_roles").create_index("id")
        get_collection("auth_users").create_index("id")

    def revoke_token(self, jti: str, expires_at: datetime) -> None:
        get_collection("auth_revoked_tokens").update_one(
            {"jti": jti},
            {"$set": {"jti": jti, "expires_at": expires_at, "revoked_at": utc_now()}},
            upsert=True,
        )

    def is_token_revoked(self, jti: str) -> bool:
        try:
            row = get_collection("auth_revoked_tokens").find_one(
                {"jti": jti, "expires_at": {"$gt": utc_now()}}
            )
            return row is not None
        except Exception:
            return False

    def get_user_permissions(self, user_id: int) -> set:
        try:
            user = get_collection("auth_users").find_one({"id": user_id}, {"role_ids": 1})
            if not user:
                return set()
            role_ids = user.get("role_ids") or []
            perms: set = set()
            for role in get_collection("auth_roles").find({"id": {"$in": role_ids}}):
                perms.update(role.get("permissions") or [])
            return perms
        except Exception:
            return set()

    def _build_user(self, user: dict) -> dict:
        org = None
        if user.get("organization_id"):
            org_doc = get_collection("auth_organizations").find_one({"id": user["organization_id"]})
            org = _serialize_org(org_doc)
        roles = []
        for rid in user.get("role_ids") or []:
            r = get_collection("auth_roles").find_one({"id": rid})
            if r:
                roles.append(_serialize_role(r))
        return {
            "id": user["id"],
            "name": user.get("name"),
            "email": user["email"],
            "username": user.get("username"),
            "app": user.get("app", "akkio"),
            "organization": org,
            "roles": roles,
            "created_at": user.get("created_at"),
            "updated_at": user.get("updated_at"),
        }

    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        try:
            user = get_collection("auth_users").find_one({"id": user_id})
            return self._build_user(user) if user else None
        except Exception:
            return None

    def get_user_by_email(self, email: str, app: str = "akkio") -> Optional[dict]:
        user = get_collection("auth_users").find_one({"email": email.lower(), "app": app})
        return self._build_user(user) if user else None

    def get_user_credentials(self, email: str, app: str = "akkio") -> Optional[dict]:
        user = get_collection("auth_users").find_one(
            {"email": email.lower(), "app": app},
            {"id": 1, "email": 1, "password_hash": 1, "app": 1},
        )
        return dict(user) if user else None

    def set_user_roles(self, user_id: int, role_ids: list[int]) -> None:
        get_collection("auth_users").update_one(
            {"id": user_id},
            {"$set": {"role_ids": role_ids, "updated_at": utc_now()}},
        )

    def create_user(
        self,
        *,
        name: str,
        email: str,
        password_hash: Optional[str],
        username: Optional[str],
        app: str,
        organization_id: Optional[int],
        role_ids: list[int],
        google_id: Optional[str] = None,
    ) -> dict:
        email = email.lower()
        if get_collection("auth_users").find_one({"email": email, "app": app}):
            raise HTTPException(status_code=400, detail="Email already registered for this app")
        now = utc_now()
        user_id = next_seq("auth_users")
        doc = {
            "id": user_id,
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "username": username,
            "app": app,
            "organization_id": organization_id,
            "google_id": google_id,
            "role_ids": role_ids,
            "created_at": now,
            "updated_at": now,
        }
        get_collection("auth_users").insert_one(doc)
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
        default_organization_id: Optional[int] = None,
        default_role_ids: Optional[list[int]] = None,
    ) -> dict:
        email = email.lower()
        existing = get_collection("auth_users").find_one({"email": email, "app": app})
        if existing:
            get_collection("auth_users").update_one(
                {"id": existing["id"]},
                {
                    "$set": {
                        "google_id": existing.get("google_id") or google_id,
                        "name": existing.get("name") or name,
                        "updated_at": utc_now(),
                    }
                },
            )
            return self.get_user_by_id(existing["id"])  # type: ignore[arg-type]

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

    def list_organizations(self) -> list[dict]:
        return [
            dict(row)
            for row in get_collection("auth_organizations").find().sort("id", 1)
        ]

    def get_organization(self, org_id: int) -> Optional[dict]:
        row = get_collection("auth_organizations").find_one({"id": org_id})
        return dict(row) if row else None

    def create_organization(self, name: str, description: Optional[str]) -> dict:
        now = utc_now()
        doc = {
            "id": next_seq("auth_organizations"),
            "name": name,
            "description": description,
            "created_at": now,
            "updated_at": now,
        }
        get_collection("auth_organizations").insert_one(doc)
        return doc

    def update_organization(self, org_id: int, updates: dict[str, Any]) -> Optional[dict]:
        patch = {k: updates[k] for k in ("name", "description") if k in updates and updates[k] is not None}
        if not patch:
            return self.get_organization(org_id)
        patch["updated_at"] = utc_now()
        row = get_collection("auth_organizations").find_one_and_update(
            {"id": org_id},
            {"$set": patch},
            return_document=True,
        )
        return dict(row) if row else None

    def delete_organization(self, org_id: int) -> bool:
        res = get_collection("auth_organizations").delete_one({"id": org_id})
        return res.deleted_count > 0

    def list_roles(self) -> list[dict]:
        return [_serialize_role(dict(r)) for r in get_collection("auth_roles").find().sort("id", 1)]

    def get_role(self, role_id: int) -> Optional[dict]:
        row = get_collection("auth_roles").find_one({"id": role_id})
        return _serialize_role(dict(row)) if row else None

    def create_role(self, name: str, permissions: list[str]) -> dict:
        if get_collection("auth_roles").find_one({"name": name}):
            raise HTTPException(status_code=400, detail="Role name already exists")
        now = utc_now()
        doc = {
            "id": next_seq("auth_roles"),
            "name": name,
            "permissions": permissions,
            "created_at": now,
            "updated_at": now,
        }
        get_collection("auth_roles").insert_one(doc)
        return _serialize_role(doc)

    def update_role(self, role_id: int, updates: dict[str, Any]) -> Optional[dict]:
        patch = {}
        if updates.get("name") is not None:
            patch["name"] = updates["name"]
        if updates.get("permissions") is not None:
            patch["permissions"] = updates["permissions"]
        if not patch:
            return self.get_role(role_id)
        patch["updated_at"] = utc_now()
        row = get_collection("auth_roles").find_one_and_update(
            {"id": role_id},
            {"$set": patch},
            return_document=True,
        )
        return _serialize_role(dict(row)) if row else None

    def delete_role(self, role_id: int) -> bool:
        res = get_collection("auth_roles").delete_one({"id": role_id})
        return res.deleted_count > 0

    def list_users(self) -> list[dict]:
        users = get_collection("auth_users").find().sort("id", 1)
        return [self._build_user(u) for u in users]

    def update_user(self, user_id: int, updates: dict[str, Any]) -> Optional[dict]:
        patch = {}
        for key in ("name", "email", "username", "app", "password_hash", "organization_id"):
            if key in updates and updates[key] is not None:
                patch[key] = str(updates[key]).lower() if key == "email" else updates[key]
        role_ids = updates.get("role_ids")
        if patch:
            patch["updated_at"] = utc_now()
            get_collection("auth_users").update_one({"id": user_id}, {"$set": patch})
        if role_ids is not None:
            self.set_user_roles(user_id, role_ids)
        return self.get_user_by_id(user_id)

    def delete_user(self, user_id: int) -> bool:
        res = get_collection("auth_users").delete_one({"id": user_id})
        return res.deleted_count > 0

    def get_default_org_and_admin_role(self) -> tuple[Optional[int], Optional[int]]:
        org = get_collection("auth_organizations").find_one(sort=[("id", 1)])
        role = get_collection("auth_roles").find_one({"name": "Admin"})
        return (org["id"] if org else None, role["id"] if role else None)


auth_store = AuthStore()
