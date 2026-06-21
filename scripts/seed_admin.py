#!/usr/bin/env python3
"""
Seed default organization, Admin role, and admin user for Akkio auth.

Usage (from akkio-fastapi directory):
  python scripts/seed_admin.py

Environment:
  PGHOST, PGUSER, PGPASSWORD, PGDATABASE  — Postgres connection
  SEED_ADMIN_EMAIL    (default: admin@example.com)
  SEED_ADMIN_PASSWORD (default: Admin@12345)
  SEED_ADMIN_NAME     (default: Akkio Admin)
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv(override=True)

from api.auth.security import hash_password  # noqa: E402
from api.auth.store import auth_store  # noqa: E402

DEFAULT_PERMISSIONS = [
    "home_read",
    "reports_read",
    "reports_write",
    "admin_read",
    "admin_write",
]


def main() -> None:
    auth_store.init_schema()

    orgs = auth_store.list_organizations()
    if orgs:
        org = orgs[0]
        print(f"Organization already exists: id={org['id']} name={org['name']}")
    else:
        org = auth_store.create_organization(
            name="Default Organization",
            description="Default Akkio organization",
        )
        print(f"Created organization: id={org['id']} name={org['name']}")

    roles = auth_store.list_roles()
    admin_role = next((r for r in roles if r["name"] == "Admin"), None)
    if admin_role:
        print(f"Admin role already exists: id={admin_role['id']}")
    else:
        admin_role = auth_store.create_role(name="Admin", permissions=DEFAULT_PERMISSIONS)
        print(f"Created Admin role: id={admin_role['id']}")

    email = os.getenv("SEED_ADMIN_EMAIL", "admin@example.com")
    password = os.getenv("SEED_ADMIN_PASSWORD", "Admin@12345")
    name = os.getenv("SEED_ADMIN_NAME", "Akkio Admin")

    existing = auth_store.get_user_by_email(email, app="akkio")
    if existing:
        print(f"Admin user already exists: id={existing['id']} email={existing['email']}")
        return

    user = auth_store.create_user(
        name=name,
        email=email,
        password_hash=hash_password(password),
        username=email.split("@")[0],
        app="akkio",
        organization_id=org["id"],
        role_ids=[admin_role["id"]],
    )
    print(f"Created admin user: id={user['id']} email={user['email']}")
    print(f"Login with email={email} password={password}")


if __name__ == "__main__":
    main()
