#!/usr/bin/env python3
"""Verify MongoDB connectivity, schema init, and admin seed."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv(override=True)


def main() -> int:
    from db.mongodb import MONGODB_DB_NAME, MONGODB_URI, ping_database, validate_db_config

    print("=== Akkio MongoDB verification ===")
    if not MONGODB_URI.strip():
        print("FAIL: MONGODB_URI is not set")
        return 1
    print(f"URI: {MONGODB_URI.split('@')[-1] if '@' in MONGODB_URI else '(local)'}")
    print(f"DB:  {MONGODB_DB_NAME}")

    try:
        validate_db_config()
        status = ping_database()
        print(f"Ping: OK — database={status.get('database')} collections={len(status.get('collections') or [])}")
    except Exception as exc:
        print(f"FAIL: ping — {exc}")
        return 1

    try:
        from db.init_db import init_schemas

        init_schemas()
        print("Init schemas: OK")
    except Exception as exc:
        print(f"FAIL: init_schemas — {exc}")
        return 1

    try:
        from scripts.seed_admin import main as seed_main

        seed_main()
        print("Seed admin: OK")
    except Exception as exc:
        print(f"FAIL: seed_admin — {exc}")
        return 1

    try:
        from api.auth.store import auth_store

        user = auth_store.get_user_by_email(os.getenv("SEED_ADMIN_EMAIL", "admin@gmail.com"), app="akkio")
        if not user:
            print("FAIL: admin user not found after seed")
            return 1
        print(f"Admin user: id={user['id']} email={user['email']}")
    except Exception as exc:
        print(f"FAIL: auth lookup — {exc}")
        return 1

    print("=== All checks passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
