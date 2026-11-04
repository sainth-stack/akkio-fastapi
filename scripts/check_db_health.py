#!/usr/bin/env python3
"""Database health check — run: python scripts/check_db_health.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from db import MongoDatabase
from db.mongodb import get_database, ping_database

EXPECTED_COLLECTIONS = [
    "auth_users",
    "auth_roles",
    "auth_organizations",
    "llm_settings",
    "user_usage",
    "multi_model_sessions",
    "multi_model_files",
    "builder_apps",
    "dynamic_app_rows",
]


def main():
    print("MongoDB health check")
    MongoDatabase._ensure_pool()
    db = MongoDatabase()
    db.ensure_connection()
    db.ensure_indexes()

    status = ping_database()
    print(f"✅ Ping OK — db={status.get('database')}")

    names = set(get_database().list_collection_names())
    for coll in EXPECTED_COLLECTIONS:
        # Collections appear after first write; index init is enough for empty DB
        marker = "✅" if coll in names or coll.startswith("auth_") else "·"
        print(f"{marker} {coll}" + (" (present)" if coll in names else " (lazy)"))

    print("Done.")


if __name__ == "__main__":
    main()
