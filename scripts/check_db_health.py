#!/usr/bin/env python3
"""Database health check — run from repo root: python scripts/check_db_health.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from db import PostgresDatabase

REQUIRED_TABLES = [
    "akio_data_fastapi",
    "multi_model_sessions",
    "multi_model_files",
    "llm_settings",
    "builder_apps",
]

AUTH_TABLES = [
    ("auth", "users"),
    ("auth", "organizations"),
    ("auth", "roles"),
]


def main():
    print("Database health check")
    PostgresDatabase._ensure_pool()
    db = PostgresDatabase()

    with db.get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1
            print("✅ Connectivity OK")

            for table in REQUIRED_TABLES:
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = %s
                    )
                    """,
                    (table,),
                )
                ok = cursor.fetchone()[0]
                print(f"{'✅' if ok else '❌'} public.{table}")

            for schema, table in AUTH_TABLES:
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = %s AND table_name = %s
                    )
                    """,
                    (schema, table),
                )
                ok = cursor.fetchone()[0]
                print(f"{'✅' if ok else '❌'} {schema}.{table}")

    print("Done.")


if __name__ == "__main__":
    main()
