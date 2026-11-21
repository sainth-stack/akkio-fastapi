#!/usr/bin/env python3
"""Delete all Agentic Builder (builder_kind=fullstack) records from MongoDB."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import os
from pymongo import MongoClient

_uri = os.environ.get("MONGODB_URI", "").strip()
_db_name = os.environ.get("MONGODB_DB_NAME", "akkio").strip()


def get_collection(name: str):
    if not _uri:
        raise SystemExit("MONGODB_URI is not set in akkio-fastapi/.env")
    client = MongoClient(_uri)
    return client[_db_name][name]


def _is_fullstack(doc: dict) -> bool:
    meta = doc.get("metadata") or {}
    if isinstance(meta, dict):
        kind = str(meta.get("builder_kind") or "").strip().lower()
        if kind == "fullstack":
            return True
    pn = str(doc.get("project_name") or "").lower()
    return pn.startswith("fullstack_app_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Purge Agentic Builder apps from MongoDB")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete (default is dry-run list only)",
    )
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="Delete every fullstack app in the DB (not just one user_id)",
    )
    parser.add_argument("--user-id", type=int, default=None, help="Limit deletes to this user_id")
    args = parser.parse_args()

    apps_coll = get_collection("builder_apps")
    dep_coll = get_collection("app_builder_deployments")

    query: dict = {}
    if args.user_id is not None:
        query["user_id"] = args.user_id

    docs = [d for d in apps_coll.find(query) if _is_fullstack(d)]
    if not docs:
        print("No Agentic Builder (fullstack) apps found.")
        return 0

    print(f"Found {len(docs)} Agentic Builder app(s):")
    for d in docs:
        print(
            f"  id={d.get('id')} user_id={d.get('user_id')} "
            f"name={d.get('name')!r} project={d.get('project_name')!r}"
        )

    if not args.confirm:
        print("\nDry run — pass --confirm to delete.")
        return 0

    deleted = 0
    for d in docs:
        aid = int(d["id"])
        dep_coll.delete_many({"app_id": aid})
        res = apps_coll.delete_one({"id": aid})
        deleted += res.deleted_count

    print(f"\nDeleted {deleted} app record(s) and related deployments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())