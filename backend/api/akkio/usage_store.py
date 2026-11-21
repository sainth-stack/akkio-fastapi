import math
import os
from typing import Optional, TypedDict, Union

from db import MongoDatabase
from db.mongodb import get_collection, utc_now

# Import log_transaction to record usage history
try:
    from .settings_store import log_transaction
except ImportError:

    def log_transaction(*args, **kwargs):
        pass


DEFAULT_CREDITS = int(os.getenv("DEFAULT_AI_CREDITS", "0"))
DEFAULT_STORAGE_MB = int(os.getenv("DEFAULT_STORAGE_MB", "0"))
TOKENS_PER_CREDIT = int(os.getenv("TOKENS_PER_CREDIT", "1000"))  # 1 credit == 1000 tokens by default


class UsageRow(TypedDict):
    email: str
    credits_remaining: int
    storage_remaining_mb: int


DbType = Union[MongoDatabase, "MongoDatabase"]


def _usage_coll():
    return get_collection("user_usage")


def _ensure_usage_table(db: DbType) -> None:
    db.ensure_connection()
    _usage_coll().create_index("email", unique=True)


def get_or_init_usage(db: DbType, email: str) -> UsageRow:
    _ensure_usage_table(db)
    email = (email or "").strip().lower()
    if not email:
        email = "anonymous"
    row = _usage_coll().find_one({"email": email})
    if row:
        return {
            "email": row["email"],
            "credits_remaining": int(row.get("credits_remaining", 0)),
            "storage_remaining_mb": int(row.get("storage_remaining_mb", 0)),
        }
    doc = {
        "email": email,
        "credits_remaining": DEFAULT_CREDITS,
        "storage_remaining_mb": DEFAULT_STORAGE_MB,
        "updated_at": utc_now(),
    }
    _usage_coll().insert_one(doc)
    return {
        "email": email,
        "credits_remaining": DEFAULT_CREDITS,
        "storage_remaining_mb": DEFAULT_STORAGE_MB,
    }


def deduct_tokens(db: DbType, email: Optional[str], total_tokens: int) -> UsageRow:
    if total_tokens is None:
        total_tokens = 0
    total_tokens = int(max(0, total_tokens))
    email = (email or "").strip().lower() or "anonymous"

    get_or_init_usage(db, email)

    denom = max(1, int(TOKENS_PER_CREDIT))
    credits_to_deduct = int(math.ceil(total_tokens / denom)) if total_tokens else 0

    row = _usage_coll().find_one({"email": email})
    assert row is not None
    new_credits = max(int(row.get("credits_remaining", 0)) - credits_to_deduct, 0)
    _usage_coll().update_one(
        {"email": email},
        {"$set": {"credits_remaining": new_credits, "updated_at": utc_now()}},
    )

    if credits_to_deduct > 0:
        try:
            log_transaction(
                db,
                email,
                source="Usage",
                type="Included",
                amount=-float(credits_to_deduct),
                details=f"Used {total_tokens} tokens",
            )
        except Exception as e:
            print(f"Failed to log transaction: {e}")

    updated = _usage_coll().find_one({"email": email})
    return {
        "email": email,
        "credits_remaining": int(updated.get("credits_remaining", 0)),
        "storage_remaining_mb": int(updated.get("storage_remaining_mb", 0)),
    }


def deduct_storage(db: DbType, email: Optional[str], file_size_bytes: int) -> UsageRow:
    if file_size_bytes is None:
        file_size_bytes = 0
    file_size_bytes = int(max(0, file_size_bytes))
    email = (email or "").strip().lower() or "anonymous"

    get_or_init_usage(db, email)
    file_size_mb = math.ceil(file_size_bytes / (1024 * 1024))

    row = _usage_coll().find_one({"email": email})
    assert row is not None
    new_storage = int(row.get("storage_remaining_mb", 0)) - file_size_mb
    _usage_coll().update_one(
        {"email": email},
        {"$set": {"storage_remaining_mb": new_storage, "updated_at": utc_now()}},
    )
    updated = _usage_coll().find_one({"email": email})
    return {
        "email": email,
        "credits_remaining": int(updated.get("credits_remaining", 0)),
        "storage_remaining_mb": int(updated.get("storage_remaining_mb", 0)),
    }
