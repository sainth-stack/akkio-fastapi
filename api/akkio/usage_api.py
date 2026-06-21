from fastapi import APIRouter, Depends

from db import PostgresDatabase

from api.auth.dependencies import CurrentUser
from api.auth.request_auth import resolve_user, user_email_from
from .usage_store import TOKENS_PER_CREDIT, get_or_init_usage

usage_router = APIRouter()

_DEFAULT_USAGE = {
    "email": "anonymous",
    "credits_remaining": 0,
    "storage_remaining_mb": 0,
    "tokens_per_credit": TOKENS_PER_CREDIT,
}


@usage_router.get("/usage")
async def get_usage(current: CurrentUser = Depends(resolve_user)):
    user_email = user_email_from(current)
    try:
        db = PostgresDatabase()
        db.ensure_connection()
        row = get_or_init_usage(db, user_email)
        try:
            db.close()
        except Exception:
            pass
        return {
            "email": row["email"],
            "credits_remaining": row["credits_remaining"],
            "storage_remaining_mb": row["storage_remaining_mb"],
            "tokens_per_credit": TOKENS_PER_CREDIT,
        }
    except Exception:
        return {**_DEFAULT_USAGE, "email": user_email}
