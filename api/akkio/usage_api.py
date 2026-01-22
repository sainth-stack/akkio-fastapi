from fastapi import APIRouter, Query

from database import PostgresDatabase

from .usage_store import TOKENS_PER_CREDIT, get_or_init_usage


usage_router = APIRouter()


@usage_router.get("/usage")
async def get_usage(user_email: str = Query(default="anonymous")):
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



