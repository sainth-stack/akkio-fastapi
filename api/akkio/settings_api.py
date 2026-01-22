from fastapi import APIRouter, Query, Body, HTTPException
from typing import Optional
from database import PostgresDatabase
from .settings_store import (
    get_workspace_settings, 
    update_workspace_settings, 
    WorkspaceSettings, 
    update_plan,
    get_transactions,
    get_daily_usage,
    get_analytics_summary,
    log_transaction # Exposed for testing/seeding
)
from .usage_store import get_or_init_usage

settings_router = APIRouter()

@settings_router.get("/settings/general")
async def get_general_settings(user_email: str = Query(..., alias="email")):
    db = PostgresDatabase()
    db.ensure_connection()
    try:
        settings = get_workspace_settings(db, user_email)
        return settings
    finally:
        db.close()

@settings_router.post("/settings/general")
async def save_general_settings(settings: WorkspaceSettings):
    db = PostgresDatabase()
    db.ensure_connection()
    try:
        updated = update_workspace_settings(db, settings.email, settings)
        return updated
    finally:
        db.close()

@settings_router.post("/settings/plan")
async def change_plan(
    email: str = Body(..., embed=True),
    plan_id: str = Body(..., embed=True)
):
    valid_plans = ["free", "pro", "advanced", "ultra"]
    if plan_id not in valid_plans:
        raise HTTPException(status_code=400, detail="Invalid plan ID")
    
    db = PostgresDatabase()
    db.ensure_connection()
    try:
        # Update plan
        new_plan = update_plan(db, email, plan_id)
        
        # Logic to add credits based on plan could go here
        # For now just updating the plan record
        
        return {"status": "success", "plan_id": new_plan}
    finally:
        db.close()

@settings_router.get("/settings/analytics")
async def get_analytics(user_email: str = Query(..., alias="email")):
    db = PostgresDatabase()
    db.ensure_connection()
    try:
        # Get usage state
        usage_state = get_or_init_usage(db, user_email)
        credits_remaining = usage_state["credits_remaining"]
        
        # Get analytics summary
        summary = get_analytics_summary(db, user_email)
        
        # Get chart data
        chart_data = get_daily_usage(db, user_email, days=30)
        
        # Get transactions
        transactions = get_transactions(db, user_email, limit=20)
        
        return {
            "summary": {
                "remaining_credits": credits_remaining,
                "total_used": summary["total_used"],
                "daily_average": summary["daily_average"],
                "active_users": summary["active_users"]
            },
            "chart_data": chart_data,
            "transactions": transactions
        }
    finally:
        db.close()

@settings_router.post("/settings/seed-data")
async def seed_data(email: str = Body(..., embed=True)):
    """Helper to seed some data for visualization if empty"""
    db = PostgresDatabase()
    db.ensure_connection()
    try:
        # Check if we have transactions
        txs = get_transactions(db, email, limit=1)
        if not txs:
            # Seed some initial data
            log_transaction(db, email, "Onboarding", "Included", 100.0, "Initial grant")
            log_transaction(db, email, "Chat", "Included", -5.0, "")
            log_transaction(db, email, "Chat", "Included", -5.0, "")
            log_transaction(db, email, "Model Training", "Included", -50.0, "Sales prediction model")
            return {"status": "seeded"}
        return {"status": "already_has_data"}
    finally:
        db.close()


