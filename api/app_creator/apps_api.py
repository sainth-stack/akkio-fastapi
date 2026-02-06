"""
App Builder Apps API – CRUD for created apps (list, create, get, update, delete).
"""

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Any

from database import PostgresDatabase

router = APIRouter(tags=["App Builder Apps"])

db = PostgresDatabase()
try:
    db.create_connection(
        user=os.environ.get("PGUSER", "test_owner"),
        password=os.environ.get("PGPASSWORD", "tcWI7unQ6REA"),
        database=os.environ.get("PGDATABASE", "test"),
        host=os.environ.get("PGHOST", "ep-yellow-recipe-a5fny139.us-east-2.aws.neon.tech"),
    )
    db.create_table()
    db.create_training_tables()
except Exception as e:
    print(f"App Builder Apps API: DB init warning: {e}")


class CreateAppRequest(BaseModel):
    user_email: str
    app_name: str
    prompt: str
    project_name: str
    prd: Optional[str] = None
    generated_uiux: Optional[str] = None
    plan: Optional[List[Any]] = None
    architecture: Optional[dict] = None


class UpdateAppRequest(BaseModel):
    app_name: Optional[str] = None
    prompt: Optional[str] = None
    project_name: Optional[str] = None
    prd: Optional[str] = None
    generated_uiux: Optional[str] = None
    plan: Optional[List[Any]] = None
    architecture: Optional[dict] = None


@router.get("/apps")
async def list_apps(user_email: str = Query(..., description="User email")):
    """List all app builder apps for the user."""
    try:
        apps = db.get_user_app_builder_apps(user_email)
        return {"status": "success", "apps": apps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apps")
async def create_app(request: CreateAppRequest):
    """Create a new app builder app (e.g. after user enters prompt)."""
    try:
        app = db.create_app_builder_app(
            user_email=request.user_email,
            app_name=request.app_name,
            prompt=request.prompt,
            project_name=request.project_name,
            prd=request.prd,
            generated_uiux=request.generated_uiux,
            plan=request.plan,
            architecture=request.architecture,
        )
        return {"status": "success", "app": app, "message": "App created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/apps/{app_id}")
async def get_app(app_id: int, user_email: str = Query(..., description="User email")):
    """Get a single app by id (for edit)."""
    try:
        app = db.get_app_builder_app(app_id, user_email=user_email)
        if not app:
            raise HTTPException(status_code=404, detail="App not found")
        return {"status": "success", "app": app}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/apps/{app_id}")
async def update_app(app_id: int, request: UpdateAppRequest, user_email: str = Query(..., description="User email")):
    """Update an app (e.g. after PRD/plan/architecture generated)."""
    try:
        count = db.update_app_builder_app(
            app_id=app_id,
            user_email=user_email,
            app_name=request.app_name,
            prompt=request.prompt,
            project_name=request.project_name,
            prd=request.prd,
            generated_uiux=request.generated_uiux,
            plan=request.plan,
            architecture=request.architecture,
        )
        if count == 0:
            raise HTTPException(status_code=404, detail="App not found")
        app = db.get_app_builder_app(app_id, user_email=user_email)
        return {"status": "success", "app": app, "message": "App updated"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/apps/{app_id}")
async def delete_app(app_id: int, user_email: str = Query(..., description="User email")):
    """Delete an app."""
    try:
        count = db.delete_app_builder_app(app_id, user_email)
        if count == 0:
            raise HTTPException(status_code=404, detail="App not found")
        return {"status": "success", "message": "App deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
