"""
App Builder Apps API – CRUD for created apps (list, create, get, update, delete).
Uses MongoDB for app creator storage (Akkio main app uses Postgres only).
"""

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from .app_creator import tree_from_files_dict

from app_builder_db import get_app_builder_db

router = APIRouter(tags=["App Builder Apps"])

db = get_app_builder_db()


class CreateAppRequest(BaseModel):
    user_email: str
    app_name: str
    prompt: str
    project_name: str
    prd: Optional[str] = None
    generated_uiux: Optional[str] = None
    plan: Optional[List[Any]] = None
    architecture: Optional[dict] = None
    agents_state: Optional[dict] = None
    generated_code_json: Optional[dict] = None


class UpdateAppRequest(BaseModel):
    app_name: Optional[str] = None
    prompt: Optional[str] = None
    project_name: Optional[str] = None
    prd: Optional[str] = None
    generated_uiux: Optional[str] = None
    plan: Optional[List[Any]] = None
    architecture: Optional[dict] = None
    agents_state: Optional[dict] = None
    generated_code_json: Optional[dict] = None


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
            agents_state=request.agents_state,
            generated_code_json=request.generated_code_json,
        )
        return {"status": "success", "app": app, "message": "App created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/apps/{app_id}")
async def get_app(app_id: str, user_email: str = Query(..., description="User email")):
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
async def update_app(app_id: str, request: UpdateAppRequest, user_email: str = Query(..., description="User email")):
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
            agents_state=request.agents_state,
            generated_code_json=request.generated_code_json,
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
async def delete_app(app_id: str, user_email: str = Query(..., description="User email")):
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


@router.get("/apps/{app_id}/code")
async def get_app_code(app_id: str, user_email: str = Query(..., description="User email")):
    """Retrieve the generated code and file tree from the database."""
    try:
        app = db.get_app_builder_app(app_id, user_email=user_email)
        if not app:
            raise HTTPException(status_code=404, detail="App not found")
        
        files = app.get("generated_code_json") or {}
        tree = tree_from_files_dict(files) if files else []
        
        return {
            "status": "success",
            "project_name": app.get("project_name"),
            "files": files,
            "tree": tree
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
