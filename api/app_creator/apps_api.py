"""
App Builder Apps API — CRUD for created apps (Postgres-backed).
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.dependencies import CurrentUser
from api.auth.request_auth import resolve_user, user_email_from
from db.app_builder import get_app_builder_db

router = APIRouter(tags=["App Builder Apps"])

db = get_app_builder_db()


def _tree_from_files_dict(files: Dict[str, str]) -> List[Dict[str, Any]]:
    root: List[Dict[str, Any]] = []

    def ensure_folder(children: List[Dict[str, Any]], name: str, path: str) -> Dict[str, Any]:
        existing = next((c for c in children if c.get("type") == "folder" and c.get("name") == name), None)
        if existing:
            return existing
        node = {"name": name, "type": "folder", "path": path, "children": []}
        children.append(node)
        return node

    for rel_path in sorted(files.keys()):
        parts = rel_path.split("/")
        cur_children = root
        cur_path = ""
        for part in parts[:-1]:
            cur_path = f"{cur_path}/{part}" if cur_path else part
            folder = ensure_folder(cur_children, part, cur_path)
            cur_children = folder["children"]
        cur_children.append({"name": parts[-1], "type": "file", "path": rel_path})
    return root


class CreateAppRequest(BaseModel):
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
    pipeline_status: Optional[str] = None
    pipeline_error: Optional[str] = None
    build_status: Optional[str] = None
    build_log: Optional[str] = None
    build_error: Optional[str] = None
    preview_url: Optional[str] = None
    live_url: Optional[str] = None
    design_tokens: Optional[dict] = None
    design_system_md: Optional[str] = None
    llm_model: Optional[str] = None


@router.get("/models")
async def list_available_models(current: CurrentUser = Depends(resolve_user)):
    from api.app_creator.pipeline_helpers import ALLOWED_MODELS
    from llm_config import get_llm_config

    default = get_llm_config(user_email_from(current)).get("model")
    return {"status": "success", "models": ALLOWED_MODELS, "default_model": default}


@router.get("/apps")
async def list_apps(current: CurrentUser = Depends(resolve_user)):
    try:
        apps = db.get_user_app_builder_apps(
            user_email_from(current), user_id=current.id or None
        )
        return {"status": "success", "apps": apps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apps")
async def create_app(request: CreateAppRequest, current: CurrentUser = Depends(resolve_user)):
    try:
        app = db.create_app_builder_app(
            user_email=user_email_from(current),
            app_name=request.app_name,
            prompt=request.prompt,
            project_name=request.project_name,
            prd=request.prd,
            generated_uiux=request.generated_uiux,
            plan=request.plan,
            architecture=request.architecture,
            agents_state=request.agents_state,
            generated_code_json=request.generated_code_json,
            user_id=current.id or None,
        )
        return {"status": "success", "app": app, "message": "App created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/apps/{app_id}")
async def get_app(app_id: str, current: CurrentUser = Depends(resolve_user)):
    try:
        app = db.get_app_builder_app(
            app_id, user_email=user_email_from(current), user_id=current.id or None
        )
        if not app:
            raise HTTPException(status_code=404, detail="App not found")
        return {"status": "success", "app": app}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/apps/{app_id}")
async def update_app(
    app_id: str,
    request: UpdateAppRequest,
    current: CurrentUser = Depends(resolve_user),
):
    try:
        email = user_email_from(current)
        count = db.update_app_builder_app(
            app_id=app_id,
            user_email=email,
            app_name=request.app_name,
            prompt=request.prompt,
            project_name=request.project_name,
            prd=request.prd,
            generated_uiux=request.generated_uiux,
            plan=request.plan,
            architecture=request.architecture,
            agents_state=request.agents_state,
            generated_code_json=request.generated_code_json,
            pipeline_status=request.pipeline_status,
            pipeline_error=request.pipeline_error,
            build_status=request.build_status,
            build_log=request.build_log,
            build_error=request.build_error,
            preview_url=request.preview_url,
            live_url=request.live_url,
            user_id=current.id or None,
        )
        if count == 0:
            raise HTTPException(status_code=404, detail="App not found")
        app = db.get_app_builder_app(app_id, user_email=email, user_id=current.id or None)
        return {"status": "success", "app": app, "message": "App updated"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/apps/{app_id}")
async def delete_app(app_id: str, current: CurrentUser = Depends(resolve_user)):
    try:
        count = db.delete_app_builder_app(
            app_id, user_email_from(current), user_id=current.id or None
        )
        if count == 0:
            raise HTTPException(status_code=404, detail="App not found")
        return {"status": "success", "message": "App deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/apps/{app_id}/status")
async def get_app_status(app_id: str, current: CurrentUser = Depends(resolve_user)):
    try:
        app = db.get_app_builder_app(
            app_id, user_email=user_email_from(current), user_id=current.id or None
        )
        if not app:
            raise HTTPException(status_code=404, detail="App not found")
        job = db.get_active_job_for_app(app_id)
        return {
            "status": "success",
            "app_id": app_id,
            "pipeline_status": app.get("pipeline_status"),
            "pipeline_error": app.get("pipeline_error"),
            "build_status": app.get("build_status"),
            "build_log": app.get("build_log"),
            "build_error": app.get("build_error"),
            "preview_url": app.get("preview_url"),
            "live_url": app.get("live_url"),
            "job": job,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{session_id}")
async def get_job(session_id: str, current: CurrentUser = Depends(resolve_user)):
    try:
        job = db.get_job_by_session_id(session_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.get("app_id"):
            app = db.get_app_builder_app(
                job["app_id"],
                user_email=user_email_from(current),
                user_id=current.id or None,
            )
            if not app:
                raise HTTPException(status_code=404, detail="Job not found")
        return {"status": "success", "job": job}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/apps/{app_id}/code")
async def get_app_code(app_id: str, current: CurrentUser = Depends(resolve_user)):
    try:
        app = db.get_app_builder_app(
            app_id, user_email=user_email_from(current), user_id=current.id or None
        )
        if not app:
            raise HTTPException(status_code=404, detail="App not found")

        files = app.get("generated_code_json") or {}
        tree = _tree_from_files_dict(files) if files else []

        return {
            "status": "success",
            "project_name": app.get("project_name"),
            "files": files,
            "tree": tree,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
