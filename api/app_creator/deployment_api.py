"""
Deployment API — local Hostinger deploy on the same VPS (Postgres-backed).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional, Union

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from api.app_creator.hostinger_deploy_service import (
    IN_PROGRESS_STATUSES,
    HostingerDeployService,
    TERMINAL_FAILURE,
    TERMINAL_SUCCESS,
)
from api.app_creator.pipeline_helpers import public_base_url
from api.auth.dependencies import CurrentUser
from api.auth.request_auth import resolve_user, user_email_from
from api.app_creator.project_access import assert_project_access
from db.app_builder import get_app_builder_db

router = APIRouter(prefix="/api/deployment", tags=["Deployment"])

logger = logging.getLogger("app_builder")
db = get_app_builder_db()


class DeployRequest(BaseModel):
    app_id: Optional[Union[int, str]] = None
    project_name: str
    rebuild: bool = False
    run_tests: Optional[bool] = None


def _resolve_app_and_project(
    app_id: Optional[Union[int, str]],
    project_name: str,
    current: CurrentUser,
) -> tuple[dict | None, str]:
    email = user_email_from(current)
    uid = current.id if current.id else None
    app = None
    resolved_name = project_name

    if app_id:
        app = db.get_app_builder_app(app_id, user_email=email, user_id=uid)
        if not app:
            raise HTTPException(status_code=404, detail="App not found")
        assert_project_access(app["project_name"], current)
        resolved_name = app.get("project_name") or project_name
    else:
        assert_project_access(project_name, current)
        app = db.get_app_by_project_name(project_name, user_id=uid, user_email=email)

    return app, resolved_name


def _assert_build_ready(app: dict | None, rebuild: bool) -> None:
    if rebuild:
        return
    if not app or app.get("build_status") != "BUILD_SUCCESS":
        raise HTTPException(
            status_code=400,
            detail=(
                "Run the app successfully in the Build tab before deploying, "
                "or set rebuild=true to build during deploy."
            ),
        )


def _assert_not_in_progress(deployment: dict | None) -> None:
    if deployment and deployment.get("deployment_status") in IN_PROGRESS_STATUSES:
        raise HTTPException(status_code=409, detail="Deployment already in progress for this app")


async def perform_deployment(
    deployment_id: str,
    app_id: Optional[Union[int, str]],
    project_name: str,
    user_email: str,
    user_id: int | None,
    rebuild: bool = False,
    run_tests: Optional[bool] = None,
    public_base: str | None = None,
):
    log_lines: list[str] = []

    def on_status(status: str, line: str, error: str | None = None) -> None:
        log_lines.append(line)
        db.update_deployment(
            deployment_id=deployment_id,
            deployment_status=status,
            error_message=error if status == TERMINAL_FAILURE else None,
            deploy_log="\n".join(log_lines),
        )

    try:
        service = HostingerDeployService(db=db)
        result = await asyncio.to_thread(
            service.deploy,
            deployment_id=deployment_id,
            app_id=app_id,
            project_name=project_name,
            user_email=user_email,
            user_id=user_id,
            rebuild=rebuild,
            run_tests=run_tests,
            on_status=on_status,
            public_base=public_base,
        )

        if result.get("status") == "success":
            db.update_deployment(
                deployment_id=deployment_id,
                frontend_url=result.get("live_url") or result.get("frontend_url"),
                backend_url=result.get("backend_url"),
                deployment_status=TERMINAL_SUCCESS,
                error_message=None,
                deploy_log=result.get("logs") or "\n".join(log_lines),
            )
        else:
            db.update_deployment(
                deployment_id=deployment_id,
                deployment_status=TERMINAL_FAILURE,
                error_message=result.get("message", "Deployment failed"),
                deploy_log=result.get("logs") or "\n".join(log_lines),
            )

    except Exception as e:
        msg = str(e)
        print(f"Deployment error: {msg}")
        log_lines.append(msg)
        db.update_deployment(
            deployment_id=deployment_id,
            deployment_status=TERMINAL_FAILURE,
            error_message=msg,
            deploy_log="\n".join(log_lines),
        )
        if app_id:
            try:
                db.update_app_builder_app(
                    app_id=app_id,
                    user_email=user_email,
                    user_id=user_id,
                    pipeline_status="DEPLOY_FAILED",
                    pipeline_error=msg,
                )
            except Exception:
                pass


def _deployment_payload(deployment: dict, app: dict | None = None) -> dict:
    live_url = deployment.get("frontend_url")
    if app and app.get("live_url"):
        live_url = app.get("live_url")
    elif app and app.get("preview_url") and not live_url:
        live_url = app.get("preview_url")
    return {
        "deployment_id": deployment["id"],
        "app_id": deployment.get("app_id"),
        "project_name": deployment["project_name"],
        "frontend_url": deployment.get("frontend_url"),
        "backend_url": deployment.get("backend_url"),
        "live_url": live_url,
        "preview_url": app.get("preview_url") if app else deployment.get("frontend_url"),
        "frontend_port": deployment.get("frontend_port"),
        "backend_port": deployment.get("backend_port"),
        "deployment_status": deployment["deployment_status"],
        "error_message": deployment.get("error_message"),
        "deploy_log": deployment.get("deploy_log"),
        "deployed_at": deployment.get("deployed_at"),
        "updated_at": deployment.get("updated_at"),
        "build_status": app.get("build_status") if app else None,
        "mode": deployment.get("mode", "deployed"),
    }


def register_local_preview(
    *,
    app_id,
    project_name: str,
    frontend_url: str,
    backend_url: str | None = None,
    user_email: str | None = None,
    user_id: int | None = None,
) -> dict | None:
    """After Run App succeeds, register preview URL so Deploy tab shows Live."""
    try:
        deployment = db.upsert_local_deployment(
            app_id=app_id,
            project_name=project_name,
            frontend_url=frontend_url,
            backend_url=backend_url,
            deploy_log="Local preview registered after Run App (BUILD_SUCCESS)",
        )
        if app_id and user_email:
            db.update_app_builder_app(
                app_id=app_id,
                user_email=user_email,
                user_id=user_id,
                preview_url=frontend_url,
                live_url=frontend_url,
            )
        return deployment
    except Exception as exc:
        print(f"[deployment] register_local_preview failed: {exc}")
        return None


@router.post("/deploy")
async def deploy_app(
    request: DeployRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    current: CurrentUser = Depends(resolve_user),
):
    try:
        email = user_email_from(current)
        uid = current.id or None
        deploy_public_base = public_base_url(
            request_base=str(http_request.base_url).rstrip("/")
        )

        app, project_name = _resolve_app_and_project(
            request.app_id, request.project_name, current
        )
        app_id = (app or {}).get("id") if app else None
        if not app_id and request.app_id:
            # Do not pass stale client app_id when the app row is missing server-side
            logger.warning(
                "[deployment] request app_id=%s not resolved — using project_name=%s only",
                request.app_id,
                project_name,
            )

        _assert_build_ready(app, request.rebuild)

        existing = None
        if app_id:
            existing = db.get_deployment_by_app_id(app_id)
        if not existing:
            existing = db.get_deployment_by_project_name(project_name)
        _assert_not_in_progress(existing)

        deployment = db.create_deployment(
            app_id=app_id,
            project_name=project_name,
            deployment_status="QUEUED",
            deploy_log="Deploy queued",
        )
        deployment_id = deployment["id"]

        background_tasks.add_task(
            perform_deployment,
            deployment_id,
            app_id,
            project_name,
            email,
            uid,
            request.rebuild,
            request.run_tests,
            deploy_public_base,
        )

        return {
            "status": "started",
            "message": "Deployment started. Poll status for progress.",
            "deployment_id": deployment_id,
            "app_id": app_id,
            "project_name": project_name,
            "deployment_status": "QUEUED",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_deployment_history(
    app_id: str = Query(..., description="App ID"),
    limit: int = Query(20, ge=1, le=100),
    current: CurrentUser = Depends(resolve_user),
):
    email = user_email_from(current)
    uid = current.id or None
    app = db.get_app_builder_app(app_id, user_email=email, user_id=uid)
    if not app:
        raise HTTPException(status_code=404, detail="App not found")
    assert_project_access(app["project_name"], current)
    deployments = db.list_deployments_by_app_id(app_id, limit=limit)
    return {
        "status": "success",
        "app_id": app_id,
        "deployments": [_deployment_payload(d, app) for d in deployments],
    }


@router.get("/status")
async def get_deployment_status(
    app_id: Optional[str] = Query(None, description="App ID"),
    project_name: Optional[str] = Query(None, description="Project name"),
    deployment_id: Optional[str] = Query(None, description="Specific deployment run"),
    current: CurrentUser = Depends(resolve_user),
):
    try:
        email = user_email_from(current)
        uid = current.id or None
        deployment = None
        app = None

        if deployment_id:
            deployment = db.get_deployment_by_id(deployment_id)
            if not deployment:
                raise HTTPException(status_code=404, detail="Deployment not found")
            project_name = deployment.get("project_name")
            assert_project_access(project_name, current)
            if deployment.get("app_id"):
                app = db.get_app_builder_app(deployment["app_id"], user_email=email, user_id=uid)
            return _deployment_payload(deployment, app)

        if app_id:
            app = db.get_app_builder_app(app_id, user_email=email, user_id=uid)
            if not app:
                raise HTTPException(status_code=404, detail="App not found")
            assert_project_access(app["project_name"], current)
            deployment = db.get_deployment_by_app_id(app_id)
            project_name = app.get("project_name")
        elif project_name:
            assert_project_access(project_name, current)
            app = db.get_app_by_project_name(project_name, user_id=uid, user_email=email)
            deployment = db.get_deployment_by_project_name(project_name)
            if app:
                app_id = app.get("id")
        else:
            raise HTTPException(status_code=400, detail="Either app_id or project_name is required")

        if not deployment:
            build_status = app.get("build_status") if app else None
            preview_url = app.get("preview_url") if app else None
            if build_status == "BUILD_SUCCESS" and preview_url:
                return {
                    "app_id": app_id,
                    "project_name": project_name,
                    "deployment_status": "LOCAL_PREVIEW",
                    "message": "App is running locally. Use the preview URL below or click Deploy to register.",
                    "build_status": build_status,
                    "preview_url": preview_url,
                    "frontend_url": preview_url,
                    "live_url": preview_url,
                    "backend_url": f"{preview_url.split('/app/')[0]}/api/apps/{project_name}" if preview_url and project_name else None,
                    "mode": "local",
                }
            return {
                "app_id": app_id,
                "project_name": project_name,
                "deployment_status": "not_deployed",
                "message": (
                    "No deployment yet. Run the app in the Build tab first, then deploy."
                    if build_status != "BUILD_SUCCESS"
                    else "Build succeeded — click Deploy to register your live URL."
                ),
                "build_status": build_status,
            }

        return _deployment_payload(deployment, app)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/redeploy")
async def redeploy_app(
    request: DeployRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    current: CurrentUser = Depends(resolve_user),
):
    """Redeploy with rebuild=true by default."""
    redeploy_request = DeployRequest(
        app_id=request.app_id,
        project_name=request.project_name,
        rebuild=True,
        run_tests=request.run_tests,
    )
    return await deploy_app(redeploy_request, background_tasks, http_request, current)
