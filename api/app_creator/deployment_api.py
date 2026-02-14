"""
Deployment API - Handles app deployment to EC2 instance.
Uses MongoDB for app creator storage (Akkio main app uses Postgres only).
"""
import os
import sys
import asyncio
from pathlib import Path

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Union

from api.app_creator.deployment_service import DeploymentService
from app_builder_db import get_app_builder_db

router = APIRouter(prefix="/api/deployment", tags=["Deployment"])

db = get_app_builder_db()


class DeployRequest(BaseModel):
    app_id: Optional[Union[int, str]] = None  # str for MongoDB ObjectId
    project_name: str
    user_email: str


class DeploymentStatusResponse(BaseModel):
    deployment_id: Optional[str]
    app_id: Optional[Union[int, str]]
    project_name: str
    frontend_url: Optional[str]
    backend_url: Optional[str]
    frontend_port: Optional[int]
    backend_port: Optional[int]
    deployment_status: str
    error_message: Optional[str]
    deployed_at: Optional[str]


def get_deployment_service() -> DeploymentService:
    """Get configured deployment service instance."""
    target_ec2_ip = os.environ.get("DEPLOY_EC2_IP")
    ssh_key_path = os.environ.get("DEPLOY_SSH_KEY_PATH")
    ssh_user = os.environ.get("DEPLOY_SSH_USER", "ubuntu")
    
    if not target_ec2_ip or not ssh_key_path:
        raise HTTPException(
            status_code=500,
            detail="Deployment not configured. Set DEPLOY_EC2_IP and DEPLOY_SSH_KEY_PATH environment variables."
        )
    
    return DeploymentService(target_ec2_ip, ssh_key_path, ssh_user)


async def perform_deployment(app_id: Optional[Union[int, str]], project_name: str, user_email: str):
    """Background task to perform the actual deployment."""
    deployment_id = None
    
    try:
        # Create initial deployment record
        deployment = db.create_deployment(
            app_id=app_id,
            project_name=project_name,
            deployment_status='deploying'
        )
        deployment_id = deployment['id']
        
        # Get project files path - use same logic as file_writer
        from app_builder.services.runtime_paths import get_projects_dir
        projects_base = get_projects_dir()
        project_path = os.path.join(projects_base, project_name)
        
        if not os.path.exists(project_path):
            raise Exception(f"Project path not found: {project_path}")
        
        # Get deployment service and deploy
        deploy_service = get_deployment_service()
        result = await asyncio.to_thread(deploy_service.deploy_app, project_name, project_path)
        
        if result['status'] == 'success':
            # Update deployment record with success
            db.update_deployment(
                deployment_id=deployment_id,
                frontend_url=result.get('frontend_url'),
                backend_url=result.get('backend_url'),
                deployment_status='deployed'
            )
        else:
            # Update with error
            db.update_deployment(
                deployment_id=deployment_id,
                deployment_status='failed',
                error_message=result.get('message', 'Unknown error')
            )
    
    except Exception as e:
        print(f"Deployment error: {e}")
        if deployment_id:
            db.update_deployment(
                deployment_id=deployment_id,
                deployment_status='failed',
                error_message=str(e)
            )


@router.post("/deploy")
async def deploy_app(request: DeployRequest, background_tasks: BackgroundTasks):
    """
    Deploy a generated app to EC2 instance.
    
    This endpoint starts the deployment process in the background and returns immediately.
    Use the /status endpoint to check deployment progress.
    """
    try:
        app_id = request.app_id
        
        # If app_id is provided, verify it exists and belongs to user
        if app_id:
            app = db.get_app_builder_app(app_id, user_email=request.user_email)
            if not app:
                raise HTTPException(status_code=404, detail="App not found")
            
            # Check if there's already a deployment in progress
            existing_deployment = db.get_deployment_by_app_id(app_id)
            if existing_deployment and existing_deployment['deployment_status'] == 'deploying':
                raise HTTPException(
                    status_code=409,
                    detail="Deployment already in progress for this app"
                )
        else:
            # No app_id provided, check by project_name
            existing_deployment = db.get_deployment_by_project_name(request.project_name)
            if existing_deployment and existing_deployment['deployment_status'] == 'deploying':
                raise HTTPException(
                    status_code=409,
                    detail="Deployment already in progress for this project"
                )
        
        # Start deployment in background
        background_tasks.add_task(
            perform_deployment,
            app_id,
            request.project_name,
            request.user_email
        )
        
        return {
            "status": "started",
            "message": "Deployment started. Check status endpoint for progress.",
            "app_id": app_id,
            "project_name": request.project_name
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_deployment_status(
    app_id: Optional[str] = Query(None, description="App ID (MongoDB ObjectId string)"),
    project_name: Optional[str] = Query(None, description="Project name"),
    user_email: str = Query(..., description="User email")
):
    """
    Get the deployment status for an app.
    
    Returns the latest deployment record with URLs if deployed successfully.
    Can query by app_id or project_name.
    """
    try:
        # Get latest deployment by app_id or project_name
        deployment = None
        
        if app_id:
            # Verify app belongs to user if app_id provided
            app = db.get_app_builder_app(app_id, user_email=user_email)
            if not app:
                raise HTTPException(status_code=404, detail="App not found")
            deployment = db.get_deployment_by_app_id(app_id)
        elif project_name:
            deployment = db.get_deployment_by_project_name(project_name)
        else:
            raise HTTPException(status_code=400, detail="Either app_id or project_name is required")
        
        if not deployment:
            return {
                "app_id": app_id,
                "project_name": project_name,
                "deployment_status": "not_deployed",
                "message": "No deployment found"
            }
        
        return {
            "deployment_id": deployment['id'],
            "app_id": deployment.get('app_id'),
            "project_name": deployment['project_name'],
            "frontend_url": deployment.get('frontend_url'),
            "backend_url": deployment.get('backend_url'),
            "frontend_port": deployment.get('frontend_port'),
            "backend_port": deployment.get('backend_port'),
            "deployment_status": deployment['deployment_status'],
            "error_message": deployment.get('error_message'),
            "deployed_at": deployment.get('deployed_at'),
            "updated_at": deployment.get('updated_at')
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/redeploy")
async def redeploy_app(
    request: DeployRequest,
    background_tasks: BackgroundTasks
):
    """
    Redeploy an existing app. This will stop the old deployment and start a new one.
    """
    try:
        app_id = request.app_id
        project_name = request.project_name
        
        # If app_id is provided, verify it
        if app_id:
            app = db.get_app_builder_app(app_id, user_email=request.user_email)
            if not app:
                raise HTTPException(status_code=404, detail="App not found")
            project_name = app['project_name'] or request.project_name
        
        if not project_name:
            raise HTTPException(status_code=400, detail="Project name is required")
        
        # Start redeployment in background
        background_tasks.add_task(
            perform_deployment,
            app_id,
            project_name,
            request.user_email
        )
        
        return {
            "status": "started",
            "message": "Redeployment started. Check status endpoint for progress.",
            "app_id": app_id,
            "project_name": project_name
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
