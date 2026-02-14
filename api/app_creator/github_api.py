"""
GitHub API - Handles GitHub integration for generated apps
"""
import os
import sys
from pathlib import Path

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from api.app_creator.github_service import GitHubService
from database import PostgresDatabase

router = APIRouter(prefix="/api/github", tags=["GitHub"])

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
    print(f"GitHub API: DB init warning: {e}")


class GitHubPushRequest(BaseModel):
    project_name: str
    repo_url: Optional[str] = None
    create_repo: bool = False
    repo_name: Optional[str] = None
    branch: str = "main"
    commit_message: Optional[str] = "Update from Akkio App Builder"


class GitHubStatusRequest(BaseModel):
    project_name: str


def get_github_service() -> GitHubService:
    """Get configured GitHub service instance."""
    github_token = os.environ.get("GITHUB_TOKEN")
    return GitHubService(github_token)


def get_project_path(project_name: str) -> str:
    """Get the full path to a project."""
    from app_builder.services.runtime_paths import get_projects_dir
    projects_base = get_projects_dir()
    project_path = os.path.join(projects_base, project_name)
    
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail=f"Project not found: {project_name}")
    
    return project_path


@router.post("/push")
async def push_to_github(request: GitHubPushRequest):
    """
    Push a generated app to GitHub.
    
    This endpoint can either:
    1. Push to an existing repository (provide repo_url)
    2. Create a new repository and push (set create_repo=True and provide repo_name)
    """
    try:
        # Get project path
        project_path = get_project_path(request.project_name)
        
        # Validate request
        if request.create_repo and not request.repo_name:
            raise HTTPException(
                status_code=400,
                detail="repo_name is required when create_repo=True"
            )
        
        if not request.create_repo and not request.repo_url:
            raise HTTPException(
                status_code=400,
                detail="Either repo_url or create_repo=True with repo_name is required"
            )
        
        # Get GitHub service and push
        github_service = get_github_service()
        
        # First commit any changes
        commit_result = github_service.commit_changes(
            project_path,
            request.commit_message
        )
        
        # Then push to GitHub
        result = github_service.push_to_github(
            project_path=project_path,
            repo_url=request.repo_url,
            branch=request.branch,
            create_repo=request.create_repo,
            repo_name=request.repo_name
        )
        
        if result['status'] == 'success':
            return {
                "status": "success",
                "message": result['message'],
                "repo_url": result.get('repo_url'),
                "branch": result.get('branch')
            }
        else:
            raise HTTPException(status_code=500, detail=result['message'])
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/init")
async def initialize_git(request: GitHubStatusRequest):
    """
    Initialize a git repository for a project.
    """
    try:
        # Get project path
        project_path = get_project_path(request.project_name)
        
        # Initialize git
        github_service = get_github_service()
        result = github_service.initialize_git_repo(project_path)
        
        if result['status'] == 'success':
            return {
                "status": "success",
                "message": result['message']
            }
        else:
            raise HTTPException(status_code=500, detail=result['message'])
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_git_status(
    project_name: str = Query(..., description="Project name")
):
    """
    Get the git status of a project.
    """
    try:
        # Get project path
        project_path = get_project_path(project_name)
        
        # Get status
        github_service = get_github_service()
        status = github_service.get_repo_status(project_path)
        
        return status
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/commit")
async def commit_changes(
    project_name: str = Query(..., description="Project name"),
    message: str = Query("Update from Akkio App Builder", description="Commit message")
):
    """
    Commit changes in a project.
    """
    try:
        # Get project path
        project_path = get_project_path(project_name)
        
        # Commit changes
        github_service = get_github_service()
        result = github_service.commit_changes(project_path, message)
        
        if result['status'] == 'success':
            return {
                "status": "success",
                "message": result['message']
            }
        else:
            raise HTTPException(status_code=500, detail=result['message'])
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
