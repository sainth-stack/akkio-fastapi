import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from api.app_creator.github_service import GitHubService
from api.auth.dependencies import CurrentUser
from api.auth.request_auth import resolve_user
from api.app_creator.project_access import assert_project_access

router = APIRouter(prefix="/api/github", tags=["GitHub"])


class GitHubPushRequest(BaseModel):
    project_name: str
    repo_url: Optional[str] = None
    create_repo: bool = False
    repo_name: Optional[str] = None
    branch: str = "main"
    commit_message: Optional[str] = "Update from Akkio App Builder"


def get_github_service() -> GitHubService:
    github_token = os.environ.get("GITHUB_TOKEN")
    return GitHubService(github_token)


def get_project_path(project_name: str) -> str:
    from app_builder.services.runtime_paths import get_projects_dir

    projects_base = get_projects_dir()
    project_path = os.path.join(projects_base, project_name)

    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail=f"Project not found: {project_name}")

    return project_path


@router.post("/push")
async def push_to_github(
    request: GitHubPushRequest,
    current: CurrentUser = Depends(resolve_user),
):
    try:
        assert_project_access(request.project_name, current)
        project_path = get_project_path(request.project_name)

        if request.create_repo and not request.repo_name:
            raise HTTPException(status_code=400, detail="repo_name is required when create_repo=True")

        if not request.create_repo and not request.repo_url:
            raise HTTPException(
                status_code=400,
                detail="Either repo_url or create_repo=True with repo_name is required",
            )

        github_service = get_github_service()
        github_service.commit_changes(project_path, request.commit_message)

        result = github_service.push_to_github(
            project_path=project_path,
            repo_url=request.repo_url,
            branch=request.branch,
            create_repo=request.create_repo,
            repo_name=request.repo_name,
        )

        if result["status"] == "success":
            return {
                "status": "success",
                "message": result["message"],
                "repo_url": result.get("repo_url"),
                "branch": result.get("branch"),
            }
        raise HTTPException(status_code=500, detail=result["message"])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
