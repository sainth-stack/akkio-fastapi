from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List, Optional

import os

from api.auth.request_auth import resolve_user
from api.auth.dependencies import CurrentUser
from api.app_creator.project_access import assert_project_access
from llm_helper import get_llm_for_user
from app_builder.agents.test_generator_agent import generate_tests
from app_builder.services.file_reader import read_project_files
from app_builder.services.runtime_paths import get_projects_dir
from app_builder.services.ui_test_runner import list_ui_test_files, run_ui_tests

router = APIRouter(prefix="/api/test-suite", tags=["Test Suite"])

class GenerateTestsRequest(BaseModel):
    project_name: str

class RunTestsRequest(BaseModel):
    project_name: str
    test_files: Optional[List[str]] = None
    access_token: Optional[str] = None

@router.post("/generate")
async def trigger_generate_tests(
    req: GenerateTestsRequest,
    current: CurrentUser = Depends(resolve_user),
):
    """
    Generates Playwright UI e2e test scripts for the project frontend.
    """
    assert_project_access(req.project_name, current)
    projects_dir = get_projects_dir()
    project_path = os.path.join(projects_dir, req.project_name)

    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail=f"Project not found at {project_path}")

    files = read_project_files(req.project_name)
    frontend_files = {
        k: v
        for k, v in files.items()
        if k.startswith("frontend/") and k.endswith((".tsx", ".ts", ".jsx", ".js"))
    }

    if not frontend_files:
        raise HTTPException(
            status_code=400,
            detail="No frontend source files found. Generate a full-stack app with UI first.",
        )

    llm = get_llm_for_user(None, temperature=0.2)

    try:
        result = await generate_tests(req.project_name, files, llm)
        if not result:
            raise HTTPException(status_code=500, detail="Failed to generate UI tests")
        return {"message": "UI tests generated successfully", "files": list(result.keys())}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list/{project_name}")
async def list_tests(project_name: str, current: CurrentUser = Depends(resolve_user)):
    """
    Lists Playwright UI test spec files for the project.
    """
    assert_project_access(project_name, current)
    return {"tests": list_ui_test_files(project_name)}

@router.post("/run")
async def run_tests(
    req: RunTestsRequest,
    request: Request,
    current: CurrentUser = Depends(resolve_user),
):
    """
    Runs Playwright UI tests against the built app preview URL.
    """
    assert_project_access(req.project_name, current)
    projects_dir = get_projects_dir()
    project_path = os.path.join(projects_dir, req.project_name)

    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Project not found")

    request_base = str(request.base_url).rstrip("/")

    try:
        passed, output, return_code = await run_ui_tests(
            req.project_name,
            test_files=req.test_files,
            access_token=req.access_token,
            request_base=request_base,
        )
        return {
            "success": passed,
            "output": output,
            "return_code": return_code,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
