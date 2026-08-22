from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import List, Optional
import os
import asyncio
import subprocess
from api.auth.request_auth import resolve_user
from api.auth.dependencies import CurrentUser
from api.app_creator.project_access import assert_project_access
from llm_helper import get_llm_for_user
from app_builder.agents.test_generator_agent import generate_tests
from app_builder.services.file_reader import read_project_files
from app_builder.services.runtime_paths import get_projects_dir

router = APIRouter(prefix="/api/test-suite", tags=["Test Suite"])

class GenerateTestsRequest(BaseModel):
    project_name: str

class RunTestsRequest(BaseModel):
    project_name: str
    test_files: Optional[List[str]] = None

@router.post("/generate")
async def trigger_generate_tests(
    req: GenerateTestsRequest,
    current: CurrentUser = Depends(resolve_user),
):
    """
    Generates test scripts for the given project.
    """
    assert_project_access(req.project_name, current)
    projects_dir = get_projects_dir()
    project_path = os.path.join(projects_dir, req.project_name)
    
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail=f"Project not found at {project_path}")

    # Read existing backend files
    # read_project_files already handles the path correctly
    files = read_project_files(req.project_name)
    backend_files = {k: v for k, v in files.items() if k.startswith("backend/") and k.endswith(".py")}
    
    if not backend_files:
         raise HTTPException(status_code=400, detail="No backend files found to test")

    llm = get_llm_for_user(None, temperature=0.2)
    
    try:
        result = await generate_tests(req.project_name, backend_files, llm)
        return {"message": "Tests generated successfully", "files": list(result.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list/{project_name}")
async def list_tests(project_name: str, current: CurrentUser = Depends(resolve_user)):
    """
    Lists all test files in the project.
    """
    assert_project_access(project_name, current)
    projects_dir = get_projects_dir()
    project_path = os.path.join(projects_dir, project_name, "backend", "tests")
    
    if not os.path.exists(project_path):
        return {"tests": []}
        
    tests = []
    for f in os.listdir(project_path):
        if f.startswith("test_") and f.endswith(".py"):
            tests.append(f)
            
    return {"tests": tests}

@router.post("/run")
async def run_tests(req: RunTestsRequest, current: CurrentUser = Depends(resolve_user)):
    """
    Runs pytest for the project and returns the output.
    """
    assert_project_access(req.project_name, current)
    projects_dir = get_projects_dir()
    project_backend_path = os.path.join(projects_dir, req.project_name, "backend")
    
    if not os.path.exists(project_backend_path):
        raise HTTPException(status_code=404, detail="Project backend not found")

    # Ensure pytest is installed in the environment (it should be)
    cmd = ["pytest"]
    if req.test_files:
        # Append specific test files
        for tf in req.test_files:
            cmd.append(f"tests/{tf}")
    else:
        cmd.append("tests/")
        
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=project_backend_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        output = stdout.decode() + "\n" + stderr.decode()
        passed = process.returncode == 0
        
        return {
            "success": passed,
            "output": output,
            "return_code": process.returncode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
