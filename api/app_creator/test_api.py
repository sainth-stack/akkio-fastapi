from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List, Optional
import time
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


class RunChecksRequest(BaseModel):
    project_name: Optional[str] = None
    app_id: Optional[str] = None


@router.post("/run-checks")
async def run_checks(
    req: RunChecksRequest,
    current: CurrentUser = Depends(resolve_user),
):
    """
    Run the Playwright gate against an existing generated app (frontend_only track).
    Returns pass/fail, duration, gate route results, and any failing stage.
    Registered at /api/test-suite/run-checks; the DeploymentView hits /test/run-checks
    via the frontend proxy alias.
    """
    project_name = req.project_name
    if not project_name and req.app_id:
        # Try to resolve project name from app_id via DB
        try:
            from db.app_builder import get_app_builder_app_by_id
            app = get_app_builder_app_by_id(req.app_id)
            if app:
                project_name = app.get("project_name") or app.get("name")
        except Exception:
            pass

    if not project_name:
        raise HTTPException(status_code=400, detail="project_name or app_id required")

    assert_project_access(project_name, current)

    files = read_project_files(project_name)
    if not files:
        raise HTTPException(status_code=404, detail="No generated files found for this project")

    # Pull blueprint from plan_json if available
    blueprint_json: dict = {}
    try:
        from db.app_builder import get_app_builder_app_by_project_name
        app_row = get_app_builder_app_by_project_name(project_name)
        if app_row and app_row.get("plan_json"):
            plan = app_row["plan_json"]
            blueprint_json = plan.get("blueprint_json") or {}
    except Exception:
        pass

    t0 = time.time()
    events = []

    async def collect_event(ev: dict):
        events.append(ev)

    try:
        from app_builder.services.fo_gate_service import run_gate
        from app_builder.services.metrics_service import GenerationMetrics
        metrics = GenerationMetrics()
        gate_result = await run_gate(
            project_name=project_name,
            files=files,
            blueprint_json=blueprint_json,
            on_event=collect_event,
            metrics=metrics,
        )
        duration_s = time.time() - t0
        return {
            "passed": gate_result.passed,
            "duration_s": round(duration_s, 2),
            "build_status": "GATE_PASSED" if gate_result.passed else "GATE_FAILED",
            "failing_stage": None if gate_result.passed else "gate_playwright",
            "gate_results": {
                "failing_routes": [
                    {
                        "route": r.route,
                        "console_errors": r.console_errors,
                        "failed_requests": r.failed_requests,
                        "blank_body": r.blank_body,
                        "error": r.error,
                    }
                    for r in (gate_result.route_results or [])
                    if not r.passed
                ],
                "all_routes": len(gate_result.route_results or []),
            },
            "build_log": gate_result.build_log,
            "error": gate_result.error,
            "events": events[-40:],  # last 40 events for debugging
        }
    except ImportError:
        # fo_gate_service not yet installed — run basic tsc verify instead
        from app_builder.services.verify_service import verify_stage
        verify_result = await verify_stage(project_name, files, files, "checks")
        duration_s = time.time() - t0
        errors = verify_result.errors or []
        return {
            "passed": verify_result.ok,
            "duration_s": round(duration_s, 2),
            "build_status": "BUILD_SUCCESS" if verify_result.ok else "BUILD_FAILED",
            "failing_stage": None if verify_result.ok else "tsc_verify",
            "gate_results": None,
            "errors": errors,
            "build_log": verify_result.log,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
