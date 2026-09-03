from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import uuid

from app_builder.agents.streaming_planner_agent import stream_plan_generation
from app_builder.agents.prd_agent import stream_prd_generation
from app_builder.agents.uiux_agent import stream_uiux_generation
from app_builder.agents.architecture_agent import stream_architecture_generation
from app_builder.agents.styling_agent import stream_styling_generation
from llm_helper import get_llm_for_user
from api.auth.dependencies import CurrentUser
from api.auth.request_auth import resolve_user, user_email_from
from api.app_creator.pipeline_helpers import (
    STEP_COMPLETE_EVENT,
    STEP_PIPELINE,
    touch_job,
    update_pipeline,
)

router = APIRouter(prefix="/api/planning", tags=["Planning"])


class GenRequest(BaseModel):
    requirement: str
    project_name: Optional[str] = None
    prd: Optional[str] = None
    uiux: Optional[str] = None
    architecture: Optional[Dict[str, Any]] = None
    plan: Optional[List[Any]] = None
    design_tokens: Optional[Dict[str, Any]] = None
    app_id: Optional[str] = None
    session_id: Optional[str] = None
    model_name: Optional[str] = None


def _planning_stream(
    step: str,
    request: GenRequest,
    email: str,
    user_id: int | None,
    stream_fn,
    start_message: str,
):
    running, complete, failed = STEP_PIPELINE[step]
    complete_event = STEP_COMPLETE_EVENT[step]
    session_id = request.session_id or f"plan_{step}_{uuid.uuid4().hex[:12]}"

    async def event_stream():
        update_pipeline(request.app_id, email, user_id, running, error=None)
        touch_job(
            session_id,
            app_id=request.app_id,
            job_type="planning",
            status="running",
            step=step,
        )
        try:
            yield json.dumps({"event": "start", "message": start_message, "session_id": session_id}) + "\n"
            async for event in stream_fn():
                if event.get("event") == complete_event:
                    update_pipeline(request.app_id, email, user_id, complete, error=None)
                    touch_job(
                        session_id,
                        app_id=request.app_id,
                        job_type="planning",
                        status="complete",
                        step=step,
                        finished=True,
                    )
                elif event.get("event") == "error":
                    err = event.get("message", "Planning step failed")
                    update_pipeline(request.app_id, email, user_id, failed, error=err)
                    touch_job(
                        session_id,
                        app_id=request.app_id,
                        job_type="planning",
                        status="failed",
                        step=step,
                        error=err,
                        finished=True,
                    )
                yield json.dumps(event) + "\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            err = str(e)
            update_pipeline(request.app_id, email, user_id, failed, error=err)
            touch_job(
                session_id,
                app_id=request.app_id,
                job_type="planning",
                status="failed",
                step=step,
                error=err,
                finished=True,
            )
            yield json.dumps({"event": "error", "message": err}) + "\n"

    return event_stream()


@router.post("/plan")
async def generate_plan_step(request: GenRequest, current: CurrentUser = Depends(resolve_user)):
    email = user_email_from(current)
    uid = current.id or None

    async def stream_fn():
        llm = get_llm_for_user(email, model_name=request.model_name, temperature=0.5, streaming=True)
        async for event in stream_plan_generation(
            request.requirement,
            request.prd or "",
            request.architecture or {},
            llm,
            request.uiux or "",
        ):
            yield event

    return StreamingResponse(
        _planning_stream("plan", request, email, uid, stream_fn, "Starting Implementation Plan..."),
        media_type="application/x-ndjson",
    )


@router.post("/prd")
async def generate_prd_step(request: GenRequest, current: CurrentUser = Depends(resolve_user)):
    email = user_email_from(current)
    uid = current.id or None

    async def stream_fn():
        llm = get_llm_for_user(email, model_name=request.model_name, temperature=0.7, streaming=True)
        async for event in stream_prd_generation(request.requirement, llm, skip_plan=True):
            yield event

    return StreamingResponse(
        _planning_stream("prd", request, email, uid, stream_fn, "Starting PRD..."),
        media_type="application/x-ndjson",
    )


@router.post("/uiux")
async def generate_uiux_step(request: GenRequest, current: CurrentUser = Depends(resolve_user)):
    email = user_email_from(current)
    uid = current.id or None

    async def stream_fn():
        llm = get_llm_for_user(email, model_name=request.model_name, temperature=0.7, streaming=True)
        async for event in stream_uiux_generation(request.requirement, request.prd or "", llm):
            yield event

    return StreamingResponse(
        _planning_stream("uiux", request, email, uid, stream_fn, "Starting UI/UX..."),
        media_type="application/x-ndjson",
    )


@router.post("/style")
async def generate_style_step(request: GenRequest, current: CurrentUser = Depends(resolve_user)):
    email = user_email_from(current)
    uid = current.id or None

    async def stream_fn():
        llm = get_llm_for_user(email, model_name=request.model_name, temperature=0.4, streaming=True)
        async for event in stream_styling_generation(
            request.requirement,
            request.prd or "",
            request.uiux or "",
            llm,
        ):
            yield event

    return StreamingResponse(
        _planning_stream("style", request, email, uid, stream_fn, "Starting Design System..."),
        media_type="application/x-ndjson",
    )


@router.post("/architecture")
async def generate_arch_step(request: GenRequest, current: CurrentUser = Depends(resolve_user)):
    email = user_email_from(current)
    uid = current.id or None

    async def stream_fn():
        llm = get_llm_for_user(email, model_name=request.model_name, temperature=0.3, streaming=True)
        async for event in stream_architecture_generation(
            request.requirement,
            request.prd or "",
            [],
            llm,
            request.uiux or "",
            request.design_tokens,
        ):
            yield event

    return StreamingResponse(
        _planning_stream("architecture", request, email, uid, stream_fn, "Starting Architecture..."),
        media_type="application/x-ndjson",
    )
