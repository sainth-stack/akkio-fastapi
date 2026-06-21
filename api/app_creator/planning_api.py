from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json

from app_builder.agents.streaming_planner_agent import stream_plan_generation
from app_builder.agents.prd_agent import stream_prd_generation
from app_builder.agents.uiux_agent import stream_uiux_generation
from app_builder.agents.architecture_agent import stream_architecture_generation
from llm_helper import get_llm_for_user
from api.auth.dependencies import CurrentUser
from api.auth.request_auth import resolve_user, user_email_from

router = APIRouter(prefix="/api/planning", tags=["Planning"])


class GenRequest(BaseModel):
    requirement: str
    project_name: Optional[str] = None
    prd: Optional[str] = None
    uiux: Optional[str] = None
    architecture: Optional[Dict[str, Any]] = None
    plan: Optional[List[Any]] = None


@router.post("/plan")
async def generate_plan_step(request: GenRequest, current: CurrentUser = Depends(resolve_user)):
    email = user_email_from(current)

    async def event_stream():
        try:
            llm = get_llm_for_user(email, temperature=0.5, streaming=True)
            yield json.dumps({"event": "start", "message": "Starting Implementation Plan..."}) + "\n"
            async for event in stream_plan_generation(
                request.requirement,
                request.prd or "",
                request.architecture or {},
                llm,
                request.uiux or "",
            ):
                yield json.dumps(event) + "\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json.dumps({"event": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.post("/prd")
async def generate_prd_step(request: GenRequest, current: CurrentUser = Depends(resolve_user)):
    email = user_email_from(current)

    async def event_stream():
        try:
            llm = get_llm_for_user(email, temperature=0.7, streaming=True)
            yield json.dumps({"event": "start", "message": "Starting PRD..."}) + "\n"
            async for event in stream_prd_generation(request.requirement, llm, skip_plan=True):
                yield json.dumps(event) + "\n"
        except Exception as e:
            yield json.dumps({"event": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.post("/uiux")
async def generate_uiux_step(request: GenRequest, current: CurrentUser = Depends(resolve_user)):
    email = user_email_from(current)

    async def event_stream():
        try:
            llm = get_llm_for_user(email, temperature=0.7, streaming=True)
            yield json.dumps({"event": "start", "message": "Starting UI/UX..."}) + "\n"
            async for event in stream_uiux_generation(request.requirement, request.prd or "", llm):
                yield json.dumps(event) + "\n"
        except Exception as e:
            yield json.dumps({"event": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.post("/architecture")
async def generate_arch_step(request: GenRequest, current: CurrentUser = Depends(resolve_user)):
    email = user_email_from(current)

    async def event_stream():
        try:
            llm = get_llm_for_user(email, temperature=0.3, streaming=True)
            yield json.dumps({"event": "start", "message": "Starting Architecture..."}) + "\n"
            async for event in stream_architecture_generation(
                request.requirement,
                request.prd or "",
                [],
                llm,
                request.uiux or "",
            ):
                yield json.dumps(event) + "\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json.dumps({"event": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
