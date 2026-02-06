from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from app_builder.agents.streaming_planner_agent import stream_plan_generation
from app_builder.agents.prd_agent import stream_prd_generation
from app_builder.agents.uiux_agent import stream_uiux_generation
from app_builder.agents.architecture_agent import stream_architecture_generation
from llm_helper import get_llm_for_user

router = APIRouter(prefix="/api/planning", tags=["Planning"])

class GenRequest(BaseModel):
    requirement: str
    user_email: Optional[str] = None
    project_name: Optional[str] = None
    # Context from previous steps
    prd: Optional[str] = None
    uiux: Optional[str] = None
    architecture: Optional[Dict[str, Any]] = None
    plan: Optional[List[Any]] = None

@router.post("/plan")
async def generate_plan_step(request: GenRequest):
    async def event_stream():
        try:
            llm = get_llm_for_user(request.user_email, temperature=0.5, streaming=True)
            yield json.dumps({"event": "start", "message": "Starting Implementation Plan..."}) + "\n"
            
            async for event in stream_plan_generation(
                request.requirement, 
                request.prd or "", 
                request.architecture or {}, 
                llm,
                request.uiux or ""
            ):
                yield json.dumps(event) + "\n"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json.dumps({"event": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")

@router.post("/prd")
async def generate_prd_step(request: GenRequest):
    async def event_stream():
        try:
            llm = get_llm_for_user(request.user_email, temperature=0.7, streaming=True)
            yield json.dumps({"event": "start", "message": "Starting PRD..."}) + "\n"
            
            # Pass skip_plan=True to only generate PRD
            async for event in stream_prd_generation(request.requirement, llm, skip_plan=True):
                yield json.dumps(event) + "\n"
                
        except Exception as e:
            yield json.dumps({"event": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")

@router.post("/uiux")
async def generate_uiux_step(request: GenRequest):
    async def event_stream():
        try:
            llm = get_llm_for_user(request.user_email, temperature=0.7, streaming=True)
            yield json.dumps({"event": "start", "message": "Starting UI/UX..."}) + "\n"
            async for event in stream_uiux_generation(request.requirement, request.prd or "", llm):
                yield json.dumps(event) + "\n"
        except Exception as e:
            yield json.dumps({"event": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")

@router.post("/architecture")
async def generate_arch_step(request: GenRequest):
    async def event_stream():
        try:
            llm = get_llm_for_user(request.user_email, temperature=0.3, streaming=True) # Lower temp for arch
            yield json.dumps({"event": "start", "message": "Starting Architecture..."}) + "\n"
            
            async for event in stream_architecture_generation(
                request.requirement, 
                request.prd or "", 
                [], 
                llm,
                request.uiux or ""
            ):
                yield json.dumps(event) + "\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json.dumps({"event": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
