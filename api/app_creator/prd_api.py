"""
PRD API - Handles PRD and Plan generation with streaming
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from app_builder.agents.prd_agent import stream_prd_generation
from llm_helper import get_llm_for_user

router = APIRouter(prefix="/api/prd", tags=["PRD"])


class PRDRequest(BaseModel):
    requirement: str
    user_email: Optional[str] = None


@router.post("/generate")
async def generate_prd(request: PRDRequest):
    """
    Generate PRD and plan with streaming response.
    
    Returns newline-delimited JSON events:
    - {"event": "prd_chunk", "data": "..."}
    - {"event": "prd_complete", "data": "full prd"}
    - {"event": "plan_start", "message": "..."}
    - {"event": "plan_complete", "data": [...]}
    - {"event": "done", "message": "..."}
    """
    async def event_stream():
        try:
            # Initialize LLM
            llm = get_llm_for_user(
                request.user_email,
                temperature=0.7,
                streaming=True
            )
            
            yield json.dumps({
                "event": "start",
                "message": "Starting PRD generation..."
            }) + "\n"
            
            # Stream PRD generation
            async for event in stream_prd_generation(request.requirement, llm):
                yield json.dumps(event) + "\n"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json.dumps({
                "event": "error",
                "message": str(e)
            }) + "\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson"
    )
