from typing import Dict, Any, AsyncGenerator
import json
from langchain_core.messages import HumanMessage, SystemMessage

async def stream_plan_generation(requirement: str, prd: str, architecture: Dict[str, Any], llm, uiux: str = "") -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streams implementation plan based on PRD and Architecture.
    """
    
    arch_str = json.dumps(architecture, indent=2) if architecture else "Standard React + FastAPI architecture"

    yield {
        "event": "plan_start",
        "message": "Generating detailed implementation plan..."
    }
    
    plan_prompt = f"""Based on the PRD and Architecture below, create a detailed implementation plan.

PRD Summary:
{prd[:2000]}...

Architecture:
{arch_str}

UI/UX Design:
{uiux if uiux else "Standard modern UI/UX design."}

Use the following JSON format for each step:
STEP: {{"id": <number>, "title": "<brief title>", "description": "<detailed description>", "status": "pending", "dependencies": [<array of step numbers>]}}

Rules:
1. Cover Setup, Database, Backend API, Frontend Logic, Frontend UI, Integration.
2. **PRODUCTION READY**: Include explicit steps for Robust Error Handling, UI Polishing (animations, responsiveness), and Final Integration Testing.
3. Be specific (mention specific files or components from architecture).
4. 10-14 steps to ensure thoroughness.
5. Output each step immediately.

Start now with STEP 1:"""

    messages = [
        SystemMessage(content="You are a technical project planner. Output each step immediately as you generate it."),
        HumanMessage(content=plan_prompt)
    ]
    
    plan_buffer = ""
    collected_steps = []
    
    async for chunk in llm.astream(messages):
        if hasattr(chunk, 'content'):
            plan_buffer += chunk.content
            
            # Look for complete STEP: {...} patterns
            while "STEP:" in plan_buffer:
                step_start = plan_buffer.find("STEP:")
                json_start = plan_buffer.find("{", step_start)
                if json_start == -1: break
                
                brace_count = 0
                json_end = -1
                for i in range(json_start, len(plan_buffer)):
                    if plan_buffer[i] == '{': brace_count += 1
                    elif plan_buffer[i] == '}': 
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end == -1: break
                
                try:
                    step_json = plan_buffer[json_start:json_end]
                    step = json.loads(step_json)
                    collected_steps.append(step)
                    yield { "event": "plan_step", "data": step }
                    plan_buffer = plan_buffer[json_end:]
                except json.JSONDecodeError:
                    plan_buffer = plan_buffer[step_start + 5:]

    yield {
        "event": "plan_complete",
        "data": collected_steps
    }
