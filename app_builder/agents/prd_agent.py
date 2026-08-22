"""
PRD Agent - Generates Product Requirements Document and detailed plan using LLM
"""
from typing import Dict, Any, AsyncGenerator
import json
from langchain_core.messages import HumanMessage, SystemMessage


def _chunk_text(content) -> str:
    """Normalize LangChain chunk content to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
        return "".join(parts)
    return str(content)


async def stream_prd_generation(requirement: str, llm, skip_plan: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streams PRD generation using LLM.
    
    Args:
        requirement: User requirement description
        llm: Initialized LLM instance
        skip_plan: If True, stops after PRD generation
    """
    
    system_prompt = """You are a Product Requirements Document (PRD) specialist.
Your task is to create a CONCISE and EFFECTIVE PRD with perfect structure and formatting.
Use clean Markdown so the document displays correctly. Avoid unnecessary details.

You MUST follow this exact structure. Use these exact headings and spacing:

# 1. Product Overview

[One clear sentence about the app.]

# 2. Business Requirements

## 2.1 Functional Requirements (FR)

- [Bullet list of functional requirements: what the system must do]
- [One requirement per bullet; be specific and actionable]

## 2.2 Non-Functional Requirements (NFR)

- [Bullet list of non-functional requirements: performance, security, usability, scalability, etc.]
- [One requirement per bullet]

# 3. Core Features

- [Bulleted list of essential features]
- [One feature per bullet]

# 4. Process Flows / Use Cases

[For each main user flow or use case, describe briefly:]
- **Use Case 1:** [Actor] – [Goal/Action] – [Outcome]
- **Use Case 2:** [Actor] – [Goal/Action] – [Outcome]
[Add 3–6 key use cases as needed]

Formatting rules:
- Use exactly one blank line between sections.
- Use `#` for main sections (1–4), `##` for subsections (2.1, 2.2).
- Use `- ` for all bullet lists; no mixed styles.
- Keep sentences short and clear. No run-on paragraphs.
- Do not add a Tech Stack section."""

    user_prompt = f"""Create a concise and well-formatted PRD for:

{requirement}

Follow the exact structure (1. Product Overview, 2. Business Requirements with FR and NFR, 3. Core Features, 4. Process Flows / Use Cases). Use clean Markdown so it displays perfectly."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # Stream the PRD generation
    full_prd = ""
    async for chunk in llm.astream(messages):
        if hasattr(chunk, 'content'):
            content = _chunk_text(chunk.content)
            if not content:
                continue
            full_prd += content
            yield {
                "event": "prd_chunk",
                "data": content
            }
    
    if not full_prd.strip():
        yield {
            "event": "error",
            "message": "PRD generation returned empty content. Check your LLM API key and model settings.",
        }
        return

    # Parse the plan from the PRD
    yield {
        "event": "prd_complete",
        "data": full_prd
    }
    
    if skip_plan:
        yield { "event": "done", "message": "PRD generation complete" }
        return

    # Now generate structured plan with streaming
    yield {
        "event": "plan_start",
        "message": "Generating structured plan..."
    }
    
    plan_prompt = f"""Based on the following PRD, create a detailed implementation plan with specific, actionable steps.

{full_prd}

You must provide 5-8 detailed steps. For EACH step, output it immediately in this exact format:

STEP: {{"id": <number>, "title": "<brief title>", "description": "<detailed description>", "status": "pending", "dependencies": [<array of step numbers this depends on>]}}

Requirements:
- Output each step immediately as you think of it
- Start with foundational setup steps
- Include dependencies on previous steps where logical
- Cover: setup, data models, backend, frontend, integration, testing
- Be specific and actionable

Start now with STEP 1:"""

    plan_messages = [
        SystemMessage(content="You are a technical project planner. Output each step immediately as you generate it."),
        HumanMessage(content=plan_prompt)
    ]
    
    plan_buffer = ""
    collected_steps = []
    
    async for chunk in llm.astream(plan_messages):
        if hasattr(chunk, 'content'):
            plan_buffer += _chunk_text(chunk.content)
            
            # Look for complete STEP: {...} patterns
            while "STEP:" in plan_buffer:
                step_start = plan_buffer.find("STEP:")
                # Find the JSON object
                json_start = plan_buffer.find("{", step_start)
                if json_start == -1:
                    break
                
                # Find matching closing brace
                brace_count = 0
                json_end = -1
                for i in range(json_start, len(plan_buffer)):
                    if plan_buffer[i] == '{':
                        brace_count += 1
                    elif plan_buffer[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end == -1:
                    break
                
                try:
                    step_json = plan_buffer[json_start:json_end]
                    step = json.loads(step_json)
                    collected_steps.append(step)
                    
                    # Stream this step immediately
                    yield {
                        "event": "plan_step",
                        "data": step
                    }
                    
                    plan_buffer = plan_buffer[json_end:]
                except json.JSONDecodeError:
                    # Move past this STEP marker and try again
                    plan_buffer = plan_buffer[step_start + 5:]
    
    # If we got steps, send them all as complete
    if collected_steps:
        yield {
            "event": "plan_complete",
            "data": collected_steps
        }
    else:
        # Fallback to basic plan
        fallback_plan = [
            {"id": 1, "title": "Initialize project structure", "description": "Set up development environment with required frameworks and tools", "status": "pending", "dependencies": []},
            {"id": 2, "title": "Design database schema", "description": "Create entity models and database structure", "status": "pending", "dependencies": [1]},
            {"id": 3, "title": "Implement data layer", "description": "Set up ORM and database connection", "status": "pending", "dependencies": [2]},
            {"id": 4, "title": "Build backend API endpoints", "description": "Create REST API for all operations", "status": "pending", "dependencies": [3]},
            {"id": 5, "title": "Setup frontend framework", "description": "Initialize React application structure", "status": "pending", "dependencies": [1]},
            {"id": 6, "title": "Create UI components", "description": "Build reusable UI components", "status": "pending", "dependencies": [5]},
            {"id": 7, "title": "Implement main views", "description": "Create application screens and layouts", "status": "pending", "dependencies": [6]},
            {"id": 8, "title": "Integrate frontend with backend", "description": "Connect UI to API endpoints", "status": "pending", "dependencies": [4, 7]},
            {"id": 9, "title": "Add authentication", "description": "Implement user authentication system", "status": "pending", "dependencies": [8]},
            {"id": 10, "title": "Implement core features", "description": "Build main application functionality", "status": "pending", "dependencies": [9]},
            {"id": 11, "title": "Add error handling", "description": "Implement comprehensive error handling", "status": "pending", "dependencies": [10]},
            {"id": 12, "title": "Testing and debugging", "description": "Test all features and fix issues", "status": "pending", "dependencies": [11]}
        ]
        
        # Stream fallback steps one by one
        for step in fallback_plan:
            yield {
                "event": "plan_step",
                "data": step
            }
        
        yield {
            "event": "plan_complete",
            "data": fallback_plan
        }
    
    yield {
        "event": "done",
        "message": "PRD and plan generation complete"
    }
