"""
Agent API - Handles agent execution with WebSocket for real-time updates
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import asyncio
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from app_builder.schemas.requirements import UserRequirement
from app_builder.schemas.plan import ProjectPlan
from app_builder.agents.requirement_agent import requirement_agent
from app_builder.agents.architecture_agent import architecture_agent
from app_builder.agents.code_generator_agent import code_generator_agent
from app_builder.agents.dynamic_code_generator import generate_code_from_plan
from app_builder.services.file_writer import file_writer, write_project_file
from llm_helper import get_llm_for_user
from database import PostgresDatabase

router = APIRouter(prefix="/api/agents", tags=["Agents"])

db = PostgresDatabase()
try:
    db.create_connection(
        user=os.environ.get("PGUSER", "test_owner"),
        password=os.environ.get("PGPASSWORD", "tcWI7unQ6REA"),
        database=os.environ.get("PGDATABASE", "test"),
        host=os.environ.get("PGHOST", "ep-yellow-recipe-a5fny139.us-east-2.aws.neon.tech"),
    )
    db.create_table()
    db.create_training_tables()
except Exception as e:
    print(f"Agent API: DB init warning: {e}")


class AgentExecutionRequest(BaseModel):
    requirement: str
    plan: List[Dict[str, Any]]
    project_name: str


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(json.dumps(message))


manager = ConnectionManager()


async def execute_requirement_agent(websocket: WebSocket, requirement: str):
    """Execute requirement clarification agent"""
    try:
        await websocket.send_text(json.dumps({
            "event": "agent_start",
            "agent": "requirement_agent",
            "message": "Analyzing and clarifying requirements..."
        }))
        
        # Simulate some processing time for better UX
        await asyncio.sleep(0.5)
        
        clarified = requirement_agent(UserRequirement(description=requirement))
        
        await websocket.send_text(json.dumps({
            "event": "agent_complete",
            "agent": "requirement_agent",
            "data": clarified,
            "message": "Requirements clarified successfully"
        }))
        
        return clarified
        
    except WebSocketDisconnect:
        print("Client disconnected during requirement clarification", file=sys.stderr)
        raise
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "event": "agent_error",
                "agent": "requirement_agent",
                "error": str(e)
            }))
        except (WebSocketDisconnect, RuntimeError):
            pass
        raise


async def execute_architecture_agent(
    websocket: WebSocket,
    requirement: str,
    prd: str,
    plan: list,
    llm,
    uiux: str = ""
):
    """Execute architecture design agent with streaming"""
    try:
        from app_builder.agents.architecture_agent import stream_architecture_generation
        
        await websocket.send_text(json.dumps({
            "event": "agent_start",
            "agent": "architecture_agent",
            "message": "Analyzing requirements and designing architecture..."
        }))
        
        architecture_data = None
        
        # Stream architecture generation events
        async for event in stream_architecture_generation(requirement, prd, plan, llm, uiux):
            event_type = event.get("event")
            
            if event_type == "architecture_progress":
                await websocket.send_text(json.dumps({
                    "event": "agent_progress",
                    "agent": "architecture_agent",
                    "message": event.get("message")
                }))
            
            elif event_type == "architecture_complete":
                architecture_data = event.get("data")
                await websocket.send_text(json.dumps({
                    "event": "agent_complete",
                    "agent": "architecture_agent",
                    "data": architecture_data,
                    "message": "Architecture designed successfully"
                }))
        
        # Return structured architecture object
        # Return structured architecture object
        from app_builder.schemas.architecture import ArchitectureDecision
        return ArchitectureDecision(
            frontend_structure=architecture_data.get("frontend_structure", {}),
            backend_structure=architecture_data.get("backend_structure", {}),
            database_schema=architecture_data.get("database_schema", {}),
            deployment=architecture_data.get("deployment", {}),
            rationale=architecture_data.get("rationale", ""),
            project_structure=architecture_data.get("project_structure", {"backend": [], "frontend": []})
        )
        
    except WebSocketDisconnect:
        print(f"Client disconnected during architecture generation", file=sys.stderr)
        raise
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "event": "agent_error",
                "agent": "architecture_agent",
                "error": str(e)
            }))
        except (WebSocketDisconnect, RuntimeError):
            pass
        raise


async def execute_code_generator_agent(
    websocket: WebSocket, 
    requirement: str,
    prd: str,
    plan: list,
    architecture, 
    project_name: str,
    uiux: str = ""
):
    """Execute dynamic code generation agent using LLM"""
    try:
        # Debug logging
        import sys
        print("\n" + "=" * 80, file=sys.stderr)
        print("CODE GENERATOR AGENT - DEBUG INFO:", file=sys.stderr)
        print(f"Requirement: {requirement[:150]}...", file=sys.stderr)
        print(f"PRD length: {len(prd)} chars", file=sys.stderr)
        print(f"Plan steps: {len(plan)}", file=sys.stderr)
        print(f"Architecture type: {type(architecture)}", file=sys.stderr)
        print(f"Architecture keys: {architecture.keys() if isinstance(architecture, dict) else 'N/A'}", file=sys.stderr)
        
        if isinstance(architecture, dict):
            db_schema = architecture.get('database_schema', {})
            tables = db_schema.get('tables', [])
            print(f"Database tables: {[t.get('name', 'unknown') for t in tables]}", file=sys.stderr)
            if tables:
                print(f"First table columns: {tables[0].get('columns', [])[:3]}", file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)
        
        await websocket.send_text(json.dumps({
            "event": "agent_start",
            "agent": "code_generator_agent",
            "message": "Generating application code based on PRD and plan..."
        }))
        
        # Initialize LLM for code generation
        llm = get_llm_for_user(
            user_email=None,  # Use default
            temperature=0.3,  # Lower temperature for more consistent code
            streaming=True
        )
        
        # Generate code using LLM
        files_dict = {}
        async for event in generate_code_from_plan(requirement, prd, plan, architecture, llm, uiux):
            if event["event"] == "generation_start":
                await websocket.send_text(json.dumps({
                    "event": "agent_progress",
                    "agent": "code_generator_agent",
                    "message": event["message"]
                }))
                
            elif event["event"] == "agent_progress":
                await websocket.send_text(json.dumps({
                    "event": "agent_progress",
                    "agent": "code_generator_agent",
                    "message": event["message"]
                }))
            
            elif event["event"] == "file_generated":
                # Clean up filename
                filename = event["file"]
                content = event["content"]
                
                # Write to disk immediately
                try:
                    write_project_file(project_name, filename, content)
                except Exception as e:
                    print(f"Error writing file {filename}: {e}", file=sys.stderr)

                # Send progress to UI
                await websocket.send_text(json.dumps({
                    "event": "agent_progress",
                    "agent": "code_generator_agent",
                    "message": f"Generated {filename}"
                }))
                
                # Update local dict for final stat
                files_dict[filename] = content
            
            elif event["event"] == "generation_complete":
                # Ensure we have the full list (possibly normalized: Vite plugin, backend .py imports)
                files_dict = event["data"]
                # Re-write post-processed files so normalized content is on disk
                for path, content in files_dict.items():
                    if path == "frontend/package.json" or (path.startswith("backend/") and path.endswith(".py")):
                        write_project_file(project_name, path, content)
                
                await websocket.send_text(json.dumps({
                    "event": "agent_progress",
                    "agent": "code_generator_agent",
                    "message": f"Generation complete. {len(files_dict)} files generated."
                }))
        
        # Files are already written during the stream.
        # We process the final dictionary just to be sure we return it correct.
        
        await websocket.send_text(json.dumps({
            "event": "agent_complete",
            "agent": "code_generator_agent",
            "data": {
                "files": list(files_dict.keys()),
                "count": len(files_dict)
            },
            "message": f"Generated {len(files_dict)} files successfully"
        }))
        
        return files_dict
        
    except WebSocketDisconnect:
        # Client disconnected, just stop
        print(f"Client disconnected during code generation", file=sys.stderr)
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({
                "event": "agent_error",
                "agent": "code_generator_agent",
                "error": str(e)
            }))
        except (WebSocketDisconnect, RuntimeError):
            # Connection likely closed
            pass
        raise


@router.websocket("/regenerate-code/{session_id}")
async def regenerate_code(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint specifically for code regeneration.
    
    Client sends:
    {
        "requirement": "...",
        "prd": "...",
        "plan": [...],
        "architecture": {...},
        "project_name": "..."
    }
    """
    await manager.connect(session_id, websocket)
    
    try:
        # Wait for request
        data = await websocket.receive_text()
        request_data = json.loads(data)
        
        requirement = request_data.get("requirement")
        prd = request_data.get("prd", "")
        plan = request_data.get("plan", [])
        architecture = request_data.get("architecture", {})
        project_name = request_data.get("project_name")
        uiux = request_data.get("uiux", "")
        
        if not requirement or not project_name:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": "Missing requirement or project_name"
            }))
            return
        
        await websocket.send_text(json.dumps({
            "event": "generation_start",
            "message": "Starting code regeneration..."
        }))
        
        # Log architecture for debugging
        import sys
        print("=" * 80, file=sys.stderr)
        print("CODE REGENERATION DEBUG:", file=sys.stderr)
        print(f"Requirement: {requirement[:100]}...", file=sys.stderr)
        print(f"Architecture received:", file=sys.stderr)
        print(f"  - Backend: {architecture.get('backend_structure', {})}", file=sys.stderr)
        print(f"  - Database Schema: {architecture.get('database_schema', {})}", file=sys.stderr)
        
        db_tables = architecture.get('database_schema', {}).get('tables', [])
        print(f"  - Tables found: {[t.get('name') for t in db_tables]}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        
        # Generate code
        files = await execute_code_generator_agent(
            websocket,
            requirement,
            prd,
            plan,
            architecture,
            project_name,
            uiux
        )

        # Store multi-agent output and generated code JSON in DB (same as documents)
        try:
            db.create_or_update_codegen_session(
                session_id=session_id,
                project_name=project_name,
                requirement=requirement,
                prd=prd,
                plan=plan,
                architecture=architecture,
                generated_code_json=files,
            )
        except Exception as store_err:
            pass  # non-fatal

        await websocket.send_text(json.dumps({
            "event": "regeneration_complete",
            "message": "Code regeneration completed",
            "data": {
                "project_name": project_name,
                "files_count": len(files),
                "files": list(files.keys())
            }
        }))
        
        websocket.close()
        
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": str(e)
            }))
        except:
            pass
        manager.disconnect(session_id)


@router.websocket("/execute/{session_id}")
async def execute_agents(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for executing agents with real-time updates.
    
    Client sends:
    {
        "requirement": "...",
        "plan": [...],
        "project_name": "..."
    }
    
    Server sends events:
    - {"event": "agent_start", "agent": "...", "message": "..."}
    - {"event": "agent_progress", "agent": "...", "message": "..."}
    - {"event": "agent_complete", "agent": "...", "data": {...}}
    - {"event": "agent_error", "agent": "...", "error": "..."}
    - {"event": "all_complete", "message": "..."}
    """
    await manager.connect(session_id, websocket)
    
    try:
        # Wait for initial request
        data = await websocket.receive_text()
        request_data = json.loads(data)
        
        requirement = request_data.get("requirement")
        plan_data = request_data.get("plan", [])
        prd_text = request_data.get("prd", "")  # Get PRD from request
        project_name = request_data.get("project_name")
        uiux_text = request_data.get("uiux", "")
        
        if not requirement or not project_name:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": "Missing requirement or project_name"
            }))
            return
        
        await websocket.send_text(json.dumps({
            "event": "execution_start",
            "message": "Starting agent execution pipeline..."
        }))
        
        # Initialize LLM for agents (from llm_config; no user_email in websocket)
        llm = get_llm_for_user(None, temperature=0.7)
        
        # Execute agents in sequence
        # 1. Requirement Agent
        clarified_requirement = await execute_requirement_agent(websocket, requirement)
        
        # 2. Architecture Agent (dynamic, streaming)
        architecture = await execute_architecture_agent(
            websocket, 
            requirement,
            prd_text,
            plan_data,
            llm,
            uiux_text
        )
        
        # Convert architecture to dict
        architecture_dict = {
            "backend_structure": architecture.backend_structure,
            "frontend_structure": architecture.frontend_structure,
            "database_schema": architecture.database_schema,
            "deployment": architecture.deployment,
            "rationale": architecture.rationale,
            "project_structure": architecture.project_structure
        }

        # Save artifacts to DB so they persist
        try:
            db.create_or_update_codegen_session(
                session_id=session_id,
                project_name=project_name,
                requirement=requirement,
                prd=prd_text,
                plan=plan_data,
                architecture=architecture_dict,
                generated_code_json=None
            )
        except Exception as e:
            print(f"Error saving codegen session: {e}", file=sys.stderr)

        # All done
        await websocket.send_text(json.dumps({
            "event": "all_complete",
            "message": "All agents completed successfully",
            "data": {
                "project_name": project_name,
                "files_count": 0
            }
        }))
        
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": str(e)
            }))
        except:
            pass
        manager.disconnect(session_id)
