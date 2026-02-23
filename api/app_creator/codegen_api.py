"""
Code Generation API - Handles dedicated code generation WebSocket
"""
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict
import json

from api.app_creator.agent_api import execute_code_generator_agent
from app_builder.agents.validation_agent import validate_and_fix_code
from app_builder.services.file_writer import write_project_file
from app_builder_db import get_app_builder_db

router = APIRouter(prefix="/api/codegen", tags=["Code Generation"])

db = get_app_builder_db()


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]


manager = ConnectionManager()


@router.websocket("/execute/{session_id}")
async def execute_code_generation(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint dedicated to code generation.

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
        data = await websocket.receive_text()
        request_data = json.loads(data)

        requirement = request_data.get("requirement")
        prd = request_data.get("prd", "") or ""
        plan = request_data.get("plan", [])
        architecture = request_data.get("architecture", {})
        project_name = request_data.get("project_name")
        uiux = request_data.get("uiux", "") or ""
        app_id = request_data.get("app_id")
        user_email = request_data.get("user_email") or ""

        # When PRD/UIUX not in request, load from DB
        if (not prd or not uiux) and (app_id or project_name):
            try:
                app_from_db = None
                if app_id:
                    app_from_db = db.get_app_builder_app(app_id)
                if not app_from_db and project_name:
                    app_from_db = db.get_app_by_project_name(project_name)
                if app_from_db:
                    if not prd:
                        prd = app_from_db.get("prd") or ""
                    if not uiux:
                        uiux = app_from_db.get("generated_uiux") or ""
                    if not plan and app_from_db.get("plan"):
                        plan = app_from_db.get("plan", [])
            except Exception as e:
                print(f"[codegen_api] DB fallback for PRD/UIUX: {e}", file=sys.stderr)

        if not requirement or not project_name or not architecture:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": "Missing requirement, architecture, or project_name"
            }))
            return

        await websocket.send_text(json.dumps({
            "event": "generation_start",
            "message": "Starting code generation..."
        }))

        files = await execute_code_generator_agent(
            websocket,
            requirement,
            prd,
            plan,
            architecture,
            project_name,
            uiux
        )

        # Execute Validation Agent
        await websocket.send_text(json.dumps({
            "event": "agent_start",
            "agent": "validation_agent",
            "message": "Validating generated code and checking dependencies..."
        }))
        
        try:
            files = validate_and_fix_code(files, architecture)
            
            # Re-write files if validation modified them (e.g. package.json/requirements.txt)
            # validate_and_fix_code modifies 'files' dict in place or returns it. 
            # We should save specific files that might have changed.
            # Ideally we save all or just check what changed.
            # For simplicity, let's re-write package.json and requirements.txt if they exist
            if "frontend/package.json" in files:
                write_project_file(project_name, "frontend/package.json", files["frontend/package.json"])
            if "backend/requirements.txt" in files:
                write_project_file(project_name, "backend/requirements.txt", files["backend/requirements.txt"])
                
            await websocket.send_text(json.dumps({
                "event": "agent_complete",
                "agent": "validation_agent",
                "message": "Validation complete. Dependencies verified."
            }))
        except Exception as val_err:
             await websocket.send_text(json.dumps({
                "event": "agent_error",
                "agent": "validation_agent",
                "message": f"Validation warning: {val_err}"
            }))
             # Continue even if validation fails, don't crash flow just for this


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
            
            # Also update the main app record if app_id and user_email available
            if app_id and user_email:
                db.update_app_builder_app(
                    app_id=app_id,
                    user_email=user_email,
                    generated_code_json=files
                )
        except Exception as store_err:
            try:
                await websocket.send_text(json.dumps({
                    "event": "warning",
                    "message": f"Code generation completed but DB store failed: {store_err}"
                }))
            except Exception:
                pass

        await websocket.send_text(json.dumps({
            "event": "codegen_complete",
            "message": "Code generation completed",
            "data": {
                "project_name": project_name,
                "files_count": len(files),
                "files": list(files.keys())
            }
        }))

        await websocket.close()

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": str(e)
            }))
        except Exception:
            pass
        manager.disconnect(session_id)
