"""
Agent API - Handles agent execution with WebSocket for real-time updates
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import asyncio
import os
import sys

from app_builder.schemas.requirements import UserRequirement
from app_builder.schemas.plan import ProjectPlan
from app_builder.agents.requirement_agent import requirement_agent
from app_builder.agents.architecture_agent import architecture_agent
from app_builder.agents.dynamic_code_generator import generate_code_from_plan
from app_builder.services.file_writer import file_writer, write_project_file
from app_builder.schemas.files import GeneratedFiles
from llm_helper import get_llm_for_user
from db.app_builder import get_app_builder_db
from api.auth.ws_auth import authenticate_websocket
from api.auth.request_auth import user_email_from
from api.app_creator.pipeline_helpers import touch_job
from api.app_creator.project_access import assert_project_access, assert_app_id_access

router = APIRouter(prefix="/api/agents", tags=["Agents"])

db = get_app_builder_db()


class AgentExecutionRequest(BaseModel):
    requirement: str
    plan: List[Dict[str, Any]]
    project_name: str


from app_builder.agents.code_update_agent import update_code_from_chat
from app_builder.services.runtime_paths import get_projects_dir, resolve_project_root



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


async def execute_requirement_agent(
    websocket: WebSocket,
    requirement: str,
    prd: str = "",
    uiux: str = "",
    llm=None
):
    """Execute requirement agent: fetches PRD + UI/UX, produces minimal end-to-end consolidated output."""
    try:
        await websocket.send_text(json.dumps({
            "event": "agent_start",
            "agent": "requirement_agent",
            "message": "Fetching details from PRD and UI/UX design, consolidating requirements..."
        }))
        
        # Use LLM when PRD/UIUX provided for consolidation
        if prd or uiux:
            await asyncio.sleep(0.3)  # Brief delay for UX
            clarified = await requirement_agent(
                UserRequirement(description=requirement),
                prd=prd or "",
                uiux=uiux or "",
                llm=llm
            )
        else:
            await asyncio.sleep(0.5)
            clarified = await requirement_agent(UserRequirement(description=requirement))
        
        await websocket.send_text(json.dumps({
            "event": "agent_complete",
            "agent": "requirement_agent",
            "data": clarified,
            "message": "Requirements consolidated from PRD and UI/UX successfully"
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
    uiux: str = "",
    user_email: str | None = None,
    model_name: str | None = None,
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
            user_email=user_email,
            model_name=model_name,
            temperature=0.3,
            streaming=True
        )
        
        # Generate code using LLM
        files_dict = {}
        codegen_error = None
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
                filename = event["file"]
                content = event["content"]
                await websocket.send_text(json.dumps({
                    "event": "agent_progress",
                    "agent": "code_generator_agent",
                    "message": f"Generated {filename}"
                }))
                files_dict[filename] = content

            elif event["event"] == "generation_complete":
                files_dict = event["data"]
                await websocket.send_text(json.dumps({
                    "event": "agent_progress",
                    "agent": "code_generator_agent",
                    "message": f"Generation complete. {len(files_dict)} files generated."
                }))

            elif event["event"] == "agent_error":
                codegen_error = event.get("message") or event.get("error") or "Code generation failed"
                await websocket.send_text(json.dumps({
                    "event": "agent_error",
                    "agent": "code_generator_agent",
                    "message": codegen_error,
                }))

        if codegen_error and not files_dict:
            raise RuntimeError(codegen_error)

        if not files_dict:
            raise RuntimeError("Code generation produced no files")

        count = len(files_dict)
        await websocket.send_text(json.dumps({
            "event": "agent_complete",
            "agent": "code_generator_agent",
            "data": {
                "files": list(files_dict.keys()),
                "count": count
            },
            "message": f"Generated {count} files successfully"
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


@router.websocket("/execute/{session_id}")
async def execute_agents(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for executing the full agent pipeline using LangGraph.
    Streams events: agent_start, agent_progress, agent_complete.
    """
    from app_builder.graph.builder_graph import app_builder_graph
    
    await manager.connect(session_id, websocket)
    
    try:
        # Wait for initial request
        data = await websocket.receive_text()
        request_data = json.loads(data)
        current, request_data = await authenticate_websocket(websocket, first_message=request_data)
        user_email = user_email_from(current)

        requirement = request_data.get("requirement")
        project_name = request_data.get("project_name")
        app_id = request_data.get("app_id")
        prd = request_data.get("prd", "") or ""
        uiux = request_data.get("uiux", "") or ""
        prebuilt_architecture = request_data.get("architecture") or {}

        if not requirement or not project_name:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": "Missing requirement or project_name"
            }))
            return

        try:
            assert_project_access(project_name, current)
        except HTTPException as exc:
            await websocket.send_text(json.dumps({"event": "error", "message": exc.detail}))
            return

        if app_id:
            try:
                assert_app_id_access(app_id, project_name, current)
            except HTTPException as exc:
                await websocket.send_text(json.dumps({"event": "error", "message": exc.detail}))
                return

        await websocket.send_text(json.dumps({
            "event": "execution_start",
            "message": "Starting new agent execution pipeline..."
        }))

        uid = current.id if current.id else None
        if app_id and user_email:
            db.update_app_builder_app(
                app_id=app_id,
                user_email=user_email,
                user_id=uid,
                pipeline_status="CODEGEN_RUNNING",
                pipeline_error=None,
            )
        touch_job(
            session_id,
            app_id=app_id,
            job_type="agents",
            status="running",
            step="structuring_agent",
        )

        # Initialize State for LangGraph
        initial_state = {
            "user_requirement": UserRequirement(description=requirement),
            "project_name": project_name,
            "saved_prd": prd,
            "saved_uiux": uiux,
            "prebuilt_architecture": prebuilt_architecture if isinstance(prebuilt_architecture, dict) else {},
            "structured_requirement": {},
            "architecture": {},
            "api_contract": {},
            "db_schema": {},
            "generated_files": {},
            "validation_results": {},
            "error": ""
        }

        # Node name to human-friendly display name mapping (matches AgentsView.js)
        node_to_agent = {
            "structuring_step": "structuring_agent",
            "architecture_step": "architecture_agent",
            "contract_step": "contract_agent",
            "schema_step": "schema_agent",
            "coding_step": "coding_agent",
            "validation_step": "validation_agent"
        }

        final_state = initial_state
        agents_state = {}
        pipeline_failed = False
        
        await websocket.send_text(json.dumps({
            "event": "agent_start",
            "agent": "structuring_agent",
            "message": "Analyzing and structuring your requirement..."
        }))

        # Stream from LangGraph
        async for output in app_builder_graph.astream(initial_state):
            if pipeline_failed:
                break
            for node_name, result in output.items():
                agent_id = node_to_agent.get(node_name, node_name)
                
                if result and result.get("error"):
                    pipeline_failed = True
                    final_state["error"] = result["error"]
                    await websocket.send_text(json.dumps({
                        "event": "agent_error",
                        "agent": agent_id,
                        "error": result["error"],
                        "project_name": project_name
                    }))
                    agents_state[agent_id] = {"status": "error", "error": result["error"]}
                    break

                # Update project_name if structuring_step returned one
                if result and result.get("project_name"):
                    project_name = result["project_name"]

                # Prepare data for completion
                completion_data = result
                
                # Disk persistence for coding and validation steps
                if node_name in ["coding_step", "validation_step"]:
                    files_dict = result.get("generated_files", {})
                    if files_dict:
                        for rel_path, content in files_dict.items():
                            try:
                                wrote = write_project_file(project_name, rel_path, content)
                                if wrote and node_name == "coding_step":
                                    await websocket.send_text(json.dumps({
                                        "event": "agent_progress",
                                        "agent": agent_id,
                                        "message": f"Generated and saved: {rel_path}",
                                        "project_name": project_name
                                    }))
                            except Exception as write_err:
                                print(f"[agent_api] Error writing file {rel_path}: {write_err}")
                        
                        # Add updated_files list so frontend can refresh its cache
                        completion_data = {**result, "updated_files": list(files_dict.keys())}
                
                # Special handling for schema_step to ensure it's not double-stringified in the UI
                if node_name == "schema_step" and result.get("db_schema"):
                    # If db_schema is already a dict, keep it as is. 
                    # Our agent returns a string for 'schema'.
                    pass

                # Send completion event for the node
                await websocket.send_text(json.dumps({
                    "event": "agent_complete",
                    "agent": agent_id,
                    "message": f"Completed {agent_id.replace('_', ' ')}",
                    "data": completion_data,
                    "project_name": project_name
                }))
                
                # Accumulate for agents_state (backend fallback save)
                agents_state[agent_id] = {
                    "status": "complete",
                    "message": f"Completed {agent_id.replace('_', ' ')}",
                    "data": completion_data,
                    "progress": [{"type": "complete", "text": f"Completed {agent_id}", "timestamp": None}],
                    "completed_at": None
                }

                touch_job(
                    session_id,
                    app_id=app_id,
                    job_type="agents",
                    status="running",
                    step=agent_id,
                )
                
                # Proactively send start for the NEXT node in flow
                next_agent_map = {
                    "structuring_step": ("architecture_agent", "Designing the system architecture..."),
                    "architecture_step": ("contract_agent", "Generating API contracts..."),
                    "contract_step": ("schema_agent", "Defining database schema..."),
                    "schema_step": ("coding_agent", "Generating backend and frontend code (this may take a minute)..."),
                    "coding_step": ("validation_agent", "Validating and fixing generated code...")
                }
                if node_name in next_agent_map:
                    next_id, next_msg = next_agent_map[node_name]
                    await websocket.send_text(json.dumps({
                        "event": "agent_start",
                        "agent": next_id,
                        "message": next_msg,
                        "project_name": project_name
                    }))

                # Special events for specific updates
                if node_name == "architecture_step":
                    await websocket.send_text(json.dumps({
                        "event": "architecture_updated",
                        "data": result.get("architecture"),
                        "project_name": project_name
                    }))
                
                # Cumulative state update
                final_state.update(result)

        if pipeline_failed or final_state.get("error"):
            err = final_state.get("error") or "Pipeline failed"
            if app_id and user_email:
                try:
                    db.update_app_builder_app(
                        app_id=app_id,
                        user_email=user_email,
                        user_id=uid,
                        pipeline_status="CODEGEN_FAILED",
                        pipeline_error=str(err),
                        agents_state=agents_state,
                    )
                except Exception:
                    pass
            touch_job(
                session_id,
                app_id=app_id,
                job_type="agents",
                status="failed",
                error=str(err),
                finished=True,
            )
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": err,
                "project_name": project_name,
            }))
            return

        # Final persistence to Postgres (builder_apps)
        try:
            if not final_state.get("error"):
                structured_prd = final_state.get("structured_requirement")
                arch = final_state.get("architecture")
                api_contract_val = final_state.get("api_contract")
                db_schema_val = final_state.get("db_schema")
                generated_files = final_state.get("generated_files", {})

                def _text_field(val):
                    if val is None:
                        return None
                    if isinstance(val, str):
                        return val
                    return json.dumps(val)

                db.create_or_update_codegen_session(
                    session_id=session_id,
                    project_name=project_name,
                    requirement=requirement,
                    prd=_text_field(structured_prd),
                    architecture=arch if isinstance(arch, (dict, list)) else None,
                    api_contract=_text_field(api_contract_val),
                    db_schema=_text_field(db_schema_val),
                    generated_files=generated_files if isinstance(generated_files, dict) else None,
                    app_id=app_id
                )
                
                if app_id and user_email:
                    db.update_app_builder_app(
                        app_id=app_id,
                        user_email=user_email,
                        user_id=uid,
                        project_name=project_name,
                        architecture=arch if isinstance(arch, dict) else None,
                        generated_code_json=generated_files if isinstance(generated_files, dict) else None,
                        prd=_text_field(structured_prd),
                        agents_state=agents_state,
                        pipeline_status="CODEGEN_COMPLETE",
                        pipeline_error=None,
                    )
                    print(f"[agent_api] Saved final app state (incl. agents_state) to Postgres for app {app_id}")
                touch_job(
                    session_id,
                    app_id=app_id,
                    job_type="agents",
                    status="complete",
                    step="validation_agent",
                    finished=True,
                )
        except Exception as db_err:
            print(f"[agent_api] Final DB save failed: {db_err}", file=sys.stderr)

        generated = final_state.get("generated_files") or {}
        if not generated:
            err = final_state.get("error") or "Code generation produced no files"
            if app_id and user_email:
                try:
                    db.update_app_builder_app(
                        app_id=app_id,
                        user_email=user_email,
                        user_id=uid,
                        pipeline_status="CODEGEN_FAILED",
                        pipeline_error=str(err),
                    )
                except Exception:
                    pass
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": str(err),
                "project_name": project_name,
            }))
            return

        await websocket.send_text(json.dumps({
            "event": "all_complete",
            "message": "Full generation pipeline completed successfully",
            "data": {
                "project_name": project_name,
                "files_count": len(final_state.get("generated_files", {}))
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


@router.websocket("/update-code-ws/{session_id}")
async def update_code_ws(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for updating existing code with real-time feedback."""
    await manager.connect(session_id, websocket)

    try:
        data = await websocket.receive_text()
        request_data = json.loads(data)
        current, request_data = await authenticate_websocket(websocket, first_message=request_data)
        user_email = user_email_from(current)

        user_request = request_data.get("user_request")
        project_name = request_data.get("project_name")
        app_id = request_data.get("app_id")

        if not user_request or not project_name:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": "Missing user_request or project_name"
            }))
            return

        try:
            assert_project_access(project_name, current)
            if app_id:
                assert_app_id_access(app_id, project_name, current)
        except HTTPException as exc:
            await websocket.send_text(json.dumps({"event": "error", "message": exc.detail}))
            return

        project_root = resolve_project_root(project_name)
        if not os.path.exists(project_root):
             await websocket.send_text(json.dumps({
                "event": "error",
                "message": "Project not found"
            }))
             return

        prd_text = ""
        original_requirement = ""
        architecture = {}
        app_record = None
        try:
            if app_id:
                app_record = db.get_app_builder_app(app_id, user_email=user_email, user_id=current.id or None)
            if not app_record:
                app_record = db.get_app_by_project_name(
                    project_name,
                    user_id=current.id or None,
                    user_email=user_email,
                )
            if app_record:
                prd_text = app_record.get("prd") or ""
                original_requirement = app_record.get("prompt") or ""
                architecture = app_record.get("architecture") or {}
        except Exception as db_err:
            print(f"[update-code-ws] DB fetch warning: {db_err}", file=sys.stderr)

        result = await update_code_from_chat(
            project_name=project_name,
            project_root=project_root,
            user_request=user_request,
            prd=prd_text,
            original_requirement=original_requirement,
            architecture=architecture,
            websocket=websocket
        )

        if result.get("status") == "success":
            try:
                updated_code = result.get("updated_code_dict") or {}
                updated_prd = result.get("updated_prd")
                updated_architecture = result.get("updated_architecture")
                
                existing_code = {}
                if app_record and app_record.get("generated_code_json"):
                    existing_code = app_record["generated_code_json"] or {}
                merged_code = {**existing_code, **updated_code}

                target_id = app_id or (app_record.get("id") if app_record else None)
                if target_id:
                    db.update_app_builder_app(
                        app_id=target_id,
                        user_email=app_record.get("user_email", "") if app_record else "",
                        generated_code_json=merged_code,
                        prd=updated_prd,
                        architecture=updated_architecture
                    )
            except Exception as save_err:
                print(f"[update-code-ws] DB save warning: {save_err}", file=sys.stderr)

        result.pop("updated_code_dict", None)

        await websocket.send_text(json.dumps({
            "event": "agent_complete",
            "agent": "update_code_agent",
            "data": result,
            "message": "Code update completed"
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
