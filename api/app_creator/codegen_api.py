"""
Code Generation API - Handles dedicated code generation WebSocket
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict
import json
import os
import sys

from api.app_creator.agent_api import execute_code_generator_agent
from api.app_creator.project_access import assert_project_access, assert_app_id_access
from app_builder.agents.validation_agent import validate_and_fix_code
from app_builder.services.file_writer import file_writer
from app_builder.services.project_config import build_project_config, persist_project_config
from app_builder.services.db_init import ensure_project_db_initialized
from app_builder.services.functionality_validator import enrich_project_config
from app_builder.services.code_post_process import should_use_template, post_process_generated_files, ensure_valid_codegen_output
from app_builder.services.scaffold_service import normalize_architecture_for_codegen
from app_builder.services.template_service import detect_template
from app_builder.schemas.files import GeneratedFiles
from db.app_builder import get_app_builder_db
from api.auth.ws_auth import authenticate_websocket
from api.auth.request_auth import user_email_from
from api.app_creator.pipeline_helpers import touch_job
from api.app_creator.build_verify_service import verify_build_and_fix

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


def _persist_project_runtime(project_name: str, files: dict, architecture: dict, requirement: str, prd: str, uiux: str):
    """Write project_config.json and initialize per-project SQLite."""
    import json as _json

    template_name = None
    if should_use_template(requirement, prd, architecture):
        template_name = "base-vite-fastapi"
    tables = (architecture or {}).get("database_schema", {}).get("tables") or []
    entities = []
    for t in tables:
        if not isinstance(t, dict):
            continue
        name = t.get("name") or "Entity"
        entities.append(
            {
                "name": name,
                "table_name": t.get("table_name") or str(name).lower() + "s",
                "fields": t.get("columns") or t.get("fields") or [],
            }
        )
    structured = {
        "project_name": project_name,
        "description": requirement,
        "prd": prd,
        "uiux": uiux,
        "entities": entities,
    }
    schema_payload = _json.dumps(tables) if tables else _json.dumps((architecture or {}).get("database_schema") or {})
    config = build_project_config(
        project_name=project_name,
        structured_requirement=structured,
        architecture=architecture or {},
        api_contract=(architecture or {}).get("api_contract") or {},
        db_schema={"schema": schema_payload},
        template_name=template_name,
    )
    from app_builder.services.app_spec_service import build_app_spec
    app_spec = build_app_spec(requirement, architecture, prd, uiux)
    config = enrich_project_config(config, architecture or {}, app_spec)
    persist_project_config(project_name, config)
    ensure_project_db_initialized(project_name, config)


@router.websocket("/execute/{session_id}")
async def execute_code_generation(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint dedicated to code generation.
    """
    await manager.connect(session_id, websocket)

    user_email = None
    uid = None
    app_id = None

    try:
        data = await websocket.receive_text()
        request_data = json.loads(data)
        current, request_data = await authenticate_websocket(websocket, first_message=request_data)
        user_email = user_email_from(current)
        uid = current.id if current.id else None

        requirement = request_data.get("requirement")
        prd = request_data.get("prd", "") or ""
        plan = request_data.get("plan", [])
        architecture = request_data.get("architecture", {})
        project_name = request_data.get("project_name")
        uiux = request_data.get("uiux", "") or ""
        app_id = request_data.get("app_id")

        if not project_name:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": "Missing project_name",
            }))
            return

        try:
            assert_project_access(project_name, current)
        except HTTPException as exc:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": exc.detail,
            }))
            return

        if app_id:
            try:
                assert_app_id_access(app_id, project_name, current)
            except HTTPException as exc:
                await websocket.send_text(json.dumps({
                    "event": "error",
                    "message": exc.detail,
                }))
                return

        if (not prd or not uiux) and (app_id or project_name):
            try:
                app_from_db = None
                if app_id:
                    app_from_db = db.get_app_builder_app(app_id, user_email=user_email, user_id=uid)
                if not app_from_db and project_name:
                    app_from_db = db.get_app_by_project_name(
                        project_name, user_id=uid, user_email=user_email
                    )
                if app_from_db:
                    if not prd:
                        prd = app_from_db.get("prd") or ""
                    if not uiux:
                        uiux = app_from_db.get("generated_uiux") or ""
                    if not plan and app_from_db.get("plan"):
                        plan = app_from_db.get("plan", [])
                    if not architecture and app_from_db.get("architecture"):
                        architecture = app_from_db.get("architecture") or {}
                    architecture = normalize_architecture_for_codegen(
                        architecture or {},
                        api_contract=app_from_db.get("api_contract"),
                    )
            except Exception as e:
                print(f"[codegen_api] DB fallback for PRD/UIUX: {e}", file=sys.stderr)

        if not requirement or not project_name or not architecture:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": "Missing requirement, architecture, or project_name"
            }))
            return

        architecture = normalize_architecture_for_codegen(architecture)

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
                job_type="codegen",
                status="running",
                step="code_generator_agent",
            )

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

        if not files:
            err = "Code generation produced no files"
            if app_id and user_email:
                db.update_app_builder_app(
                    app_id=app_id,
                    user_email=user_email,
                    user_id=uid,
                    pipeline_status="CODEGEN_FAILED",
                    pipeline_error=err,
                )
            touch_job(
                session_id,
                app_id=app_id,
                job_type="codegen",
                status="failed",
                step="code_generator_agent",
                error=err,
                finished=True,
            )
            await websocket.send_text(json.dumps({"event": "error", "message": err}))
            return

        await websocket.send_text(json.dumps({
            "event": "agent_start",
            "agent": "validation_agent",
            "message": "Validating generated code and checking dependencies..."
        }))
        touch_job(
            session_id,
            app_id=app_id,
            job_type="codegen",
            status="running",
            step="validation_agent",
        )

        template_name = "base-vite-fastapi"

        try:
            files = validate_and_fix_code(files, architecture, template_name=template_name)
            from app_builder.services.app_spec_service import build_app_spec
            app_spec = build_app_spec(requirement, architecture, prd, uiux)
            files = post_process_generated_files(
                files, architecture, template_name=template_name, uiux=uiux,
                requirement=requirement, prd=prd, app_spec=app_spec,
            )
            files, validation_errors = ensure_valid_codegen_output(
                files, architecture, uiux=uiux,
                requirement=requirement, prd=prd, app_spec=app_spec,
            )
            if validation_errors:
                raise ValueError("; ".join(validation_errors[:5]))

            await websocket.send_text(json.dumps({
                "event": "agent_start",
                "agent": "functionality_validator_agent",
                "message": "Checking API calls and runtime CRUD functionality...",
            }))
            touch_job(
                session_id,
                app_id=app_id,
                job_type="codegen",
                status="running",
                step="functionality_validator_agent",
            )

            from app_builder.services.functionality_validator import run_functionality_pipeline
            from app_builder.services.project_config import get_project_config

            _persist_project_runtime(project_name, files, architecture, requirement, prd, uiux)
            config = get_project_config(project_name) or {}
            files, func_ok, func_msgs, func_err = run_functionality_pipeline(
                files, project_name, architecture, app_spec=app_spec, config=config,
            )
            for msg in func_msgs:
                await websocket.send_text(json.dumps({
                    "event": "agent_progress",
                    "agent": "functionality_validator_agent",
                    "message": msg,
                }))
            if not func_ok:
                from app_builder.services.app_generators import apply_deterministic_fallback
                logger_msg = f"Functionality check failed ({func_err}) — applying CRUD fallback"
                await websocket.send_text(json.dumps({
                    "event": "agent_progress",
                    "agent": "functionality_validator_agent",
                    "message": logger_msg,
                }))
                files = apply_deterministic_fallback(files, app_spec, uiux=uiux, prd=prd)
                files = post_process_generated_files(
                    files, architecture, template_name=template_name, uiux=uiux,
                    requirement=requirement, prd=prd, app_spec=app_spec,
                )
                files, validation_errors = ensure_valid_codegen_output(
                    files, architecture, uiux=uiux,
                    requirement=requirement, prd=prd, app_spec=app_spec,
                )
                if validation_errors:
                    raise ValueError(f"Functionality fallback failed: {'; '.join(validation_errors[:5])}")
                _persist_project_runtime(project_name, files, architecture, requirement, prd, uiux)
                config = get_project_config(project_name) or {}
                files, func_ok, func_msgs, func_err = run_functionality_pipeline(
                    files, project_name, architecture, app_spec=app_spec, config=config,
                )
                if not func_ok:
                    raise ValueError(f"Runtime CRUD verification failed: {func_err}")

            await websocket.send_text(json.dumps({
                "event": "agent_complete",
                "agent": "functionality_validator_agent",
                "message": "Functionality verified — API calls and CRUD smoke test passed.",
            }))

            file_writer(project_name, GeneratedFiles(files=files))
            _persist_project_runtime(project_name, files, architecture, requirement, prd, uiux)
            await websocket.send_text(json.dumps({
                "event": "agent_complete",
                "agent": "validation_agent",
                "message": "Validation complete. Dependencies verified."
            }))
        except Exception as val_err:
            err = f"Validation failed: {val_err}"
            print(f"[codegen_api] {err}", file=sys.stderr)
            if app_id and user_email:
                db.update_app_builder_app(
                    app_id=app_id,
                    user_email=user_email,
                    user_id=uid,
                    pipeline_status="CODEGEN_FAILED",
                    pipeline_error=err,
                )
            touch_job(
                session_id,
                app_id=app_id,
                job_type="codegen",
                status="failed",
                step="validation_agent",
                error=err,
                finished=True,
            )
            await websocket.send_text(json.dumps({
                "event": "agent_error",
                "agent": "validation_agent",
                "message": err,
            }))
            await websocket.send_text(json.dumps({"event": "error", "message": err}))
            return

        async def _emit_build_event(payload: dict):
            await websocket.send_text(json.dumps(payload))

        await websocket.send_text(json.dumps({
            "event": "agent_start",
            "agent": "build_verify_agent",
            "message": "Verifying npm build (install + build + auto-fix)...",
        }))
        touch_job(
            session_id,
            app_id=app_id,
            job_type="codegen",
            status="running",
            step="build_verify",
        )

        verify_build = os.environ.get("CODEGEN_VERIFY_BUILD", "true").lower() not in ("0", "false", "no")
        if verify_build:
            files, build_ok, build_log = await verify_build_and_fix(
                project_name,
                files,
                user_email,
                on_event=_emit_build_event,
                max_attempts=int(os.environ.get("CODEGEN_BUILD_FIX_ATTEMPTS", "3")),
            )
            file_writer(project_name, GeneratedFiles(files=files))
            if not build_ok:
                err = f"Build verification failed after auto-fix attempts: {(build_log or '')[-800:]}"
                if app_id and user_email:
                    db.update_app_builder_app(
                        app_id=app_id,
                        user_email=user_email,
                        user_id=uid,
                        pipeline_status="CODEGEN_FAILED",
                        pipeline_error=err,
                        build_status="BUILD_FAILED",
                        build_error=err,
                        build_log=build_log[-8000:] if build_log else err,
                    )
                touch_job(
                    session_id,
                    app_id=app_id,
                    job_type="codegen",
                    status="failed",
                    step="build_verify",
                    error=err,
                    finished=True,
                )
                await websocket.send_text(json.dumps({
                    "event": "agent_error",
                    "agent": "build_verify_agent",
                    "message": err,
                }))
                await websocket.send_text(json.dumps({"event": "error", "message": err}))
                return

            await websocket.send_text(json.dumps({
                "event": "agent_complete",
                "agent": "build_verify_agent",
                "message": "Build verified — npm run build succeeded.",
            }))
        else:
            await websocket.send_text(json.dumps({
                "event": "agent_complete",
                "agent": "build_verify_agent",
                "message": "Build verify skipped (CODEGEN_VERIFY_BUILD=false).",
            }))

        store_err = None
        try:
            db.create_or_update_codegen_session(
                session_id=session_id,
                project_name=project_name,
                requirement=requirement,
                prd=prd,
                plan=plan,
                architecture=architecture,
                generated_code_json=files,
                app_id=app_id,
            )
        except Exception as exc:
            store_err = str(exc)
            print(f"[codegen_api] codegen session store failed: {store_err}", file=sys.stderr)

        if app_id and user_email:
            try:
                db.update_app_builder_app(
                    app_id=app_id,
                    user_email=user_email,
                    user_id=uid,
                    generated_code_json=files,
                    pipeline_status="CODEGEN_COMPLETE",
                    pipeline_error=None,
                    build_status="BUILD_SUCCESS",
                )
            except Exception as exc:
                store_err = store_err or str(exc)
                print(f"[codegen_api] app update failed: {exc}", file=sys.stderr)

        touch_job(
            session_id,
            app_id=app_id,
            job_type="codegen",
            status="complete",
            step="codegen_complete",
            finished=True,
        )

        if store_err:
            await websocket.send_text(json.dumps({
                "event": "warning",
                "message": f"Code generation completed but DB store failed: {store_err}"
            }))

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
        err = (
            "Code generation connection closed (server may have reloaded). "
            "Restart the API with: python run_server.py"
        )
        if app_id and user_email:
            try:
                db.update_app_builder_app(
                    app_id=app_id,
                    user_email=user_email,
                    user_id=uid,
                    pipeline_status="CODEGEN_FAILED",
                    pipeline_error=err,
                )
            except Exception:
                pass
        touch_job(
            session_id,
            app_id=app_id,
            job_type="codegen",
            status="failed",
            step="code_generator_agent",
            error=err,
            finished=True,
        )
    except Exception as e:
        if app_id and user_email:
            try:
                db.update_app_builder_app(
                    app_id=app_id,
                    user_email=user_email,
                    user_id=uid,
                    pipeline_status="CODEGEN_FAILED",
                    pipeline_error=str(e),
                )
            except Exception:
                pass
        try:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": str(e)
            }))
        except Exception:
            pass
        manager.disconnect(session_id)
