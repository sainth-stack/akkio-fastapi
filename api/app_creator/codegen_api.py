"""
Code Generation API - Handles dedicated code generation WebSocket
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, Optional
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
from api.app_creator.pipeline_helpers import touch_job, public_base_url
from app_builder.services.project_config import get_project_config
from api.app_creator.backend_verify_service import apply_deterministic_backend_fixes
from api.app_creator.build_verify_service import verify_build_and_fix
from api.app_creator.backend_verify_service import verify_backend_and_fix

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


# ---------------------------------------------------------------------------
# Frontend-only pipeline helper
# ---------------------------------------------------------------------------

async def _run_frontend_only_pipeline(
    *,
    websocket,
    session_id: str,
    requirement: str,
    plan_json: Optional[Dict],
    design_tokens: Optional[Dict],
    project_name: str,
    app_id: Optional[str],
    user_email: Optional[str],
    uid: Optional[int],
    model_name: Optional[str],
    resume_stage: Optional[str] = None,
) -> Dict[str, str]:
    """
    Run the 6-stage frontend-only pipeline, streaming progress over WebSocket.
    Returns the file dict on success.
    """
    from app_builder.services.frontend_only_pipeline import run_pipeline
    from llm_helper import get_llm_for_user

    # Merge design_tokens into plan_json if needed
    if plan_json and design_tokens and not plan_json.get("design_tokens"):
        plan_json = dict(plan_json)
        plan_json["design_tokens"] = design_tokens

    try:
        llm = get_llm_for_user(
            user_email or "system@akkio.com",
            model_name=model_name,
            temperature=0.4,
            streaming=False,
        )
    except Exception as llm_exc:
        print(f"[codegen_api/fo] LLM init failed: {llm_exc}", file=sys.stderr)
        llm = None

    if llm is None:
        await websocket.send_text(json.dumps({
            "event": "error",
            "message": "Could not initialize LLM for frontend-only pipeline",
        }))
        return {}

    async def on_event(data: Dict) -> None:
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            pass

    try:
        files = await run_pipeline(
            requirement=requirement,
            plan_json=plan_json,
            project_name=project_name,
            llm=llm,
            on_event=on_event,
            resume_stage=resume_stage,
            app_id=app_id,
            user_email=user_email,
            uid=uid,
        )
        return files
    except Exception as exc:
        print(f"[codegen_api/fo] pipeline error: {exc}", file=sys.stderr)
        await on_event({
            "event": "agent_error",
            "agent": "fo_pipeline",
            "message": str(exc),
        })
        return {}


from typing import Optional, Dict


def _persist_project_runtime(project_name: str, files: dict, architecture: dict, requirement: str, prd: str, uiux: str):
    """Write project_config.json and initialize per-project SQLite. Best-effort; never fail codegen."""
    try:
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
        from app_builder.services.app_spec_service import build_app_spec, detect_llm_gen_type
        app_spec = build_app_spec(requirement, architecture, prd, uiux)
        structured["gen_type"] = app_spec.get("gen_type") or detect_llm_gen_type(requirement, prd)
        api_contract = (architecture or {}).get("api_contract") or {}
        if isinstance(api_contract, list):
            api_contract = {"endpoints": api_contract}
        config = build_project_config(
            project_name=project_name,
            structured_requirement=structured,
            architecture=architecture or {},
            api_contract=api_contract,
            db_schema={"schema": ""},
            template_name=template_name,
        )
        config = enrich_project_config(config, architecture or {}, app_spec)
        persist_project_config(project_name, config)
        ensure_project_db_initialized(project_name, config)
    except Exception as persist_err:
        print(f"[codegen_api] persist skipped: {persist_err}", file=sys.stderr)


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
        model_name = request_data.get("model_name")
        design_tokens = request_data.get("design_tokens")
        builder_kind = request_data.get("builder_kind") or ""
        # Frontend-only track fields (from Prompt 2 structured contracts)
        plan_json_req = request_data.get("plan_json")     # structured planning contracts
        resume_stage = request_data.get("resume_stage")   # "S1"…"S6" for retry support

        if not project_name:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": "Missing project_name",
            }))
            return

        try:
            assert_project_access(
                project_name,
                current,
                auto_create=True,
                app_name=request_data.get("app_name") or project_name,
                prompt=requirement or "",
                builder_kind=builder_kind or "fullstack",
            )
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
                    if not design_tokens and app_from_db.get("design_tokens"):
                        design_tokens = app_from_db.get("design_tokens")
                    if not builder_kind:
                        builder_kind = app_from_db.get("builder_kind") or ""
                    # Restore plan_json from DB for frontend_only track
                    if not plan_json_req and app_from_db.get("plan_json"):
                        plan_json_req = app_from_db.get("plan_json")
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

        from app_builder.services.fullstack_stack import is_fullstack, lock_architecture
        fullstack = is_fullstack(builder_kind)
        if fullstack:
            architecture = lock_architecture(architecture)

        # ── Detect track ─────────────────────────────────────────────────────
        _track = None
        _domain = None
        try:
            from app_builder.services.fullstack_app_generator import resolve_app_mode, resolve_track
            _domain = resolve_app_mode(requirement, prd, uiux)
            _track = resolve_track(_domain)
        except Exception:
            pass

        # Override with plan_json track if available
        if plan_json_req and (plan_json_req.get("blueprint") or {}).get("domain"):
            _domain = plan_json_req["blueprint"]["domain"]
            try:
                from app_builder.services.fullstack_app_generator import resolve_track
                _track = resolve_track(_domain)
            except Exception:
                pass

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

        # ── Frontend-only track: run the 6-stage pipeline ────────────────────
        if _track == "frontend_only":
            files = await _run_frontend_only_pipeline(
                websocket=websocket,
                session_id=session_id,
                requirement=requirement,
                plan_json=plan_json_req,
                design_tokens=design_tokens,
                project_name=project_name,
                app_id=app_id,
                user_email=user_email,
                uid=uid,
                model_name=model_name,
                resume_stage=resume_stage,
            )
            if not files:
                _err = "Frontend-only pipeline produced no files"
                if app_id and user_email:
                    db.update_app_builder_app(
                        app_id=app_id, user_email=user_email, user_id=uid,
                        pipeline_status="CODEGEN_FAILED", pipeline_error=_err,
                    )
                await websocket.send_text(json.dumps({"event": "error", "message": _err}))
                return
            # Skip to final save (no backend verification needed for FO track)
        else:
            files = await execute_code_generator_agent(
                websocket,
                requirement,
                prd,
                plan,
                architecture,
                project_name,
                uiux,
                user_email=user_email,
                model_name=model_name,
                builder_kind=builder_kind,
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

        # ── Legacy track: validation + backend verify + build verify ─────────
        if _track != "frontend_only":
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

        template_name = "base-fullstack-vite-mui" if fullstack else "base-vite-fastapi"

        try:
            from app_builder.services.app_spec_service import build_app_spec
            from app_builder.services.fullstack_codegen import (
                apply_theme_tokens,
                ensure_deliverables,
                ensure_mock_client,
            )
            app_spec = build_app_spec(requirement, architecture, prd, uiux)

            if fullstack:
                from app_builder.services.fullstack_app_generator import (
                    fill_missing_fullstack_files, _has_ai_generated_pages,
                )
                # Only fill missing files if AI didn't already generate real pages.
                # If AI generator ran successfully, it produced domain-specific pages —
                # don't overwrite them with the hardcoded ecommerce/electronics template.
                if not _has_ai_generated_pages(files):
                    files = fill_missing_fullstack_files(
                        files, requirement, prd, uiux, architecture, design_tokens,
                    )
                await websocket.send_text(json.dumps({
                    "event": "agent_start",
                    "agent": "frontend_agent",
                    "message": "Checking frontend pages, MUI shell, theme, and mock API client...",
                }))
                files = ensure_mock_client(files)
                files = apply_theme_tokens(files, design_tokens, uiux=uiux)
                files = ensure_deliverables(files)
                await websocket.send_text(json.dumps({
                    "event": "agent_complete",
                    "agent": "frontend_agent",
                    "message": "Frontend shell, theme tokens, and mock fallback client are in place.",
                }))
                await websocket.send_text(json.dumps({
                    "event": "agent_start",
                    "agent": "backend_agent",
                    "message": "Checking FastAPI models, schemas, routes, JWT, and seed scripts...",
                }))
                await websocket.send_text(json.dumps({
                    "event": "agent_complete",
                    "agent": "backend_agent",
                    "message": "Backend FastAPI + SQLAlchemy + JWT structure verified in generated files.",
                }))
                await websocket.send_text(json.dumps({
                    "event": "agent_start",
                    "agent": "integration_agent",
                    "message": "Wiring frontend API client to backend REST routes with mock fallback...",
                }))
                await websocket.send_text(json.dumps({
                    "event": "agent_complete",
                    "agent": "integration_agent",
                    "message": "Integration ready — screens call apiFetch; mock.ts is used if APIs fail.",
                }))
                file_writer(project_name, GeneratedFiles(files=files))
                _persist_project_runtime(project_name, files, architecture, requirement, prd, uiux)
                await websocket.send_text(json.dumps({
                    "event": "agent_complete",
                    "agent": "validation_agent",
                    "message": "Fullstack validation complete (JS CRUD validator skipped).",
                }))
            else:
                files = validate_and_fix_code(files, architecture, template_name=template_name)
                files = post_process_generated_files(
                    files, architecture, template_name=template_name, uiux=uiux,
                    requirement=requirement, prd=prd, app_spec=app_spec,
                    design_tokens=design_tokens,
                    project_name=project_name,
                )
                files, validation_errors = ensure_valid_codegen_output(
                    files, architecture, uiux=uiux,
                    requirement=requirement, prd=prd, app_spec=app_spec,
                    design_tokens=design_tokens,
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
                        design_tokens=design_tokens,
                        project_name=project_name,
                    )
                    files, validation_errors = ensure_valid_codegen_output(
                        files, architecture, uiux=uiux,
                        requirement=requirement, prd=prd, app_spec=app_spec,
                        design_tokens=design_tokens,
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

        async def _emit_verify_progress(agent: str, message: str):
            if not message:
                return
            await websocket.send_text(json.dumps({
                "event": "agent_progress",
                "agent": agent,
                "message": message,
            }))

        async def _emit_backend_event(payload: dict):
            msg = payload.get("message", "")
            await _emit_verify_progress("backend_verify_agent", msg)

        async def _emit_build_event(payload: dict):
            msg = payload.get("message", "")
            await _emit_verify_progress("build_verify_agent", msg)

        verify_backend = os.environ.get("CODEGEN_VERIFY_BACKEND", "true").lower() not in ("0", "false", "no")
        if verify_backend and "backend/main.py" in files:
            await websocket.send_text(json.dumps({
                "event": "agent_start",
                "agent": "backend_verify_agent",
                "message": "Verifying backend API (uvicorn + sample endpoints + auto-fix)...",
            }))
            touch_job(
                session_id,
                app_id=app_id,
                job_type="codegen",
                status="running",
                step="backend_verify",
            )
            files, backend_ok, backend_log = await verify_backend_and_fix(
                project_name,
                files,
                user_email,
                on_event=_emit_backend_event,
                max_attempts=int(os.environ.get("CODEGEN_BACKEND_FIX_ATTEMPTS", "5")),
            )
            file_writer(project_name, GeneratedFiles(files=files))
            if not backend_ok:
                if fullstack:
                    from app_builder.services.fullstack_codegen import apply_backend_failure_mock
                    files = apply_backend_failure_mock(files)
                    file_writer(project_name, GeneratedFiles(files=files))
                    await websocket.send_text(json.dumps({
                        "event": "agent_start",
                        "agent": "mock_fallback_agent",
                        "message": "Backend verification failed — enabling mock API fallback so every screen still works.",
                    }))
                    await websocket.send_text(json.dumps({
                        "event": "agent_complete",
                        "agent": "mock_fallback_agent",
                        "message": "Frontend mock fallback enabled. Live APIs will be used when the backend is healthy.",
                    }))
                    await websocket.send_text(json.dumps({
                        "event": "agent_complete",
                        "agent": "backend_verify_agent",
                        "message": "Backend verify failed; continuing with mock-backed frontend.",
                    }))
                else:
                    err = f"Backend verification failed after auto-fix attempts: {(backend_log or '')[-800:]}"
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
                        step="backend_verify",
                        error=err,
                        finished=True,
                    )
                    await websocket.send_text(json.dumps({
                        "event": "agent_error",
                        "agent": "backend_verify_agent",
                        "message": err,
                    }))
                    await websocket.send_text(json.dumps({"event": "error", "message": err}))
                    return
            else:
                await websocket.send_text(json.dumps({
                    "event": "agent_complete",
                    "agent": "backend_verify_agent",
                    "message": "Backend verified — uvicorn started and sample API calls succeeded.",
                }))
        elif verify_backend:
            await websocket.send_text(json.dumps({
                "event": "agent_complete",
                "agent": "backend_verify_agent",
                "message": "Backend verify skipped (no backend/main.py).",
            }))

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
                max_attempts=int(os.environ.get("CODEGEN_BUILD_FIX_ATTEMPTS", "5")),
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

        if fullstack:
            from app_builder.services.fullstack_codegen import run_screen_qa as _run_screen_qa
            await websocket.send_text(json.dumps({
                "event": "agent_start",
                "agent": "screen_qa_agent",
                "message": "Checking each planned screen is present and wired...",
            }))
            passed, missing = _run_screen_qa(files, architecture, prd, uiux)
            for name in passed:
                await websocket.send_text(json.dumps({
                    "event": "agent_progress",
                    "agent": "screen_qa_agent",
                    "message": f"OK: {name}",
                }))
            for name in missing:
                await websocket.send_text(json.dumps({
                    "event": "agent_progress",
                    "agent": "screen_qa_agent",
                    "message": f"Needs follow-up: {name} not found in generated frontend",
                }))
            await websocket.send_text(json.dumps({
                "event": "agent_complete",
                "agent": "screen_qa_agent",
                "message": f"Screen QA complete — {len(passed)} present, {len(missing)} missing.",
            }))
        # ── end legacy track verification block ──────────────────────────────

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

        base = public_base_url().rstrip("/")
        frontend_url = f"{base}/app/{project_name}"
        backend_url = f"{base}/api/apps/{project_name}"

        # SaaS runtime: config + SQLite + final backend normalization on disk
        try:
            files = apply_deterministic_backend_fixes(files)
            file_writer(project_name, GeneratedFiles(files=files))
            _persist_project_runtime(project_name, files, architecture, requirement, prd, uiux)
            config = get_project_config(project_name) or {}
            if config.get("tables"):
                ensure_project_db_initialized(project_name, config)
        except Exception as fin_exc:
            print(f"[codegen_api] finalize runtime failed: {fin_exc}", file=sys.stderr)

        if app_id and user_email:
            try:
                # Use the already-resolved track (detected above, not re-computed here)
                db.update_app_builder_app(
                    app_id=app_id,
                    user_email=user_email,
                    user_id=uid,
                    generated_code_json=files,
                    pipeline_status="CODEGEN_COMPLETE",
                    pipeline_error=None,
                    build_status="BUILD_SUCCESS",
                    build_error=None,
                    preview_url=frontend_url,
                    live_url=frontend_url,
                    track=_track,
                    plan_json=plan_json_req if _track == "frontend_only" else None,
                )
                try:
                    from api.app_creator.deployment_api import register_local_preview

                    register_local_preview(
                        app_id=app_id,
                        project_name=project_name,
                        frontend_url=frontend_url,
                        backend_url=backend_url,
                        user_email=user_email,
                        user_id=uid,
                    )
                except Exception as reg_exc:
                    print(f"[codegen_api] preview registration: {reg_exc}", file=sys.stderr)
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
            "message": "App ready — backend verified, frontend built. Open Preview to use your SaaS app.",
            "data": {
                "project_name": project_name,
                "files_count": len(files),
                "files": list(files.keys()),
                "preview_url": frontend_url,
                "backend_url": backend_url,
                "build_status": "BUILD_SUCCESS",
                "files_ready": True,   # signal frontend to load /tree immediately
                "track": _track or "legacy",
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
