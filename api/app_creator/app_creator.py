from fastapi import APIRouter, HTTPException, Query, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import os

logger = logging.getLogger("app_builder")
# Ensure app_builder logs are visible (INFO level, console handler)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
import json
import shlex
import time
import subprocess
import shutil
import tempfile
import asyncio

from app_builder.services.runtime_paths import get_projects_dir, resolve_project_root
from app_builder.schemas.files import GeneratedFiles
from app_builder.services.file_writer import file_writer
from api.auth.request_auth import resolve_user, CurrentUser, user_email_from
from api.app_creator.project_access import assert_project_access
from db.app_builder import get_app_builder_db
from api.app_creator.cra_npm_patch import CRA_BUILD_ENV, prepare_frontend_dir_on_disk
from api.app_creator.build_verify_service import verify_build_and_fix
from api.app_creator.e2b_sandbox_build import build_frontend_in_e2b, e2b_available
from api.app_creator.pipeline_helpers import npm_build_timeout, npm_install_timeout, public_base_url
from api.app_creator.static_app_router import _resolve_static_dir

router = APIRouter(prefix="/api/app-builder", tags=["App Builder"])

_app_builder_db = get_app_builder_db()


def _assert_project_access(project_name: str, current: CurrentUser) -> None:
    assert_project_access(project_name, current)


PROJECTS_DIR = get_projects_dir()


def _project_root(project_name: str) -> str:
    return resolve_project_root(project_name)


def _safe_join_project(project_name: str, relative_path: str) -> str:
    # Prevent path traversal
    root = os.path.abspath(_project_root(project_name))
    full = os.path.abspath(os.path.join(root, relative_path))
    if not (full == root or full.startswith(root + os.sep)):
        raise HTTPException(status_code=400, detail="Invalid path")
    return full


EXCLUDE_DIRS = {
    "node_modules",
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
}


def _build_tree(project_name: str) -> List[Dict[str, Any]]:
    root = _project_root(project_name)
    if not os.path.exists(root):
        raise HTTPException(status_code=404, detail="Project not found")

    nodes: Dict[str, Dict[str, Any]] = {}

    def ensure_dir(path_parts: List[str]) -> Dict[str, Any]:
        cur_key = ""
        parent = None
        for part in path_parts:
            cur_key = f"{cur_key}/{part}" if cur_key else part
            if cur_key not in nodes:
                nodes[cur_key] = {"name": part, "type": "folder", "path": cur_key, "children": []}
                if parent is not None:
                    parent["children"].append(nodes[cur_key])
            parent = nodes[cur_key]
        return parent or {"children": []}

    # Create virtual root list
    top_level: List[Dict[str, Any]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        rel_dir = os.path.relpath(dirpath, root)
        rel_dir = "" if rel_dir == "." else rel_dir

        # Ensure dir node exists in tree
        if rel_dir:
            dir_node = ensure_dir(rel_dir.split(os.sep))
        else:
            dir_node = {"children": top_level}

        for filename in filenames:
            rel_path = os.path.join(rel_dir, filename) if rel_dir else filename
            file_node = {"name": filename, "type": "file", "path": rel_path}
            dir_node["children"].append(file_node)

        # Attach first-level directories to top_level
        if not rel_dir:
            # we already add file nodes to top_level via dir_node["children"]
            pass
        else:
            first = rel_dir.split(os.sep)[0]
            if first in nodes and nodes[first] not in top_level:
                top_level.append(nodes[first])

    # Sort folders first, then files
    def sort_children(node: Dict[str, Any]) -> None:
        if node.get("type") != "folder":
            return
        node["children"].sort(key=lambda x: (0 if x.get("type") == "folder" else 1, x.get("name", "")))
        for c in node["children"]:
            sort_children(c)

    for n in top_level:
        sort_children(n)

    top_level.sort(key=lambda x: (0 if x.get("type") == "folder" else 1, x.get("name", "")))
    return top_level


class UpdateFileRequest(BaseModel):
    path: str
    content: str


class RunRequest(BaseModel):
    backend_port: Optional[int] = None
    frontend_port: Optional[int] = None
    install: bool = True
    skip_build: bool = False  # True when frontend is already built/deployed (e.g. S3)
    # None = follow APP_BUILDER_USE_E2B + E2B_API_KEY; True = require E2B; False = local npm only
    use_sandbox: Optional[bool] = None


# Simple in-memory index of processes per project (legacy; no longer used for proc_id).
PROJECT_RUN_STATE: Dict[str, Dict[str, Any]] = {}


def _capture_npm_error(r: subprocess.CompletedProcess, prefix: str, cwd: Optional[str] = None) -> str:
    """Build error message from npm subprocess result. Handles empty output (e.g. PATH or permission issues)."""
    raw = (r.stderr or b"") + (r.stdout or b"")
    try:
        err = raw.decode("utf-8", errors="replace").strip()
    except Exception:
        err = raw.decode("latin-1", errors="replace").strip()
    # Keep full npm output for build failures (stored in DB / streamed to UI)
    max_len = 8000
    if len(err) > max_len:
        err = err[:max_len] + f"\n... (truncated, {len(raw)} bytes total)"
    if not err:
        err = (
            f"Exit code {r.returncode}. No output captured. "
            "On servers: ensure Node.js and npm are in PATH for the process user, and the process has read/write access to the project directory."
        )
        if cwd:
            err += f" Cwd was: {cwd}"
    return f"{prefix} (exit {r.returncode}): {err}"


def _resolve_e2b_build(use_sandbox: Optional[bool]) -> Tuple[bool, Optional[str]]:
    """
    Decide whether to run npm in an E2B sandbox.
    Returns (use_e2b, error_message). error_message is set only when use_sandbox=True but deps are missing.
    """
    if use_sandbox is True:
        if not os.environ.get("E2B_API_KEY"):
            return False, "use_sandbox=true requires E2B_API_KEY (https://e2b.dev)"
        try:
            __import__("e2b")
        except ImportError:
            return False, "use_sandbox=true requires the `e2b` package (add to requirements.txt)"
        return True, None
    if use_sandbox is False:
        return False, None
    if os.environ.get("APP_BUILDER_USE_E2B", "").lower() not in ("1", "true", "yes"):
        return False, None
    if not e2b_available():
        logger.warning(
            "APP_BUILDER_USE_E2B is set but E2B_API_KEY or e2b package is missing; using local npm"
        )
        return False, None
    return True, None


def _build_frontend(
    project_name: str,
    install: bool = True,
    use_sandbox: Optional[bool] = None,
    on_sandbox_log: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Run npm build in frontend dir or E2B sandbox. Returns (static_dir_path, error_message)."""
    project_root = _project_root(project_name)
    if not os.path.exists(project_root):
        return None, "Project not found"
    frontend_dir = None
    if os.path.isdir(os.path.join(project_root, "frontend")):
        frontend_dir = _resolve_frontend_dir(project_root, "frontend")
        if not frontend_dir:
            frontend_dir = os.path.join(project_root, "frontend")
    elif os.path.exists(os.path.join(project_root, "package.json")):
        frontend_dir = project_root
    if not frontend_dir or not os.path.exists(os.path.join(frontend_dir, "package.json")):
        return None, "No frontend found"

    from api.app_creator.vite_build import is_vite_frontend, prepare_vite_frontend_dir, run_vite_build

    if is_vite_frontend(frontend_dir):
        logger.info("[build] Vite frontend — npm install + vite build")
        prepare_vite_frontend_dir(frontend_dir)
        static_dir, err, _log = run_vite_build(frontend_dir, project_name, install=install)
        if err:
            return None, err
        return static_dir, None

    patched = prepare_frontend_dir_on_disk(frontend_dir)
    if patched:
        logger.info("[build] Applied CRA/craco patch — clean npm install required")

    use_e2b, forced_err = _resolve_e2b_build(use_sandbox)
    if forced_err:
        return None, forced_err
    if use_e2b:
        return build_frontend_in_e2b(
            project_name,
            frontend_dir,
            install=install,
            on_log=on_sandbox_log,
        )

    try:
        if install:
            # Do not use --silent so we capture real errors (e.g. on EC2: permissions, network, node version)
            r = subprocess.run(
                ["npm", "install", "--legacy-peer-deps"],
                cwd=frontend_dir,
                capture_output=True,
                timeout=npm_install_timeout(),
            )
            if r.returncode != 0:
                return None, _capture_npm_error(r, "npm install failed", cwd=frontend_dir)
        # PUBLIC_URL ensures CRA/Vite build asset paths match /app/{project_id} base path
        build_env = os.environ.copy()
        build_env["PUBLIC_URL"] = f"/app/{project_name}"
        for key, val in CRA_BUILD_ENV.items():
            build_env.setdefault(key, val)
        r = subprocess.run(
            ["npm", "run", "build"],
            cwd=frontend_dir,
            capture_output=True,
            timeout=npm_build_timeout(),
            env=build_env,
        )
        if r.returncode != 0:
            return None, _capture_npm_error(r, "npm run build failed", cwd=frontend_dir)
        dist = os.path.join(frontend_dir, "dist")
        build_dir = os.path.join(frontend_dir, "build")
        if os.path.isdir(dist):
            return dist, None
        if os.path.isdir(build_dir):
            return build_dir, None
        return None, "Build completed but dist/build not found"
    except subprocess.TimeoutExpired:
        return None, "Build timed out"
    except FileNotFoundError as e:
        msg = str(e).lower()
        if "npm" in msg or getattr(e, "filename", "") == "npm":
            return None, "npm not found. On the server, install Node.js and npm and ensure they are in PATH for the process (e.g. systemd service)."
        return None, str(e)
    except Exception as e:
        return None, str(e)


@router.get("/projects/{project_name}/tree")
async def get_project_tree(project_name: str, current: CurrentUser = Depends(resolve_user)):
    _assert_project_access(project_name, current)
    tree = _build_tree(project_name)
    return JSONResponse(content={"project_name": project_name, "tree": tree})


@router.get("/projects/{project_name}/file")
async def get_project_file(
    project_name: str,
    path: str = Query(..., description="Relative path within project"),
    current: CurrentUser = Depends(resolve_user),
):
    _assert_project_access(project_name, current)
    full_path = _safe_join_project(project_name, path)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return JSONResponse(content={"project_name": project_name, "path": path, "content": content})


@router.put("/projects/{project_name}/file")
async def update_project_file(
    project_name: str,
    request: UpdateFileRequest,
    current: CurrentUser = Depends(resolve_user),
):
    _assert_project_access(project_name, current)
    full_path = _safe_join_project(project_name, request.path)
    dir_name = os.path.dirname(full_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(request.content)
    return JSONResponse(content={"status": "ok", "project_name": project_name, "path": request.path})


@router.get("/projects/{project_name}/download")
async def download_project(project_name: str, current: CurrentUser = Depends(resolve_user)):
    _assert_project_access(project_name, current)
    """
    Zips the project directory and returns it as a downloadable file.
    """
    project_root = _project_root(project_name)
    if not os.path.exists(project_root):
        raise HTTPException(status_code=404, detail="Project not found")

    # Create a temporary file for the zip
    # We use mkdtemp to ensure we have a clean place, or just name the file in temp
    temp_dir = tempfile.mkdtemp()
    base_name = os.path.join(temp_dir, project_name)
    
    # shutil.make_archive creates user_provided_base_name + .zip
    archive_path = shutil.make_archive(base_name, 'zip', project_root)
    
    # Check if created
    if not os.path.exists(archive_path):
        raise HTTPException(status_code=500, detail="Failed to create zip archive")

    # Simple cleanup task? 
    # FastAPI BackgroundTasks is better for cleanup, but let's keep it simple for now or rely on OS temp cleanup/BackgroundTask if needed.
    # We will use BackgroundTask to clean up the temp file after sending.
    from starlette.background import BackgroundTask

    def cleanup(path: str, dir_path: str):
        try:
            os.remove(path)
            os.rmdir(dir_path)
        except Exception as e:
            print(f"Error cleaning up temp zip: {e}")

    return FileResponse(
        path=archive_path, 
        filename=f"{project_name}.zip", 
        media_type='application/zip', 
        background=BackgroundTask(cleanup, archive_path, temp_dir)
    )


def _resolve_frontend_dir(project_root: str, frontend_sub: str):
    """Resolve frontend directory: may be project_root/frontend, nested client, or project_root (single app). Returns path or None."""
    frontend_dir = os.path.join(project_root, frontend_sub)
    if not os.path.isdir(frontend_dir):
        return None
    possible = os.path.join(frontend_dir, "client")
    if os.path.isdir(possible) and os.path.exists(os.path.join(possible, "package.json")):
        return possible
    return frontend_dir


def _sync_project_disk_from_db(project_name: str, current: CurrentUser) -> int:
    """Hydrate disk from Postgres generated_code_json. Returns file count written."""
    user_email = user_email_from(current)
    uid = current.id if current.id else None
    app = _app_builder_db.get_app_by_project_name(project_name, user_id=uid, user_email=user_email)
    if not app:
        return 0
    files = app.get("generated_code_json") or {}
    if not isinstance(files, dict) or not files:
        return 0
    try:
        from app_builder.services.code_post_process import post_process_generated_files
        files = post_process_generated_files(
            files,
            app.get("architecture") or {},
            uiux=app.get("generated_uiux") or "",
        )
    except Exception as exc:
        logger.warning("[sync-from-db] post_process failed: %s", exc)
    file_writer(project_name, GeneratedFiles(files=files))
    project_root = _project_root(project_name)
    frontend_dir = _resolve_frontend_dir(project_root, "frontend")
    if frontend_dir:
        prepare_frontend_dir_on_disk(frontend_dir)
    return len(files)


def _resolve_static_dir_from_disk(project_name: str) -> Optional[str]:
    project_root = _project_root(project_name)
    frontend_dir = _resolve_frontend_dir(project_root, "frontend")
    if not frontend_dir:
        frontend_dir = os.path.join(project_root, "frontend")
    for sub in ("build", "dist", os.path.join("client", "build"), os.path.join("client", "dist")):
        path = os.path.join(frontend_dir, sub)
        if os.path.isdir(path):
            return path
    return None


def _project_files_from_app(app_row: Optional[dict]) -> dict:
    files = (app_row or {}).get("generated_code_json") or {}
    return dict(files) if isinstance(files, dict) else {}


@router.post("/projects/{project_name}/sync-from-db")
async def sync_project_from_db(
    project_name: str,
    current: CurrentUser = Depends(resolve_user),
):
    """Write generated_code_json from Postgres to disk (fixes DB/disk desync)."""
    _assert_project_access(project_name, current)
    count = _sync_project_disk_from_db(project_name, current)
    if count == 0:
        raise HTTPException(status_code=404, detail="No generated code found in database for this project")
    return {"status": "ok", "project_name": project_name, "files_count": count}


@router.post("/projects/{project_name}/run")
async def run_project(
    project_name: str,
    request: RunRequest,
    http_request: Request,
    current: CurrentUser = Depends(resolve_user),
):
    """Build frontend and serve via main backend. No subprocess spawn."""
    _assert_project_access(project_name, current)
    user_email = user_email_from(current)
    uid = current.id if current.id else None
    app_row = _app_builder_db.get_app_by_project_name(project_name, user_id=uid, user_email=user_email)
    app_id = app_row.get("id") if app_row else None

    async def event_generator():
        build_log_lines: list[str] = []
        try:
            logger.info("[run] START | project=%s (single-backend mode)", project_name)
            msg = f"Building and serving {project_name}..."
            build_log_lines.append(msg)
            yield json.dumps({"event": "status", "message": msg}) + "\n"

            if app_id:
                _app_builder_db.update_app_builder_app(
                    app_id=app_id,
                    user_email=user_email,
                    user_id=uid,
                    build_status="BUILDING",
                    build_error=None,
                    build_log="\n".join(build_log_lines),
                )

            synced = _sync_project_disk_from_db(project_name, current)
            if synced:
                sync_msg = f"Synced {synced} files from database to disk before build."
                build_log_lines.append(sync_msg)
                yield json.dumps({"event": "status", "message": sync_msg}) + "\n"

            project_root = _project_root(project_name)
            if not os.path.exists(project_root):
                logger.error("[run] project not found: %s", project_root)
                err = "Project not found on disk (no generated code in database to sync)."
                build_log_lines.append(err)
                if app_id:
                    _app_builder_db.update_app_builder_app(
                        app_id=app_id,
                        user_email=user_email,
                        user_id=uid,
                        build_status="BUILD_FAILED",
                        build_error=err,
                        build_log="\n".join(build_log_lines),
                    )
                yield json.dumps({"event": "error", "message": err}) + "\n"
                return

            static_dir = None
            err = None
            if request.skip_build:
                sync_msg = "Skipping build (frontend already deployed)."
                build_log_lines.append(sync_msg)
                yield json.dumps({"event": "status", "message": sync_msg}) + "\n"
                static_dir = _resolve_static_dir(project_name)
            else:
                use_e2b, e2b_err = _resolve_e2b_build(request.use_sandbox)
                if e2b_err:
                    build_log_lines.append(e2b_err)
                    if app_id:
                        _app_builder_db.update_app_builder_app(
                            app_id=app_id,
                            user_email=user_email,
                            user_id=uid,
                            build_status="BUILD_FAILED",
                            build_error=e2b_err,
                            build_log="\n".join(build_log_lines),
                        )
                    yield json.dumps({"event": "error", "message": e2b_err}) + "\n"
                    return
                if use_e2b:
                    status_msg = "Building frontend in E2B cloud sandbox (npm install + build)..."
                else:
                    status_msg = "Building frontend..."
                build_log_lines.append(status_msg)
                yield json.dumps({"event": "status", "message": status_msg}) + "\n"

                build_task = asyncio.create_task(
                    asyncio.to_thread(
                        _build_frontend,
                        project_name,
                        request.install,
                        request.use_sandbox,
                    )
                )
                while not build_task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(build_task), timeout=12.0)
                    except asyncio.TimeoutError:
                        heartbeat = "Still building (npm install / npm run build in progress)..."
                        build_log_lines.append(heartbeat)
                        yield json.dumps({"event": "status", "message": heartbeat}) + "\n"

                static_dir, err = build_task.result()

                auto_fix = os.environ.get("RUN_BUILD_AUTO_FIX", "true").lower() not in ("0", "false", "no")
                if err and auto_fix and not use_e2b:
                    fix_msg = "Build failed — applying CRA patch and AI auto-fix (up to 3 attempts)..."
                    build_log_lines.append(fix_msg)
                    yield json.dumps({"event": "status", "message": fix_msg}) + "\n"

                    files = _project_files_from_app(app_row)
                    event_queue: asyncio.Queue = asyncio.Queue()

                    async def on_fix_event(payload: dict):
                        await event_queue.put(payload)

                    verify_task = asyncio.create_task(
                        verify_build_and_fix(
                            project_name,
                            files,
                            user_email,
                            on_event=on_fix_event,
                            max_attempts=int(os.environ.get("RUN_BUILD_FIX_ATTEMPTS", "3")),
                        )
                    )

                    while not verify_task.done():
                        try:
                            payload = await asyncio.wait_for(event_queue.get(), timeout=12.0)
                            msg = payload.get("message", "")
                            if msg:
                                build_log_lines.append(msg)
                                yield json.dumps({"event": "status", "message": msg}) + "\n"
                        except asyncio.TimeoutError:
                            hb = "Still fixing / rebuilding..."
                            yield json.dumps({"event": "status", "message": hb}) + "\n"

                    _files, build_ok, build_log = await verify_task
                    if build_log:
                        build_log_lines.append(build_log[-4000:])

                    if build_ok:
                        static_dir = _resolve_static_dir_from_disk(project_name)
                        err = None if static_dir else "Build succeeded but output folder not found"
                        if app_id and _files:
                            try:
                                _app_builder_db.update_app_builder_app(
                                    app_id=app_id,
                                    user_email=user_email,
                                    user_id=uid,
                                    generated_code_json=_files,
                                )
                            except Exception as save_exc:
                                logger.warning("[run] Could not save auto-fixed files: %s", save_exc)
                    elif not err:
                        err = (build_log or "Build failed after auto-fix attempts")[-800:]

                if err:
                    build_log_lines.append(err)
                    if app_id:
                        _app_builder_db.update_app_builder_app(
                            app_id=app_id,
                            user_email=user_email,
                            user_id=uid,
                            build_status="BUILD_FAILED",
                            build_error=err,
                            build_log="\n".join(build_log_lines),
                        )
                    yield json.dumps({"event": "error", "message": err}) + "\n"
                    return
                build_log_lines.append("Build complete. Serving from main backend.")
                yield json.dumps({"event": "status", "message": "Build complete. Serving from main backend."}) + "\n"

            resolved = static_dir or _resolve_static_dir(project_name)
            if not resolved or not os.path.isdir(resolved):
                err = (
                    "Build output not found on disk. Expected frontend/dist, frontend/build, "
                    "frontend/client/dist, or frontend/client/build."
                )
                build_log_lines.append(err)
                if app_id:
                    _app_builder_db.update_app_builder_app(
                        app_id=app_id,
                        user_email=user_email,
                        user_id=uid,
                        build_status="BUILD_FAILED",
                        build_error=err,
                        build_log="\n".join(build_log_lines),
                    )
                yield json.dumps({"event": "error", "message": err}) + "\n"
                return

            base = public_base_url(str(http_request.base_url).rstrip("/"))
            frontend_url = f"{base}/app/{project_name}"
            backend_url = f"{base}/api/apps/{project_name}"

            state = {
                "project_name": project_name,
                "backend_url": backend_url,
                "frontend_url": frontend_url,
                "backend": None,
                "frontend": None,
                "started_at": time.time(),
            }
            PROJECT_RUN_STATE[project_name] = state

            if app_id:
                _app_builder_db.update_app_builder_app(
                    app_id=app_id,
                    user_email=user_email,
                    user_id=uid,
                    build_status="BUILD_SUCCESS",
                    build_error=None,
                    build_log="\n".join(build_log_lines),
                    preview_url=frontend_url,
                    live_url=frontend_url,
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
                    logger.warning("[run] local preview registration failed: %s", reg_exc)

            logger.info("[run] DONE | project=%s | frontend=%s | backend=%s", project_name, frontend_url, backend_url)
            yield json.dumps({
                "event": "frontend_url",
                "frontend_url": frontend_url,
                "message": "Open this URL to view your app",
            }) + "\n"
            yield json.dumps({"event": "ready", "data": state, "message": "Application is running!"}) + "\n"

        except Exception as e:
            logger.exception("[run] FAILED: %s", e)
            import traceback
            tb = traceback.format_exc()
            err_msg = f"{str(e)}\n\nTraceback:\n{tb}"
            build_log_lines.append(err_msg)
            if app_id:
                try:
                    _app_builder_db.update_app_builder_app(
                        app_id=app_id,
                        user_email=user_email,
                        user_id=uid,
                        build_status="BUILD_FAILED",
                        build_error=str(e),
                        build_log="\n".join(build_log_lines),
                    )
                except Exception:
                    pass
            yield json.dumps({"event": "error", "message": err_msg}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

