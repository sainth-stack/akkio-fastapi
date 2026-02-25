from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Tuple
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
import re
import shlex
import time
import subprocess
import signal
import shutil
import tempfile

from app_builder.graph.builder_graph import app_builder_graph
from app_builder.schemas.requirements import UserRequirement
from app_builder.schemas.files import GeneratedFiles
from app_builder.services.file_writer import file_writer
from app_builder.services.runtime_paths import get_projects_dir, resolve_project_root
from app_builder.services.command_runner import find_free_port, process_registry


router = APIRouter(prefix="/api/app-builder", tags=["App Builder"])

PROJECTS_DIR = get_projects_dir()
AKKIO_FASTAPI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEGACY_PROJECTS_DIR = os.path.join(AKKIO_FASTAPI_DIR, "app_builder", ".runtime", "projects")

# Fixed ports for generated apps (frontend on 5002, backend on 5001)
FIXED_BACKEND_PORT = 5001
FIXED_FRONTEND_PORT = 5002


def kill_process_on_port(port: int) -> None:
    """Kill any process using the specified port."""
    try:
        # Find process using the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"Killed process {pid} on port {port}")
                except ProcessLookupError:
                    pass
                except Exception as e:
                    print(f"Error killing process {pid}: {e}")
            # Wait a moment for processes to die
            time.sleep(1)
    except Exception as e:
        print(f"Error checking port {port}: {e}")


def _project_root(project_name: str) -> str:
    return resolve_project_root(project_name)


def _is_port_available(port: int) -> bool:
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def _is_port_listening(port: int) -> bool:
    """Return True if something is accepting connections on the port."""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False


def _fix_backend_python_relative_imports(backend_dir: str) -> None:
    """Fix relative imports in all backend .py files so uvicorn main:app works (no parent package)."""
    if not os.path.isdir(backend_dir):
        return
    for name in os.listdir(backend_dir):
        if not name.endswith(".py"):
            continue
        path = os.path.join(backend_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if "from . import " not in content and "from ." not in content:
                continue
            content = re.sub(r"\bfrom\s+\.\s+import\s+", "import ", content)
            content = re.sub(r"\bfrom\s+\.(\w+)\s+import\s+", r"from \1 import ", content)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            pass


def _try_create_missing_backend_module_stub(backend_dir: str, log_content: str) -> bool:
    """
    If logs show ModuleNotFoundError: No module named 'XXX', and backend has no XXX.py/XXX/,
    create backend/XXX.py with stub classes by parsing main.py for 'from XXX import A, B'.
    Returns True if a stub was created so caller can retry backend start.
    """
    match = re.search(r"ModuleNotFoundError:\s*No module named ['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]", log_content)
    if not match:
        return False
    mod_name = match.group(1)
    mod_py = os.path.join(backend_dir, f"{mod_name}.py")
    mod_dir = os.path.join(backend_dir, mod_name)
    if os.path.isfile(mod_py) or os.path.isdir(mod_dir):
        return False
    main_py = os.path.join(backend_dir, "main.py")
    if not os.path.isfile(main_py):
        return False
    try:
        with open(main_py, "r", encoding="utf-8") as f:
            main_content = f.read()
        # Find "from MOD import X, Y" or "from MOD import X"
        imp_match = re.search(
            rf"\bfrom\s+{re.escape(mod_name)}\s+import\s+([^\n]+)",
            main_content,
        )
        if not imp_match:
            return False
        symbols_str = imp_match.group(1).strip()
        symbols = [s.strip().split(" as ")[0] for s in symbols_str.split(",") if s.strip()]
        if not symbols:
            return False
        stub_lines = ["# Auto-generated stub so backend can start. Implement as needed.", ""]
        for sym in symbols:
            if sym.isidentifier():
                stub_lines.append(f"class {sym}:")
                stub_lines.append("    pass")
                stub_lines.append("")
        with open(mod_py, "w", encoding="utf-8") as f:
            f.write("\n".join(stub_lines))
        return True
    except Exception:
        return False


def _try_patch_backend_response_model(backend_dir: str, log_content: str) -> bool:
    """
    If logs show FastAPIError about response_model (ORM model used instead of Pydantic schema),
    patch main.py: use schemas.X, or add response_model=None when ORM is defined in main (no schemas).
    Returns True if a patch was applied so caller can retry backend start.
    """
    if "Invalid args for response field" not in log_content or "valid Pydantic field type" not in log_content:
        return False
    match = re.search(r"<class ['\"](\w+)\.(\w+)['\"]>", log_content)
    main_py = os.path.join(backend_dir, "main.py")
    if not os.path.isfile(main_py):
        return False
    try:
        with open(main_py, "r", encoding="utf-8") as f:
            content = f.read()
        changed = False
        # response_model=models.X -> response_model=schemas.X
        new_content = re.sub(
            r"\bresponse_model=models\.(\w+)",
            r"response_model=schemas.\1",
            content,
        )
        if new_content != content:
            content = new_content
            changed = True
        if match:
            cls_name = match.group(2)
            # response_model=Todo -> response_model=schemas.Todo
            new_content = re.sub(
                rf"\bresponse_model={re.escape(cls_name)}\b",
                f"response_model=schemas.{cls_name}",
                content,
            )
            if new_content != content:
                content = new_content
                changed = True
            # Return type annotations: -> Todo, -> List[Todo], etc.
            new_content = re.sub(
                rf"\b->\s*{re.escape(cls_name)}\b",
                f"-> schemas.{cls_name}",
                content,
            )
            if new_content != content:
                content = new_content
                changed = True
            new_content = re.sub(
                rf"List\[{re.escape(cls_name)}\]",
                f"List[schemas.{cls_name}]",
                content,
            )
            if new_content != content:
                content = new_content
                changed = True
            new_content = re.sub(
                rf"Optional\[{re.escape(cls_name)}\]",
                f"Optional[schemas.{cls_name}]",
                content,
            )
            if new_content != content:
                content = new_content
                changed = True
        # Fallback: ORM defined in main.py (no schemas) – add response_model=None to route decorators
        if not changed:
            def add_response_model_none(m):
                line = m.group(0)
                if "response_model" in line:
                    return line
                return line[:-1] + ", response_model=None)"
            new_content = re.sub(
                r"@app\.(get|post|put|delete|patch)\([^)]+\)",
                add_response_model_none,
                content,
            )
            if new_content != content:
                content = new_content
                changed = True
        if changed:
            with open(main_py, "w", encoding="utf-8") as f:
                f.write(content)
            return True
    except Exception:
        pass
    return False


def _is_mongodb_backend(backend_dir: str) -> bool:
    for name in ("database.py", "db.py", "config.py"):
        path = os.path.join(backend_dir, name)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    if "MongoClient" in f.read() or "pymongo" in f.read():
                        return True
            except Exception:
                pass
    return False


def _ensure_backend_tables_created(backend_dir: str) -> None:
    if not os.path.isdir(backend_dir) or _is_mongodb_backend(backend_dir):
        return
    venv_python = os.path.join(backend_dir, "venv", "bin", "python")
    python_cmd = venv_python if os.path.isfile(venv_python) else "python3"
    for script in [
        "import sys; sys.path.insert(0, %r); import database; "
        "import models; database.Base.metadata.create_all(bind=database.engine)",
        "import sys; sys.path.insert(0, %r); import database; "
        "import models; models.Base.metadata.create_all(bind=database.engine)",
    ]:
        try:
            r = subprocess.run(
                [python_cmd, "-c", script % backend_dir],
                cwd=backend_dir,
                capture_output=True,
                timeout=10,
            )
            if r.returncode == 0:
                break
        except Exception:
            pass


def _try_patch_backend_database_to_sqlite(backend_dir: str, log_content: str) -> bool:
    if "5432" not in log_content and "Connection refused" not in log_content:
        return False
    if "psycopg2" not in log_content and "postgresql" not in log_content.lower():
        return False
    for name in ("database.py", "db.py", "config.py"):
        path = os.path.join(backend_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if "postgresql" not in content.lower():
                continue
            new_content = re.sub(
                r"getenv\s*\(\s*[\"']DATABASE_URL[\"']\s*,\s*[\"']postgresql[^\"']*[\"']\s*\)",
                "getenv(\"DATABASE_URL\", \"sqlite:///./app.db\")",
                content,
                flags=re.IGNORECASE,
            )
            if new_content == content:
                new_content = re.sub(
                    r"[\"']postgresql://[^\"']*[\"']",
                    "\"sqlite:///./app.db\"",
                    content,
                    count=1,
                )
            if new_content != content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return True
        except Exception:
            pass
    return False



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


def tree_from_files_dict(files: Dict[str, str]) -> List[Dict[str, Any]]:
    """Build a lightweight tree from an in-memory files dict."""
    root: List[Dict[str, Any]] = []

    def ensure_folder(children: List[Dict[str, Any]], name: str, path: str) -> Dict[str, Any]:
        existing = next((c for c in children if c.get("type") == "folder" and c.get("name") == name), None)
        if existing:
            return existing
        node = {"name": name, "type": "folder", "path": path, "children": []}
        children.append(node)
        return node

    for rel_path in sorted(files.keys()):
        parts = rel_path.split("/")
        cur_children = root
        cur_path = ""
        for part in parts[:-1]:
            cur_path = f"{cur_path}/{part}" if cur_path else part
            folder = ensure_folder(cur_children, part, cur_path)
            cur_children = folder["children"]
        cur_children.append({"name": parts[-1], "type": "file", "path": rel_path})

    return root


class GenerateRequest(BaseModel):
    requirement: str
    project_name: Optional[str] = "generated_app"


class UpdateFileRequest(BaseModel):
    path: str
    content: str


class RunRequest(BaseModel):
    backend_port: Optional[int] = None
    frontend_port: Optional[int] = None
    install: bool = True


# Simple in-memory index of processes per project.
PROJECT_RUN_STATE: Dict[str, Dict[str, Any]] = {}


@router.post("/generate")
async def generate_app(request: GenerateRequest):
    async def event_generator():
        try:
            logger.info("[generate] START | requirement=%r | project_name=%s", request.requirement[:80] + "..." if len(request.requirement) > 80 else request.requirement, request.project_name)
            yield json.dumps(
                {"event": "meta", "project_name": request.project_name, "message": "Starting app generation..."}
            ) + "\n"

            initial_state = {
                "user_requirement": UserRequirement(description=request.requirement),
                "clarified_requirement": "",
                "plan": None,
                "architecture": None,
                "generated_files": None,
                "template_data": None,
                "error": ""
            }

            for step in app_builder_graph.stream(initial_state):
                for node_name, node_output in step.items():
                    logger.info("[generate] step=%s | output_keys=%s", node_name, list(node_output.keys()) if node_output else [])

                    if "structured_requirement" in node_output:
                        sr = node_output.get("structured_requirement", {})
                        tmpl = sr.get("template_name") if isinstance(sr, dict) else None
                        proj = sr.get("project_name", "?") if isinstance(sr, dict) else "?"
                        logger.info("[generate] structuring done | project_name=%s | template_name=%s", proj, tmpl)

                    if "clarified_requirement" in node_output and node_output["clarified_requirement"]:
                        yield json.dumps(
                            {
                                "event": "clarified",
                                "agent": node_name,
                                "message": "Requirement clarified.",
                                "clarified_requirement": node_output["clarified_requirement"],
                            }
                        ) + "\n"

                    if "plan" in node_output and node_output["plan"]:
                        plan = node_output["plan"]
                        yield json.dumps(
                            {"event": "plan", "agent": node_name, "plan": plan.steps, "message": "Plan ready."}
                        ) + "\n"

                    if "architecture" in node_output and node_output["architecture"]:
                        yield json.dumps(
                            {"event": "architecture", "agent": node_name, "message": "Architecture decided."}
                        ) + "\n"

                    if "generated_files" in node_output and node_output["generated_files"]:
                        files_raw = node_output["generated_files"]
                        files_obj = GeneratedFiles(files=files_raw) if isinstance(files_raw, dict) else files_raw
                        files_dict = files_obj.files if hasattr(files_obj, "files") else files_raw
                        logger.info("[generate] writing %d files to project=%s", len(files_dict), request.project_name)
                        file_writer(request.project_name, GeneratedFiles(files=files_dict))
                        tree = tree_from_files_dict(files_dict)
                        yield json.dumps(
                            {"event": "files", "agent": node_name, "files": files_dict, "tree": tree}
                        ) + "\n"

                    if "error" in node_output and node_output["error"]:
                        logger.error("[generate] error from %s: %s", node_name, node_output["error"])
                        yield json.dumps({"event": "error", "agent": node_name, "message": node_output["error"]}) + "\n"

            logger.info("[generate] DONE | project=%s", request.project_name)
            yield json.dumps({"event": "done", "message": "App generation successful"}) + "\n"

        except Exception as e:
            logger.exception("[generate] EXCEPTION: %s", e)
            import traceback
            traceback.print_exc()
            yield json.dumps({"event": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@router.get("/projects/{project_name}/tree")
async def get_project_tree(project_name: str):
    tree = _build_tree(project_name)
    return JSONResponse(content={"project_name": project_name, "tree": tree})


@router.get("/projects/{project_name}/file")
async def get_project_file(project_name: str, path: str = Query(..., description="Relative path within project")):
    full_path = _safe_join_project(project_name, path)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return JSONResponse(content={"project_name": project_name, "path": path, "content": content})


@router.put("/projects/{project_name}/file")
async def update_project_file(project_name: str, request: UpdateFileRequest):
    full_path = _safe_join_project(project_name, request.path)
    dir_name = os.path.dirname(full_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(request.content)
    return JSONResponse(content={"status": "ok", "project_name": project_name, "path": request.path})


@router.get("/projects/{project_name}/download")
async def download_project(project_name: str):
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


@router.post("/projects/{project_name}/run")
async def run_project(project_name: str, request: RunRequest, http_request: Request):
    import asyncio

    async def event_generator():
        try:
            logger.info("[run] START | project=%s | backend_port=%s | frontend_port=%s", project_name, request.backend_port or FIXED_BACKEND_PORT, request.frontend_port or FIXED_FRONTEND_PORT)
            yield json.dumps({"event": "status", "message": f"Initializing run for {project_name}..."}) + "\n"

            project_root = _project_root(project_name)
            if not os.path.exists(project_root):
                logger.error("[run] project not found: %s", project_root)
                yield json.dumps({"event": "error", "message": "Project not found"}) + "\n"
                return

            backend_dir_candidate = os.path.join(project_root, "backend")
            frontend_dir_candidate = os.path.join(project_root, "frontend")
            root_has_package = os.path.exists(os.path.join(project_root, "package.json"))

            has_backend_dir = os.path.isdir(backend_dir_candidate)
            has_frontend_dir = os.path.isdir(frontend_dir_candidate)

            # Resolve backend dir (optional)
            backend_dir = None
            if has_backend_dir:
                backend_dir = backend_dir_candidate
                possible = os.path.join(backend_dir, "server")
                if os.path.isdir(possible) and (
                    os.path.exists(os.path.join(possible, "package.json"))
                    or os.path.exists(os.path.join(possible, "requirements.txt"))
                ):
                    backend_dir = possible
                    yield json.dumps({"event": "status", "message": "Detected nested backend..."}) + "\n"

            # Resolve frontend dir: frontend/ subdir, or project root if it has package.json (frontend-only app)
            frontend_dir = None
            if has_frontend_dir:
                frontend_dir = _resolve_frontend_dir(project_root, "frontend")
                if frontend_dir != frontend_dir_candidate:
                    yield json.dumps({"event": "status", "message": "Detected nested frontend..."}) + "\n"
            elif root_has_package:
                # Single app at root (e.g. Vite/React only)
                frontend_dir = project_root
                yield json.dumps({"event": "status", "message": "Running as single frontend app (project root)."}) + "\n"

            if not backend_dir and not frontend_dir:
                yield json.dumps({
                    "event": "error",
                    "message": "Project has no runnable app: add backend/, frontend/, or package.json in project root.",
                }) + "\n"
                return

            backend_port = request.backend_port or FIXED_BACKEND_PORT
            frontend_port = request.frontend_port or FIXED_FRONTEND_PORT
            yield json.dumps({"event": "status", "message": f"Cleaning up ports {backend_port} and {frontend_port}..."}) + "\n"
            kill_process_on_port(backend_port)
            kill_process_on_port(frontend_port)
            if not _is_port_available(backend_port):
                backend_port = find_free_port()
                yield json.dumps({"event": "status", "message": f"Backend port in use, switching to {backend_port}..."}) + "\n"
            if not _is_port_available(frontend_port):
                frontend_port = find_free_port()
                yield json.dumps({"event": "status", "message": f"Frontend port in use, switching to {frontend_port}..."}) + "\n"

            # Use request host so displayed URL works when accessed via EC2 IP (e.g. http://18.143.150.140/)
            _host = (http_request.headers.get("x-forwarded-host") or http_request.url.hostname or "localhost").split(":")[0]
            backend_url = f"http://{_host}:{backend_port}" if backend_dir else None
            frontend_url = f"http://{_host}:{frontend_port}" if frontend_dir else None

            backend_proc = None
            frontend_proc = None

            # ----- Start backend (if present) -----
            if backend_dir:
                yield json.dumps({"event": "status", "message": "Preparing backend..."}) + "\n"
                # Fix relative imports in all backend .py files so uvicorn main:app works (existing projects)
                _fix_backend_python_relative_imports(backend_dir)
                python_cmd = os.environ.get("PYTHON", "python3")
                venv_path = os.path.join(backend_dir, "venv")
                venv_python = os.path.join(venv_path, "bin", "python")
                venv_pip = os.path.join(venv_path, "bin", "pip")
                # Quote paths for shell (handles project names with spaces)
                q_venv_python = shlex.quote(venv_python)
                q_venv_pip = shlex.quote(venv_pip)
                q_python_cmd = shlex.quote(python_cmd)
                
                # Check if it's a Node.js backend
                if os.path.exists(os.path.join(backend_dir, "package.json")):
                    # FIXED: Always install dependencies if request.install is True
                    if request.install:
                        yield json.dumps({"event": "status", "message": "Installing backend dependencies..."}) + "\n"
                        install_cmd = "npm install --silent"
                        start_cmd = "npm run start:dev"
                        backend_cmd = f"{install_cmd} && PORT={backend_port} {start_cmd}"
                    else:
                        start_cmd = "npm run start:dev"
                        backend_cmd = f"PORT={backend_port} {start_cmd}"
                        
                # Python backend
                elif os.path.exists(os.path.join(backend_dir, "requirements.txt")):
                    if request.install:
                        yield json.dumps({"event": "status", "message": "Setting up Python environment and installing dependencies..."}) + "\n"
                        setup_cmd = f"if [ ! -d venv ]; then {q_python_cmd} -m venv venv; fi && {q_venv_pip} install -q -r requirements.txt"
                        try:
                            r = subprocess.run(["/bin/bash", "-c", setup_cmd], cwd=backend_dir, capture_output=True, timeout=120)
                            if r.returncode != 0:
                                yield json.dumps({"event": "warning", "message": f"Dependency install had issues: {r.stderr.decode()[:500]}"}) + "\n"
                        except Exception as e:
                            yield json.dumps({"event": "warning", "message": f"Setup failed: {e}"}) + "\n"
                        backend_cmd = f"{q_venv_python} -m uvicorn main:app --host 0.0.0.0 --port {backend_port}"
                    else:
                        backend_cmd = f"{q_venv_python} -m uvicorn main:app --host 0.0.0.0 --port {backend_port}" if os.path.exists(venv_python) else f"{q_python_cmd} -m uvicorn main:app --host 0.0.0.0 --port {backend_port}"
                else:
                    # Default Python backend without requirements.txt
                    backend_cmd = f"{q_venv_python} -m uvicorn main:app --host 0.0.0.0 --port {backend_port}" if os.path.exists(venv_python) else f"{q_python_cmd} -m uvicorn main:app --host 0.0.0.0 --port {backend_port}"
                
                # Use SQLite by default so app runs without PostgreSQL (generated apps often use getenv("DATABASE_URL"))
                backend_env = dict(os.environ)
                backend_env.setdefault("DATABASE_URL", "sqlite:///./app.db")

                # Ensure database tables exist before starting (prevents "no such table" error)
                _ensure_backend_tables_created(backend_dir)

                yield json.dumps({"event": "status", "message": "Starting backend process..."}) + "\n"
                backend_proc = process_registry.create(
                    name=f"{project_name}:backend",
                    command=["/bin/bash", "-c", backend_cmd],
                    cwd=backend_dir,
                    env=backend_env,
                )
                backend_proc.start()
                
                # Wait for backend to start; on ModuleNotFoundError, Postgres error, or FastAPI response_model error, try fix and retry once
                backend_ready = False
                backend_failed_continue_to_frontend = False
                last_log_idx = 0
                stub_retry_done = False
                db_retry_done = False
                response_model_retry_done = False

                for attempt in range(60):
                    if backend_proc.return_code() is not None:
                        _, logs = backend_proc.get_logs()
                        log_content = "\n".join(logs[-50:]) if logs else "No logs captured"

                        if not stub_retry_done and "ModuleNotFoundError" in log_content:
                            if _try_create_missing_backend_module_stub(backend_dir, log_content):
                                stub_retry_done = True
                                yield json.dumps({"event": "status", "message": "Created missing module stub, retrying backend..."}) + "\n"
                                backend_proc = process_registry.create(
                                    name=f"{project_name}:backend",
                                    command=["/bin/bash", "-c", backend_cmd],
                                    cwd=backend_dir,
                                    env=backend_env,
                                )
                                backend_proc.start()
                                last_log_idx = 0
                                await asyncio.sleep(1)
                                continue
                        if not db_retry_done and _try_patch_backend_database_to_sqlite(backend_dir, log_content):
                            db_retry_done = True
                            yield json.dumps({"event": "status", "message": "Switched database to SQLite (PostgreSQL not running), retrying backend..."}) + "\n"
                            backend_proc = process_registry.create(
                                name=f"{project_name}:backend",
                                command=["/bin/bash", "-c", backend_cmd],
                                cwd=backend_dir,
                                env=backend_env,
                                )
                            backend_proc.start()
                            last_log_idx = 0
                            await asyncio.sleep(1)
                            continue
                        if not response_model_retry_done and _try_patch_backend_response_model(backend_dir, log_content):
                            response_model_retry_done = True
                            yield json.dumps({"event": "status", "message": "Fixed FastAPI response_model, retrying backend..."}) + "\n"
                            backend_proc = process_registry.create(
                                name=f"{project_name}:backend",
                                command=["/bin/bash", "-c", backend_cmd],
                                cwd=backend_dir,
                                env=backend_env,
                            )
                            backend_proc.start()
                            last_log_idx = 0
                            await asyncio.sleep(1)
                            continue

                        # Backend failed - continue to frontend anyway
                        backend_url = None
                        backend_proc = None
                        yield json.dumps({
                            "event": "warning",
                            "message": f"Backend failed to start after fixes. Frontend will run without backend.\n\nLogs:\n{log_content}"
                        }) + "\n"
                        break

                    
                    # Stream logs to user
                    next_idx, lines = backend_proc.get_logs(since=last_log_idx)
                    last_log_idx = next_idx
                    for line in lines:
                        s = line.strip()
                        if s:
                            yield json.dumps({"event": "status", "message": f"[Backend] {s}"}) + "\n"
                            # Check for common error patterns
                            if "ModuleNotFoundError" in s or "ImportError" in s:
                                yield json.dumps({
                                    "event": "warning",
                                    "message": f"Backend has missing dependencies. Frontend will run without backend. Try install=true for backend."
                                }) + "\n"
                                backend_proc.terminate()
                                backend_url = None
                                backend_proc = None
                                backend_failed_continue_to_frontend = True
                                break
                            if "Error" in s and "Address already in use" in s:
                                yield json.dumps({
                                    "event": "warning",
                                    "message": f"Backend port {backend_port} in use. Frontend will run without backend."
                                }) + "\n"
                                backend_proc.terminate()
                                backend_url = None
                                backend_proc = None
                                backend_failed_continue_to_frontend = True
                                break
                    
                    if backend_failed_continue_to_frontend:
                        break

                    # FIXED: Actually check if backend is listening on port
                    if _is_port_listening(backend_port):
                        backend_ready = True
                        yield json.dumps({"event": "status", "message": f"Backend is ready on http://localhost:{backend_port}"}) + "\n"
                        break
                    
                    await asyncio.sleep(1)
                
                if not backend_ready and backend_proc:
                    _, logs = backend_proc.get_logs()
                    log_content = "\n".join(logs[-20:]) if logs else "No logs captured"
                    yield json.dumps({
                        "event": "warning",
                        "message": f"Backend did not start within 30 seconds. Frontend will run without backend.\n\nLogs:\n{log_content}"
                    }) + "\n"
                    backend_proc.terminate()
                    backend_url = None
                    backend_proc = None

            # ----- Start frontend (if present) -----
            if frontend_dir:
                yield json.dumps({"event": "status", "message": "Preparing frontend..."}) + "\n"
                # Write .env so CRA/Vite reliably gets backend URL (avoids undefined in browser)
                # Use empty REACT_APP_BACKEND_URL so frontend falls back to window.location.hostname - works
                # for localhost (dev) and EC2/public IP (deployment). Templates use getBackendUrl() which
                # returns window.location.protocol//hostname:5001 when env is empty.
                backend_url_val = ""
                env_file = os.path.join(frontend_dir, ".env")
                try:
                    with open(env_file, "w", encoding="utf-8") as f:
                        f.write(f"PORT={frontend_port}\n")
                        f.write(f"HOST=0.0.0.0\n")
                        f.write(f"REACT_APP_BACKEND_URL={backend_url_val}\n")
                        f.write(f"VITE_BACKEND_URL={backend_url_val}\n")
                        f.write("BROWSER=none\n")
                except Exception as e:
                    yield json.dumps({"event": "warning", "message": f"Could not write .env: {e}"}) + "\n"
                # Use full env so node/npm/nvm are on PATH; then override app vars
                # HOST=0.0.0.0 so React dev server listens on all interfaces (EC2 deployment)
                frontend_env = dict(os.environ)
                frontend_env.update({
                    "PORT": str(frontend_port),
                    "HOST": "0.0.0.0",
                    "BROWSER": "none",
                    "REACT_APP_BACKEND_URL": backend_url_val,
                    "VITE_BACKEND_URL": backend_url_val,
                    "NODE_OPTIONS": os.environ.get("NODE_OPTIONS", "--openssl-legacy-provider"),
                })
                frontend_cmd = "npm start"
                pkg_path = os.path.join(frontend_dir, "package.json")
                if os.path.exists(pkg_path):
                    try:
                        with open(pkg_path, "r", encoding="utf-8") as f:
                            pkg = json.load(f)
                        scripts = (pkg.get("scripts") or {})
                        if scripts.get("dev"):
                            frontend_cmd = f"npm run dev -- --host 0.0.0.0 --port {frontend_port}"
                        elif scripts.get("preview"):
                            frontend_cmd = f"npm run preview -- --host 0.0.0.0 --port {frontend_port}"
                        elif scripts.get("start"):
                            frontend_cmd = "npm start"
                    except Exception:
                        pass
                
                # FIXED: Install dependencies before starting. For Vite projects, ensure
                # @vitejs/plugin-react is installed (generated package.json sometimes omits it).
                if request.install and os.path.exists(pkg_path):
                    yield json.dumps({"event": "status", "message": "Installing frontend dependencies (this may take a moment)..."}) + "\n"
                    # Use --legacy-peer-deps and --force so peer/missing-version issues don't block install
                    install_parts = ["npm install --legacy-peer-deps --force"]
                    vite_config_js = os.path.join(frontend_dir, "vite.config.js")
                    vite_config_ts = os.path.join(frontend_dir, "vite.config.ts")
                    if os.path.exists(vite_config_js) or os.path.exists(vite_config_ts):
                        install_parts.append("npm install @vitejs/plugin-react --save-dev --legacy-peer-deps --force")
                    # CRA (react-scripts) often needs ajv@8; MUI needs @emotion/react, @emotion/styled, @mui/icons-material
                    try:
                        with open(pkg_path, "r", encoding="utf-8") as f:
                            pkg_data = json.load(f)
                        deps = pkg_data.get("dependencies") or {}
                        if deps.get("react-scripts"):
                            install_parts.append("npm install ajv@8 --legacy-peer-deps --force")
                        if deps.get("@mui/material") or deps.get("@mui/icons-material"):
                            install_parts.append(
                                "npm install @emotion/react @emotion/styled @mui/material @mui/icons-material --legacy-peer-deps --force"
                            )
                    except Exception:
                        pass
                    frontend_cmd = " && ".join(install_parts) + " && " + frontend_cmd
                
                frontend_proc = process_registry.create(
                    name=f"{project_name}:frontend",
                    command=["/bin/bash", "-c", frontend_cmd],
                    cwd=frontend_dir,
                    env=frontend_env,
                )
                frontend_proc.start()
                yield json.dumps({"event": "status", "message": f"Frontend starting on port {frontend_port}..."}) + "\n"
                
                last_log_idx = 0
                frontend_ready = False

                for attempt in range(120):
                    await asyncio.sleep(1)

                    if frontend_proc.return_code() is not None:
                        await asyncio.sleep(0.5)
                        _, lines = frontend_proc.get_logs(since=0)
                        all_logs = "\n".join(lines).strip() or "No output captured"
                        yield json.dumps({
                            "event": "error",
                            "message": f"Frontend process exited with code {frontend_proc.return_code()}.\n\nLogs:\n{all_logs}"
                        }) + "\n"
                        if backend_proc:
                            backend_proc.terminate()
                        return

                    
                    # Stream logs and check for errors
                    next_idx, lines = frontend_proc.get_logs(since=last_log_idx)
                    last_log_idx = next_idx
                    for line in lines:
                        s = line.strip()
                        if s:
                            yield json.dumps({"event": "status", "message": f"[Frontend] {s}"}) + "\n"
                            # Check for common error patterns
                            if "Module not found" in s or "Cannot find module" in s:
                                yield json.dumps({
                                    "event": "error",
                                    "message": f"Frontend has missing dependencies. Try running with install=true. Error: {s}"
                                }) + "\n"
                                frontend_proc.terminate()
                                if backend_proc:
                                    backend_proc.terminate()
                                return
                            if "EADDRINUSE" in s:
                                yield json.dumps({
                                    "event": "error",
                                    "message": f"Frontend port {frontend_port} is already in use."
                                }) + "\n"
                                frontend_proc.terminate()
                                if backend_proc:
                                    backend_proc.terminate()
                                return
                    
                    # Check if frontend is listening
                    if _is_port_listening(frontend_port):
                        frontend_ready = True
                        yield json.dumps({"event": "status", "message": f"Frontend is ready on http://localhost:{frontend_port}"}) + "\n"
                        yield json.dumps({"event": "frontend_url", "frontend_url": frontend_url, "message": "Open this URL to view your app"}) + "\n"
                        break
                
                if not frontend_ready:
                    _, all_logs = frontend_proc.get_logs()
                    log_content = "\n".join(all_logs[-30:]) if all_logs else "No logs captured"
                    yield json.dumps({
                        "event": "warning",
                        "message": f"Frontend did not start listening within 60 seconds. It may still be compiling. Check http://localhost:{frontend_port} in a moment.\n\nRecent logs:\n{log_content}"
                    }) + "\n"
                    # Still emit frontend_url so user can try opening it
                    yield json.dumps({"event": "frontend_url", "frontend_url": frontend_url, "message": "Frontend may still be compiling - try opening this URL"}) + "\n"

            # Always return frontend_url when frontend exists - app runs with or without backend
            state = {
                "project_name": project_name,
                "backend_url": backend_url,
                "frontend_url": frontend_url,
                "backend": {"proc_id": backend_proc.id} if backend_proc else None,
                "frontend": {"proc_id": frontend_proc.id} if frontend_proc else None,
                "started_at": time.time(),
            }
            PROJECT_RUN_STATE[project_name] = state
            logger.info("[run] DONE | project=%s | backend=%s | frontend=%s", project_name, backend_url, frontend_url)
            yield json.dumps({"event": "ready", "data": state, "message": "Application is running!"}) + "\n"

        except Exception as e:
            logger.exception("[run] FAILED: %s", e)
            import traceback
            tb = traceback.format_exc()
            yield json.dumps({"event": "error", "message": f"{str(e)}\n\nTraceback:\n{tb}"}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")