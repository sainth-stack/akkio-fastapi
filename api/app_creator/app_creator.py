from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import re
import time
import subprocess
import signal
import shutil
import tempfile

from app_builder.graph.builder_graph import app_builder_graph
from app_builder.schemas.requirements import UserRequirement
from app_builder.services.file_writer import file_writer
from app_builder.services.runtime_paths import get_projects_dir
from app_builder.services.command_runner import find_free_port, process_registry

router = APIRouter(prefix="/api/app-builder", tags=["App Builder"])

PROJECTS_DIR = get_projects_dir()
AKKIO_FASTAPI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEGACY_PROJECTS_DIR = os.path.join(AKKIO_FASTAPI_DIR, "app_builder", ".runtime", "projects")

# Fixed ports for generated apps (5000 often used by macOS AirPlay, so use 5002 for frontend)
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
    primary = os.path.join(PROJECTS_DIR, project_name)
    if os.path.exists(primary):
        return primary
    legacy = os.path.join(LEGACY_PROJECTS_DIR, project_name)
    if os.path.exists(legacy):
        return legacy
    return primary


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


def _try_patch_backend_database_to_sqlite(backend_dir: str, log_content: str) -> bool:
    """
    If logs show PostgreSQL connection error (psycopg2, Connection refused, 5432),
    patch backend database config to use SQLite so app runs without PostgreSQL.
    Returns True if a patch was applied so caller can retry backend start.
    """
    if "5432" not in log_content and "Connection refused" not in log_content:
        return False
    if "psycopg2" not in log_content and "postgresql" not in log_content.lower():
        return False
    # Find database.py or similar that sets DB URL
    for name in ("database.py", "db.py", "config.py"):
        path = os.path.join(backend_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if "postgresql" not in content.lower():
                continue
            # Replace postgresql URL with sqlite default so app runs without Postgres
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


def _tree_from_files_dict(files: Dict[str, str]) -> List[Dict[str, Any]]:
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
            yield json.dumps(
                {"event": "meta", "project_name": request.project_name, "message": "Starting app generation..."}
            ) + "\n"
            print(f"Received requirement: {request.requirement} (project={request.project_name})")
            
            initial_state = {
                "user_requirement": UserRequirement(description=request.requirement),
                "clarified_requirement": "",
                "plan": None,
                "architecture": None,
                "generated_files": None,
                "error": ""
            }
            
            # Using .stream() to get updates as they happen
            for step in app_builder_graph.stream(initial_state):
                for node_name, node_output in step.items():
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
                        files = node_output["generated_files"]
                        # Write files to disk
                        file_writer(request.project_name, files)
                        tree = _tree_from_files_dict(files.files)
                        yield json.dumps(
                            {"event": "files", "agent": node_name, "files": files.files, "tree": tree}
                        ) + "\n"
                    
                    if "error" in node_output and node_output["error"]:
                        yield json.dumps({"event": "error", "agent": node_name, "message": node_output["error"]}) + "\n"

            yield json.dumps({"event": "done", "message": "App generation successful"}) + "\n"

        except Exception as e:
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
async def run_project(project_name: str, request: RunRequest):
    import asyncio

    async def event_generator():
        try:
            yield json.dumps({"event": "status", "message": f"Initializing run for {project_name}..."}) + "\n"

            project_root = _project_root(project_name)
            if not os.path.exists(project_root):
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

            backend_url = f"http://localhost:{backend_port}" if backend_dir else None
            frontend_url = f"http://localhost:{frontend_port}" if frontend_dir else None

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
                    # FIXED: Create venv and install dependencies properly
                    if request.install:
                        yield json.dumps({"event": "status", "message": "Setting up Python environment and installing dependencies..."}) + "\n"
                        backend_cmd = f"""
if [ ! -d "venv" ]; then 
    {python_cmd} -m venv venv
fi
{venv_pip} install -q -r requirements.txt
{venv_python} -m uvicorn main:app --host 0.0.0.0 --port {backend_port}
"""
                    else:
                        backend_cmd = f"{venv_python} -m uvicorn main:app --host 0.0.0.0 --port {backend_port}" if os.path.exists(venv_python) else f"{python_cmd} -m uvicorn main:app --host 0.0.0.0 --port {backend_port}"
                else:
                    # Default Python backend without requirements.txt
                    backend_cmd = f"{venv_python} -m uvicorn main:app --host 0.0.0.0 --port {backend_port}" if os.path.exists(venv_python) else f"{python_cmd} -m uvicorn main:app --host 0.0.0.0 --port {backend_port}"
                
                # Use SQLite by default so app runs without PostgreSQL (generated apps often use getenv("DATABASE_URL"))
                backend_env = dict(os.environ)
                backend_env.setdefault("DATABASE_URL", "sqlite:///./app.db")
                
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
                last_log_idx = 0
                stub_retry_done = False
                db_retry_done = False
                response_model_retry_done = False
                for attempt in range(30):  # 30 seconds max wait
                    if backend_proc.return_code() is not None:
                        _, logs = backend_proc.get_logs()
                        log_content = "\n".join(logs[-25:]) if logs else "No logs captured"
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
                            yield json.dumps({"event": "status", "message": "Fixed FastAPI response_model (use Pydantic schemas), retrying backend..."}) + "\n"
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
                        yield json.dumps({
                            "event": "error",
                            "message": f"Backend failed to start. Exit code: {backend_proc.return_code()}\n\nLogs:\n{log_content}"
                        }) + "\n"
                        return
                    
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
                                    "event": "error",
                                    "message": f"Backend has missing dependencies. Try running with install=true. Error: {s}"
                                }) + "\n"
                                backend_proc.terminate()
                                return
                            if "Error" in s and "Address already in use" in s:
                                yield json.dumps({
                                    "event": "error",
                                    "message": f"Backend port {backend_port} is still in use. Please wait and try again."
                                }) + "\n"
                                backend_proc.terminate()
                                return
                    
                    # FIXED: Actually check if backend is listening on port
                    if _is_port_listening(backend_port):
                        backend_ready = True
                        yield json.dumps({"event": "status", "message": f"Backend is ready on http://localhost:{backend_port}"}) + "\n"
                        break
                    
                    await asyncio.sleep(1)
                
                if not backend_ready:
                    _, logs = backend_proc.get_logs()
                    log_content = "\n".join(logs[-20:]) if logs else "No logs captured"
                    yield json.dumps({
                        "event": "error",
                        "message": f"Backend did not start listening on port {backend_port} within 30 seconds.\n\nLogs:\n{log_content}"
                    }) + "\n"
                    backend_proc.terminate()
                    return

            # ----- Start frontend (if present) -----
            if frontend_dir:
                yield json.dumps({"event": "status", "message": "Preparing frontend..."}) + "\n"
                # Use full env so node/npm/nvm are on PATH; then override app vars
                frontend_env = dict(os.environ)
                frontend_env.update({
                    "PORT": str(frontend_port),
                    "BROWSER": "none",
                    "REACT_APP_BACKEND_URL": backend_url or "",
                    "VITE_BACKEND_URL": backend_url or "",
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
                # FIXED: Increase timeout for frontend compilation and add better error checking
                for attempt in range(60):  # 60 seconds for frontend (needs more time for compilation)
                    await asyncio.sleep(1)
                    
                    # Check if frontend process crashed
                    if frontend_proc.return_code() is not None:
                        # Brief wait so reader thread flushes stdout; then get full log
                        await asyncio.sleep(0.5)
                        _, lines = frontend_proc.get_logs(since=0)
                        all_logs = "\n".join(lines).strip() or "No output captured (process may have exited before npm produced output; check that node/npm are on PATH)."
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
                        break
                
                if not frontend_ready:
                    _, all_logs = frontend_proc.get_logs()
                    log_content = "\n".join(all_logs[-30:]) if all_logs else "No logs captured"
                    yield json.dumps({
                        "event": "warning",
                        "message": f"Frontend did not start listening within 60 seconds. It may still be compiling. Check http://localhost:{frontend_port} in a moment.\n\nRecent logs:\n{log_content}"
                    }) + "\n"
                    # FIXED: Don't return here - let it continue but warn user

            # FIXED: Only return success if everything is actually ready
            state = {
                "project_name": project_name,
                "backend_url": backend_url,
                "frontend_url": frontend_url,
                "backend": {"proc_id": backend_proc.id} if backend_proc else None,
                "frontend": {"proc_id": frontend_proc.id} if frontend_proc else None,
                "started_at": time.time(),
            }
            PROJECT_RUN_STATE[project_name] = state
            yield json.dumps({"event": "ready", "data": state, "message": "Application is running!"}) + "\n"

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            yield json.dumps({"event": "error", "message": f"{str(e)}\n\nTraceback:\n{tb}"}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")