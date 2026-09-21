"""
Verify generated FastAPI backends by running uvicorn and hitting sample API endpoints.

Similar to build_verify_service (frontend npm build loop), this:
1. pip install backend/requirements.txt
2. Starts uvicorn main:app on a free port
3. GET /health + sample CRUD routes from routes.py
4. Applies deterministic fixes, then LLM fix on failure
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

LogCallback = Optional[Callable[[str], None]]


def _log_line(logs: list[str], on_log: LogCallback, message: str) -> None:
    logs.append(message)
    if on_log:
        on_log(message)

from app_builder.schemas.files import GeneratedFiles
from app_builder.services.file_writer import file_writer
from app_builder.services.runtime_paths import resolve_project_root

logger = logging.getLogger("app_builder")

_STARTUP_TIMEOUT = int(os.environ.get("BACKEND_VERIFY_STARTUP_TIMEOUT", "25"))
_REQUEST_TIMEOUT = int(os.environ.get("BACKEND_VERIFY_REQUEST_TIMEOUT", "10"))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _backend_dir(project_name: str) -> Optional[str]:
    root = resolve_project_root(project_name)
    backend = os.path.join(root, "backend")
    if os.path.isfile(os.path.join(backend, "main.py")):
        return backend
    return None


def _capture_output(result: subprocess.CompletedProcess, cwd: str) -> str:
    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout.decode("utf-8", errors="replace"))
    if result.stderr:
        parts.append(result.stderr.decode("utf-8", errors="replace"))
    text = "\n".join(parts).strip()
    if len(text) > 12000:
        text = text[-12000:]
    return text or f"Command failed in {cwd} (exit {result.returncode})"


def _http_request(method: str, url: str, body: Optional[dict] = None) -> Tuple[int, str]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.getcode(), raw[:2000]
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        return exc.code, raw[:2000]
    except Exception as exc:
        return 0, str(exc)


def _parse_router_prefix(routes_content: str) -> Optional[str]:
    m = re.search(r"APIRouter\s*\(\s*prefix\s*=\s*['\"]([^'\"]+)['\"]", routes_content)
    if m:
        return m.group(1)
    if "APIRouter()" in routes_content:
        return ""
    return None


def _parse_create_payload(schemas_content: str) -> dict:
    """Build minimal POST body from first *Create schema in schemas.py."""
    m = re.search(r"class\s+\w+Create\s*\([^)]*\):\s*\n((?:\s+\w+[^\n]*\n)*)", schemas_content)
    if not m:
        return {"title": "__smoke_test__"}
    payload: dict = {}
    for line in m.group(1).splitlines():
        fm = re.match(r"\s+(\w+)\s*:\s*([^=\n]+)(?:\s*=\s*(.+))?", line)
        if not fm:
            continue
        name, typ, default = fm.group(1), fm.group(2).strip(), fm.group(3)
        if name == "pass" or name.startswith("_"):
            continue
        if default:
            default = default.strip()
            if default in ("True", "False"):
                payload[name] = default == "True"
            elif default.startswith('"') or default.startswith("'"):
                payload[name] = default.strip("\"'")
            elif default.isdigit():
                payload[name] = int(default)
            else:
                payload[name] = default
        elif "str" in typ:
            payload[name] = "__smoke_test__"
        elif "bool" in typ:
            payload[name] = False
        elif "int" in typ or "float" in typ:
            payload[name] = 0
        else:
            payload[name] = "__smoke_test__"
    if not payload:
        payload["title"] = "__smoke_test__"
    return payload


def discover_smoke_endpoints(files: Dict[str, str]) -> List[dict]:
    """Return HTTP calls to exercise after /health."""
    routes = files.get("backend/routes.py", "")
    schemas = files.get("backend/schemas.py", "")
    if not routes.strip() or "APIRouter" not in routes:
        return []

    prefix = _parse_router_prefix(routes)
    if prefix is None:
        return []

    base = prefix.rstrip("/") or ""
    endpoints: List[dict] = []

    if re.search(r'@router\.get\s*\(\s*["\']/?["\']', routes):
        endpoints.append({"method": "GET", "path": base or "/", "label": "list"})

    if re.search(r'@router\.post\s*\(\s*["\']/?["\']', routes):
        body = _parse_create_payload(schemas) if schemas else {"title": "__smoke_test__"}
        endpoints.append({"method": "POST", "path": base or "/", "body": body, "label": "create"})

    return endpoints


def apply_deterministic_backend_fixes(files: Dict[str, str]) -> Dict[str, str]:
    """Re-run backend post-process fixups before retry."""
    from app_builder.agents import dynamic_code_generator as dcg

    out = dict(files)
    dcg._fix_backend_routes_import(out)
    dcg._normalize_backend_requirements(out)
    dcg._ensure_sqlite_database_default(out)
    dcg._fix_backend_python_relative_imports(out)
    dcg._fix_sqlalchemy_uuid_imports(out)
    dcg._fix_backend_pydantic_and_common(out)
    dcg._ensure_database_tables_created(out)
    dcg._ensure_cors_in_backend(out)
    dcg._validate_and_fix_backend_imports(out)
    return out


def _backend_deps_available() -> bool:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        import sqlalchemy  # noqa: F401
        import pydantic  # noqa: F401
        return True
    except ImportError:
        return False


def _install_backend_deps(backend_dir: str, logs: list[str], on_log: LogCallback = None) -> Optional[str]:
    if _backend_deps_available():
        _log_line(logs, on_log, "[backend] Core deps already available — skipping pip install")
        return None

    req = os.path.join(backend_dir, "requirements.txt")
    if not os.path.isfile(req):
        _log_line(logs, on_log, "[backend] No requirements.txt — skipping pip install")
        return None

    venv_dir = os.path.join(backend_dir, ".verify_venv")
    python = sys.executable
    if not os.path.isdir(venv_dir):
        _log_line(logs, on_log, f"[backend] Creating verify venv at {venv_dir}")
        try:
            subprocess.run(
                [python, "-m", "venv", venv_dir],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except Exception as exc:
            return f"Could not create verify venv: {exc}"

    venv_python = os.path.join(venv_dir, "bin", "python")
    if not os.path.isfile(venv_python):
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    if not os.path.isfile(venv_python):
        venv_python = python

    _log_line(logs, on_log, f"[backend] pip install -r requirements.txt ({venv_python})")
    try:
        result = subprocess.run(
            [venv_python, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
            cwd=backend_dir,
            capture_output=True,
            timeout=int(os.environ.get("BACKEND_PIP_TIMEOUT", "120")),
        )
        out = _capture_output(result, backend_dir)
        if out:
            for line in out[-1500:].splitlines()[-8:]:
                if line.strip():
                    _log_line(logs, on_log, line.strip())
        if result.returncode != 0:
            return f"pip install failed (exit {result.returncode}): {out[-800:]}"
        return None
    except subprocess.TimeoutExpired:
        return "pip install timed out"
    except Exception as exc:
        return str(exc)


def _wait_for_server(
    base_url: str,
    proc: subprocess.Popen,
    logs: list[str],
    on_log: LogCallback = None,
) -> Optional[str]:
    deadline = time.time() + _STARTUP_TIMEOUT
    last_heartbeat = 0.0
    while time.time() < deadline:
        if proc.poll() is not None:
            err = (proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else "")
            out = (proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else "")
            combined = (err + "\n" + out).strip()
            return f"uvicorn exited early (code {proc.returncode}): {combined[-2000:]}"
        code, _ = _http_request("GET", f"{base_url}/health")
        if code == 200:
            _log_line(logs, on_log, "[backend] GET /health → 200")
            return None
        code, _ = _http_request("GET", f"{base_url}/docs")
        if code == 200:
            _log_line(logs, on_log, "[backend] GET /docs → 200 (no /health route)")
            return None
        now = time.time()
        if now - last_heartbeat >= 2.0:
            remaining = max(0, int(deadline - now))
            _log_line(logs, on_log, f"[backend] Waiting for uvicorn to respond... ({remaining}s left)")
            last_heartbeat = now
        time.sleep(0.4)
    return f"Backend did not respond within {_STARTUP_TIMEOUT}s"


def run_backend_api_smoke(
    project_name: str,
    files: Optional[Dict[str, str]] = None,
    on_log: LogCallback = None,
) -> Tuple[bool, str, str]:
    """
    Start generated backend with uvicorn and hit /health + sample routes.
    Returns (success, error_message, log_output).
    """
    backend_dir = _backend_dir(project_name)
    if not backend_dir:
        return True, "", "No backend/main.py — backend verify skipped"

    file_map = files or {}
    logs: list[str] = []
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    _log_line(logs, on_log, f"[backend] Preparing smoke test on port {port}")
    pip_err = _install_backend_deps(backend_dir, logs, on_log)
    if pip_err:
        return False, pip_err, "\n".join(logs)

    venv_python = os.path.join(backend_dir, ".verify_venv", "bin", "python")
    if not os.path.isfile(venv_python):
        venv_python = os.path.join(backend_dir, ".verify_venv", "Scripts", "python.exe")
    runner = venv_python if os.path.isfile(venv_python) else sys.executable

    _log_line(logs, on_log, f"[backend] Starting uvicorn main:app --port {port}")
    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(
            [runner, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        startup_err = _wait_for_server(base_url, proc, logs, on_log)
        if startup_err:
            return False, startup_err, "\n".join(logs)

        endpoints = discover_smoke_endpoints(file_map)
        if not endpoints and os.path.isfile(os.path.join(backend_dir, "routes.py")):
            with open(os.path.join(backend_dir, "routes.py"), encoding="utf-8") as fh:
                endpoints = discover_smoke_endpoints({"backend/routes.py": fh.read()})

        if endpoints:
            _log_line(logs, on_log, f"[backend] Testing {len(endpoints)} sample API endpoint(s)...")
        else:
            _log_line(logs, on_log, "[backend] No CRUD routes found — health check only")

        created_id: Optional[int] = None
        for ep in endpoints:
            path = ep["path"]
            if not path.startswith("/"):
                path = f"/{path}"
            url = f"{base_url}{path}"
            method = ep["method"]
            body = ep.get("body")
            label = ep.get("label", method.lower())
            _log_line(logs, on_log, f"[backend] Calling {method} {path} ({label})...")
            code, raw = _http_request(method, url, body)
            _log_line(logs, on_log, f"[backend] {method} {path} → HTTP {code}")
            if code == 0 or code >= 500:
                return False, f"{method} {path} failed: HTTP {code} — {raw[:500]}", "\n".join(logs)
            if method == "POST" and 200 <= code < 300:
                try:
                    data = json.loads(raw)
                    if isinstance(data, dict) and "id" in data:
                        created_id = int(data["id"])
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

        if created_id is not None:
            coll_path = endpoints[-1]["path"] if endpoints else ""
            if not coll_path.startswith("/"):
                coll_path = f"/{coll_path}"
            del_url = f"{base_url}{coll_path.rstrip('/')}/{created_id}"
            _log_line(logs, on_log, f"[backend] Cleaning up test row DELETE {coll_path}/{created_id}...")
            del_code, _ = _http_request("DELETE", del_url)
            _log_line(logs, on_log, f"[backend] DELETE {coll_path}/{created_id} → HTTP {del_code}")
            if del_code not in (0, 200, 204, 404):
                return False, f"DELETE cleanup failed: HTTP {del_code}", "\n".join(logs)

        _log_line(logs, on_log, "[backend] API smoke test passed ✓")
        return True, "", "\n".join(logs)
    except FileNotFoundError:
        return False, "Python or uvicorn not available", "\n".join(logs)
    except Exception as exc:
        logger.warning("[backend_verify] smoke test error: %s", exc)
        return False, str(exc), "\n".join(logs)
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _pick_backend_files_for_fix(files: Dict[str, str], log: str) -> Dict[str, str]:
    priority = [
        "backend/main.py",
        "backend/routes.py",
        "backend/models.py",
        "backend/schemas.py",
        "backend/database.py",
        "backend/requirements.txt",
    ]
    out: Dict[str, str] = {}
    for path in priority:
        if path in files:
            out[path] = files[path]
    for path in sorted(files.keys()):
        if path.startswith("backend/") and path.endswith(".py") and path not in out:
            out[path] = files[path]
    return out


async def _llm_fix_backend(
    files: Dict[str, str],
    error_log: str,
    user_email: Optional[str],
) -> Dict[str, str]:
    from llm_helper import get_llm_for_user

    subset = _pick_backend_files_for_fix(files, error_log)
    snippets = [f"--- {path} ---\n{content[:4500]}" for path, content in subset.items()]

    prompt = f"""You are fixing a generated FastAPI backend that failed to start or respond to API smoke tests.

ERROR LOG:
```
{error_log[-6500:]}
```

CURRENT FILES:
{chr(10).join(snippets)}

Rules:
1. Backend runs as `uvicorn main:app` from the backend/ directory — use absolute imports (from routes import router), NOT relative imports.
2. main.py must include router from routes.py and call Base.metadata.create_all on startup if using SQLAlchemy.
3. Use SQLite (sqlite:///./app.db) — no PostgreSQL credentials.
4. Pydantic v2: use model_config = ConfigDict(from_attributes=True), .model_dump() not .dict().
5. routes.py should export `router = APIRouter(...)` matching what main.py imports.
6. Include GET /health returning {{"status": "ok"}}.
7. Return ONLY changed files as valid JSON: {{"files": {{"backend/path.py": "full file content"}}}}
8. Return complete file contents — do not truncate."""

    llm = get_llm_for_user(user_email, temperature=0.15)
    resp = await llm.ainvoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(text, list):
        text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)

    match = re.search(r"\{[\s\S]*\"files\"[\s\S]*\}", text)
    if not match:
        return files
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
            data = json.loads(repair_json(match.group(0)))
        except Exception:
            return files

    updated = dict(files)
    for path, content in (data.get("files") or {}).items():
        if isinstance(content, str) and content.strip():
            updated[path.replace("\\", "/")] = content
    return apply_deterministic_backend_fixes(updated)


async def _run_smoke_with_live_logs(
    project_name: str,
    files: Dict[str, str],
    on_event: Optional[Callable[[dict], Any]],
) -> Tuple[bool, str, str]:
    """Run blocking smoke test in a thread while streaming log lines to the UI."""
    import asyncio

    loop = asyncio.get_running_loop()
    log_queue: asyncio.Queue[str] = asyncio.Queue()

    def sync_log(message: str) -> None:
        loop.call_soon_threadsafe(log_queue.put_nowait, message)

    async def emit_log(message: str) -> None:
        if on_event:
            await on_event({"event": "backend_log", "message": message})

    async def pump_logs(task: asyncio.Task) -> None:
        while not task.done() or not log_queue.empty():
            try:
                message = await asyncio.wait_for(log_queue.get(), timeout=0.35)
                await emit_log(message)
            except asyncio.TimeoutError:
                if task.done():
                    break

    task = asyncio.create_task(
        asyncio.to_thread(run_backend_api_smoke, project_name, files, sync_log)
    )
    pump = asyncio.create_task(pump_logs(task))
    try:
        ok, err, log = await task
    finally:
        await pump
    return ok, err, log


async def verify_backend_and_fix(
    project_name: str,
    files: Dict[str, str],
    user_email: Optional[str],
    on_event: Optional[Callable[[dict], Any]] = None,
    max_attempts: int = 3,
) -> Tuple[Dict[str, str], bool, str]:
    """
    Write files, run uvicorn smoke test, auto-fix on failure.
    Returns (updated_files, success, last_log).
    """
    if "backend/main.py" not in files:
        return files, True, "No backend — verify skipped"

    async def emit(event: str, message: str, **extra):
        payload = {"event": event, "message": message, **extra}
        if on_event:
            await on_event(payload)

    await emit("backend_log", "Applying backend import/route fixes...")
    files = apply_deterministic_backend_fixes(files)
    file_writer(project_name, GeneratedFiles(files=files))
    await emit("backend_log", "Backend files written to disk")

    last_log = ""
    for attempt in range(1, max_attempts + 1):
        await emit(
            "backend_start",
            f"Attempt {attempt}/{max_attempts}: install deps → start uvicorn → test sample APIs",
            attempt=attempt,
        )
        ok, err, log = await _run_smoke_with_live_logs(project_name, files, on_event)
        last_log = log or err or ""

        if ok:
            await emit("backend_success", "Backend API verified — uvicorn + sample endpoints OK.")
            return files, True, last_log

        await emit("backend_failed", err or "Backend smoke test failed", log=last_log[-2000:], attempt=attempt)
        if attempt >= max_attempts:
            break

        await emit(
            "backend_fix_attempt",
            f"Attempt {attempt}/{max_attempts} failed — applying deterministic fixes...",
            attempt=attempt,
        )
        files = apply_deterministic_backend_fixes(files)
        file_writer(project_name, GeneratedFiles(files=files))
        ok2, _, log2 = await _run_smoke_with_live_logs(project_name, files, on_event)
        if ok2:
            last_log = log2
            await emit("backend_success", "Backend fixed with deterministic patches.")
            return files, True, last_log

        await emit("backend_fix_attempt", "Deterministic fixes insufficient — AI is fixing backend code...")
        try:
            files = await _llm_fix_backend(files, (log2 or err or last_log), user_email)
            file_writer(project_name, GeneratedFiles(files=files))
            await emit("backend_log", "AI fix applied — retrying smoke test...")
        except Exception as fix_err:
            logger.warning("[backend_verify] LLM fix failed: %s", fix_err)
            await emit("backend_fix_attempt", f"Auto-fix error: {fix_err}")

    return files, False, last_log
