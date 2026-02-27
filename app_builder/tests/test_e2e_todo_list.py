"""
End-to-end test for todo-list app builder flow.

Creates a todo-list project from template, runs backend server, and verifies basic API flow.

Run (from akkio-fastapi directory):

  # Option 1: Use the script (creates venv + installs deps if needed)
  ./scripts/run_e2e_todo_test.sh

  # Option 2: With existing venv (same env as uvicorn)
  source venv/bin/activate   # or: source .venv/bin/activate
  pip install pytest
  python app_builder/tests/test_e2e_todo_list.py

  # Option 3: With pytest
  python -m pytest app_builder/tests/test_e2e_todo_list.py -v -s
"""
import asyncio
import os
import shutil
import subprocess
import sys
import time

# Add akkio-fastapi to path
_current = os.path.dirname(os.path.abspath(__file__))
_app_builder = os.path.dirname(_current)
_akkio_fastapi = os.path.dirname(_app_builder)
if _akkio_fastapi not in sys.path:
    sys.path.insert(0, _akkio_fastapi)

# Load .env from project root so OPENAI_API_KEY is available for LLM tests
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(_akkio_fastapi, ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass

try:
    import pytest
except ImportError:
    pytest = None

from app_builder.services.template_service import get_template_code_files
from app_builder.services.file_writer import file_writer
from app_builder.services.runtime_paths import get_projects_dir
from app_builder.agents.validation_agent import polish_template_output
from app_builder.schemas.files import GeneratedFiles


TEST_PROJECT = "e2e-todo-test"
BACKEND_PORT = 5098


def _project_dir():
    return os.path.join(get_projects_dir(), TEST_PROJECT)


def _backend_dir():
    return os.path.join(_project_dir(), "backend")


def _kill_port(port: int):
    try:
        out = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.stdout.strip():
            for pid in out.stdout.strip().split():
                try:
                    os.kill(int(pid), 9)
                    time.sleep(0.5)
                except (ProcessLookupError, ValueError):
                    pass
    except Exception:
        pass


def _wait_for_backend(url: str, timeout: int = 30) -> bool:
    try:
        import urllib.request
        for _ in range(timeout):
            try:
                with urllib.request.urlopen(f"{url}/todos", timeout=2) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(1)
    except Exception:
        pass
    return False


@pytest.fixture(autouse=True)
def cleanup():
    yield
    _kill_port(BACKEND_PORT)


@pytest.fixture
def todo_project():
    """Create todo-list project from template."""
    project_dir = _project_dir()
    backend_dir = _backend_dir()

    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)

    files = get_template_code_files("todo-list")
    assert files, "Todo-list template not found"
    polished = polish_template_output(files, "todo-list")
    file_writer(TEST_PROJECT, GeneratedFiles(files=polished))

    assert os.path.exists(backend_dir)
    assert os.path.exists(os.path.join(backend_dir, "main.py"))
    assert os.path.exists(os.path.join(backend_dir, "requirements.txt"))

    return project_dir


def test_e2e_todo_create_and_run_backend(todo_project):
    """Create todo project, run backend, verify it starts."""
    backend_dir = _backend_dir()
    _kill_port(BACKEND_PORT)

    # Setup venv and install deps
    venv_python = os.path.join(backend_dir, "venv", "bin", "python")
    venv_pip = os.path.join(backend_dir, "venv", "bin", "pip")
    if not os.path.exists(venv_python):
        subprocess.run(
            [sys.executable, "-m", "venv", "venv"],
            cwd=backend_dir,
            capture_output=True,
            check=True,
        )
    subprocess.run(
        [venv_pip, "install", "-q", "-r", "requirements.txt"],
        cwd=backend_dir,
        capture_output=True,
        check=True,
        timeout=60,
    )

    # Start uvicorn
    proc = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "DATABASE_URL": "sqlite:///./app.db"},
    )

    try:
        url = f"http://127.0.0.1:{BACKEND_PORT}"
        assert _wait_for_backend(url), "Backend did not start in time"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_e2e_todo_api_crud(todo_project):
    """Create todo project, run backend, test CRUD operations."""
    backend_dir = _backend_dir()
    _kill_port(BACKEND_PORT)

    venv_python = os.path.join(backend_dir, "venv", "bin", "python")
    venv_pip = os.path.join(backend_dir, "venv", "bin", "pip")
    if not os.path.exists(venv_python):
        subprocess.run([sys.executable, "-m", "venv", "venv"], cwd=backend_dir, capture_output=True, check=True)
    subprocess.run([venv_pip, "install", "-q", "-r", "requirements.txt"], cwd=backend_dir, capture_output=True, check=True, timeout=60)

    proc = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "DATABASE_URL": "sqlite:///./app.db"},
    )

    try:
        url = f"http://127.0.0.1:{BACKEND_PORT}"
        assert _wait_for_backend(url), "Backend did not start"

        import urllib.request
        import json

        # GET /todos - should be empty
        with urllib.request.urlopen(f"{url}/todos") as r:
            assert r.status == 200
            data = json.loads(r.read().decode())
            assert data == []

        # POST /todos - create
        req = urllib.request.Request(
            f"{url}/todos",
            data=json.dumps({"title": "Test task", "description": "E2E test", "completed": False, "priority": "medium"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            assert r.status == 200
            created = json.loads(r.read().decode())
            assert created["title"] == "Test task"
            assert created["id"] is not None
            todo_id = created["id"]

        # GET /todos - should have one
        with urllib.request.urlopen(f"{url}/todos") as r:
            data = json.loads(r.read().decode())
            assert len(data) == 1
            assert data[0]["title"] == "Test task"

        # GET /todos/{id}
        with urllib.request.urlopen(f"{url}/todos/{todo_id}") as r:
            one = json.loads(r.read().decode())
            assert one["id"] == todo_id
            assert one["title"] == "Test task"

        # PUT /todos/{id} - update
        req = urllib.request.Request(
            f"{url}/todos/{todo_id}",
            data=json.dumps({"title": "Updated", "description": "Done", "completed": True, "priority": "high"}).encode(),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(req) as r:
            updated = json.loads(r.read().decode())
            assert updated["title"] == "Updated"
            assert updated["completed"] is True

        # DELETE /todos/{id}
        req = urllib.request.Request(f"{url}/todos/{todo_id}", method="DELETE")
        with urllib.request.urlopen(req) as r:
            assert r.status == 200

        # GET /todos - should be empty again
        with urllib.request.urlopen(f"{url}/todos") as r:
            data = json.loads(r.read().decode())
            assert data == []

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_e2e_todo_full_graph():
    """Full graph flow: requirement -> generate -> write -> verify files."""
    if os.getenv("SKIP_LLM_TESTS") == "1":
        pytest.skip("SKIP_LLM_TESTS=1 - skipping LLM-dependent test")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping LLM-dependent test")

    import asyncio
    from app_builder.graph.builder_graph import app_builder_graph
    from app_builder.schemas.requirements import UserRequirement

    project_name = "e2e-todo-graph-test"
    project_dir = os.path.join(get_projects_dir(), project_name)
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)

    async def run():
        initial = {
            "user_requirement": UserRequirement(description="Create a simple todo list app to track tasks"),
            "project_name": "",
            "template_name": "",
            "structured_requirement": {},
            "architecture": {},
            "api_contract": {},
            "db_schema": {},
            "generated_files": {},
            "validation_results": {},
            "error": "",
        }
        return await app_builder_graph.ainvoke(initial)

    result = asyncio.run(run())

    assert not result.get("error"), f"Graph error: {result.get('error')}"
    assert result.get("generated_files")
    files = result["generated_files"]
    if hasattr(files, "files"):
        files = files.files
    assert "backend/main.py" in files
    assert "backend/requirements.txt" in files
    assert "frontend" in str(files.keys())

    assert os.path.exists(project_dir)
    assert os.path.exists(os.path.join(project_dir, "backend", "main.py"))


if __name__ == "__main__":
    # Run as script: cd akkio-fastapi && python app_builder/tests/test_e2e_todo_list.py
    # Run with pytest: python -m pytest app_builder/tests/test_e2e_todo_list.py -v -s
    if pytest and "--pytest" in sys.argv:
        sys.argv.remove("--pytest")
        pytest.main([__file__, "-v", "-s"])
    else:
        def _run():
            print("Creating todo project...")
            project_dir = _project_dir()
            if os.path.exists(project_dir):
                shutil.rmtree(project_dir)
            files = get_template_code_files("todo-list")
            assert files, "Todo-list template not found"
            polished = polish_template_output(files, "todo-list")
            file_writer(TEST_PROJECT, GeneratedFiles(files=polished))
            print("Project created.")

            backend_dir = _backend_dir()
            _kill_port(BACKEND_PORT)

            venv_python = os.path.join(backend_dir, "venv", "bin", "python")
            venv_pip = os.path.join(backend_dir, "venv", "bin", "pip")
            print("Setting up venv...")
            if not os.path.exists(venv_python):
                subprocess.run([sys.executable, "-m", "venv", "venv"], cwd=backend_dir, capture_output=True, check=True)
            subprocess.run([venv_pip, "install", "-q", "-r", "requirements.txt"], cwd=backend_dir, capture_output=True, check=True, timeout=60)
            print("Starting backend...")
            proc = subprocess.Popen(
                [venv_python, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
                cwd=backend_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env={**os.environ, "DATABASE_URL": "sqlite:///./app.db"},
            )
            try:
                url = f"http://127.0.0.1:{BACKEND_PORT}"
                assert _wait_for_backend(url), "Backend did not start"
                print("Backend ready. Testing API...")

                import urllib.request
                import json

                with urllib.request.urlopen(f"{url}/todos") as r:
                    assert r.status == 200
                    assert json.loads(r.read().decode()) == []

                req = urllib.request.Request(f"{url}/todos",
                    data=json.dumps({"title": "E2E task", "description": "test", "completed": False, "priority": "medium"}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req) as r:
                    created = json.loads(r.read().decode())
                    todo_id = created["id"]

                with urllib.request.urlopen(f"{url}/todos") as r:
                    assert len(json.loads(r.read().decode())) == 1

                req = urllib.request.Request(f"{url}/todos/{todo_id}", method="DELETE")
                urllib.request.urlopen(req)

                print("All tests passed!")
            finally:
                proc.terminate()
                proc.wait(timeout=5)

        _run()
