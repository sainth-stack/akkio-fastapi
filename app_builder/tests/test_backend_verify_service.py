"""Tests for backend API smoke verification."""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app_builder.services.scaffold_service import get_base_scaffold_files
from app_builder.services.file_writer import file_writer
from app_builder.schemas.files import GeneratedFiles
from app_builder.services.app_generators import generate_crud_backend
from api.app_creator.backend_verify_service import (
    discover_smoke_endpoints,
    apply_deterministic_backend_fixes,
    run_backend_api_smoke,
)


def test_discover_smoke_endpoints_from_crud_routes():
    spec = {"primary_table": "tasks", "fields": ["title", "completed"]}
    backend = generate_crud_backend(spec)
    eps = discover_smoke_endpoints(backend)
    methods = {ep["method"] for ep in eps}
    assert "GET" in methods
    assert "POST" in methods
    assert any(ep["path"].endswith("tasks") or ep["path"] == "/tasks" for ep in eps)


def test_base_scaffold_backend_starts():
    """Base scaffold backend should pass uvicorn + /health smoke test."""
    files = get_base_scaffold_files()
    assert files, "base scaffold missing"
    files = apply_deterministic_backend_fixes(files)

    tmp = tempfile.mkdtemp(prefix="akkio_backend_verify_")
    try:
        os.environ["APP_BUILDER_RUNTIME_DIR"] = tmp
        project = "scaffold_smoke_test"
        file_writer(project, GeneratedFiles(files=files))
        ok, err, log = run_backend_api_smoke(project, files)
        assert ok, f"Backend smoke failed: {err}\n{log}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop("APP_BUILDER_RUNTIME_DIR", None)


if __name__ == "__main__":
    test_discover_smoke_endpoints_from_crud_routes()
    test_base_scaffold_backend_starts()
    print("ALL OK")
