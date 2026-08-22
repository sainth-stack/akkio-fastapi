"""Tests for single base-vite-fastapi scaffold."""
import os
import subprocess
import tempfile

from app_builder.services.scaffold_service import (
    BASE_TEMPLATE_NAME,
    detect_template,
    get_base_scaffold_files,
    get_template_code_files,
    load_template,
    merge_llm_into_base,
    extract_api_contract,
    validate_generated_contract,
)
from app_builder.services.code_post_process import post_process_generated_files, should_use_template


def test_always_use_base_template():
    assert should_use_template("todo", "prd text", {"tables": []}) is True
    assert detect_template("anything") == BASE_TEMPLATE_NAME


def test_base_scaffold_has_vite_files():
    files = get_base_scaffold_files()
    assert "frontend/vite.config.js" in files
    assert "frontend/package.json" in files
    assert "vite" in files["frontend/package.json"].lower()
    assert "frontend/src/App.jsx" in files
    assert "frontend/src/api/client.js" in files
    assert len(files) >= 12


def test_merge_preserves_frozen_files():
    base = get_base_scaffold_files()
    llm = {
        "frontend/package.json": '{"name":"hacked"}',
        "frontend/src/App.jsx": "export default function App(){return <div>Hi</div>}",
    }
    merged = merge_llm_into_base(base, llm)
    assert "hacked" not in merged["frontend/package.json"]
    # Stub App without apiFetch is rejected — base App.jsx kept
    assert "apiFetch" in merged["frontend/src/App.jsx"]


def test_merge_rejects_empty_backend():
    base = get_base_scaffold_files()
    llm = {
        "backend/routes.py": "",
        "backend/models.py": "   ",
        "frontend/src/App.jsx": base["frontend/src/App.jsx"],
    }
    merged = merge_llm_into_base(base, llm)
    assert merged["backend/routes.py"].strip()
    assert "APIRouter" in merged["backend/routes.py"]


def test_extract_api_contract_prefers_tasks():
    arch = {
        "database_schema": {
            "tables": [
                {"name": "users", "columns": [{"name": "id"}]},
                {"name": "tasks", "columns": [{"name": "id"}, {"name": "title"}]},
            ]
        }
    }
    contract = extract_api_contract(arch)
    assert contract["table_name"] == "tasks"
    assert contract["api_prefix"] == "/tasks"


def test_adapt_scaffold_to_tasks():
    from app_builder.services.scaffold_service import adapt_scaffold_to_contract
    base = get_base_scaffold_files()
    contract = {"api_prefix": "/tasks", "table_name": "tasks", "entity": "task"}
    adapted = adapt_scaffold_to_contract(base, contract)
    assert "/tasks" in adapted["frontend/src/App.jsx"]
    assert 'prefix="/tasks"' in adapted["backend/routes.py"] or '"/tasks"' in adapted["backend/routes.py"]


def test_post_process_includes_full_scaffold():
    llm_only = {"frontend/src/App.jsx": "export default function App(){return <div>x</div>}"}
    out = post_process_generated_files(llm_only, architecture={}, uiux="")
    assert "frontend/vite.config.js" in out
    assert "backend/main.py" in out


def test_vite_build_from_scaffold():
    files = get_base_scaffold_files()
    processed = post_process_generated_files(dict(files), architecture={}, uiux="")
    with tempfile.TemporaryDirectory() as tmp:
        for rel, content in processed.items():
            full = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
        frontend = os.path.join(tmp, "frontend")
        subprocess.run(["npm", "install", "--legacy-peer-deps"], cwd=frontend, check=True, capture_output=True)
        env = os.environ.copy()
        env["VITE_BASE_PATH"] = "/app/verify_test/"
        r = subprocess.run(["npm", "run", "build"], cwd=frontend, env=env, capture_output=True)
        assert r.returncode == 0, r.stderr.decode()
        assert os.path.isdir(os.path.join(frontend, "dist"))


def test_template_code_files_alias():
    assert get_template_code_files() is not None
    assert load_template()["name"] == "base-vite-fastapi"
