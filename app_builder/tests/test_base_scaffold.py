"""Tests for generic base-vite-fastapi scaffold and App Spec."""
import os
import subprocess
import tempfile

from app_builder.services.app_spec_service import (
    build_app_spec,
    build_codegen_system_prompt,
    detect_app_kind,
    singularize,
    validate_against_app_spec,
)
from app_builder.services.scaffold_service import (
    BASE_TEMPLATE_NAME,
    detect_template,
    get_base_scaffold_files,
    get_template_code_files,
    load_template,
    merge_llm_into_base,
)
from app_builder.services.code_post_process import post_process_generated_files, should_use_template


def test_always_use_base_template():
    assert should_use_template("quiz", "prd text", {"tables": []}) is True
    assert detect_template("anything") == BASE_TEMPLATE_NAME


def test_generic_base_has_no_domain_crud():
    files = get_base_scaffold_files()
    app = files["frontend/src/App.jsx"]
    assert "apiFetch" not in app
    assert "shell-notice" in app or "generic" in app.lower()
    assert "completed" not in app or "filter" not in app
    assert files["backend/routes.py"].strip().endswith("router = APIRouter()") or "APIRouter()" in files["backend/routes.py"]
    assert "class Item" not in files.get("backend/models.py", "")


def test_detect_app_kind():
    assert detect_app_kind("create a quiz app", "") == "quiz"
    assert detect_app_kind("create todo list", "") == "crud"
    assert detect_app_kind("build travel planner", "") == "crud"


def test_singularize():
    assert singularize("quizzes") == "quiz"
    assert singularize("tasks") == "task"
    assert singularize("categories") == "category"


def test_app_spec_quiz():
    arch = {
        "database_schema": {
            "tables": [
                {"name": "users", "columns": [{"name": "id"}]},
                {"name": "quizzes", "columns": [{"name": "id"}, {"name": "title"}]},
                {"name": "questions", "columns": [{"name": "id"}]},
                {"name": "options", "columns": [{"name": "id"}]},
            ]
        }
    }
    spec = build_app_spec("create mcq quiz app", arch)
    assert spec["app_kind"] == "quiz"
    assert spec["primary_table"] == "quizzes"
    assert spec["api_prefix"] == "/quizzes"
    assert "questions" in spec["mvp_tables"]


def test_merge_rejects_shell_app():
    base = get_base_scaffold_files()
    llm = {"frontend/src/App.jsx": base["frontend/src/App.jsx"]}
    merged = merge_llm_into_base(base, llm, app_spec=build_app_spec("quiz", {}))
    assert "shell-notice" in merged["frontend/src/App.jsx"]


def test_post_process_includes_scaffold():
    out = post_process_generated_files({}, architecture={}, requirement="quiz app")
    assert "frontend/vite.config.js" in out
    assert "backend/main.py" in out


def test_validate_rejects_generic_shell():
    files = get_base_scaffold_files()
    spec = build_app_spec("create quiz app", {
        "database_schema": {"tables": [{"name": "quizzes", "columns": []}]}
    })
    errors = validate_against_app_spec(files, spec)
    assert any("shell" in e.lower() or "quiz" in e.lower() for e in errors)


def test_codegen_prompt_mentions_quiz():
    spec = build_app_spec("quiz app", {"database_schema": {"tables": [{"name": "quizzes"}]}})
    prompt = build_codegen_system_prompt(spec, "", {})
    assert "quiz" in prompt.lower()
    assert "task list" not in prompt.lower()


def test_vite_build_from_generic_scaffold():
    files = get_base_scaffold_files()
    processed = post_process_generated_files(dict(files), architecture={}, requirement="test")
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
    assert load_template()["kind"] == "generic"
