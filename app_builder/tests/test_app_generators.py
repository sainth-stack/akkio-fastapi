"""Tests for deterministic app generators (all app kinds)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app_builder.services.app_spec_service import build_app_spec, validate_against_app_spec
from app_builder.services.app_generators import apply_deterministic_fallback
from app_builder.services.code_post_process import ensure_valid_codegen_output, post_process_generated_files


SAMPLE_UIUX = """
1) Question 1
- What is the capital of France?
 - A) Berlin
 - B) Madrid
 - C) Paris
 - D) Rome
- Correct: C (Paris)
"""

SAMPLE_PRD_STATIC = """
No network calls are required. Content is static and embedded.
5 static multiple-choice questions.
"""


def _finalize(files, spec, requirement="", prd="", uiux=""):
    return post_process_generated_files(
        files, {}, requirement=requirement, prd=prd, uiux=uiux, app_spec=spec
    )


def test_crud_todo_fallback():
    spec = build_app_spec(
        "create a todo list app",
        {"database_schema": {"tables": [{"name": "tasks", "columns": [{"name": "title"}, {"name": "completed"}]}]}},
        "Task manager with add, delete, filter",
    )
    assert spec["app_kind"] == "crud"
    files = apply_deterministic_fallback({}, spec)
    files = _finalize(files, spec, requirement="todo", prd="Task manager")
    errors = validate_against_app_spec(files, spec)
    assert errors == [], errors
    assert "apiFetch" in files.get("frontend/src/App.jsx", "")
    assert "class Task" in files.get("backend/models.py", "")


def test_llm_generator_fallback():
    spec = build_app_spec(
        "linkedin post idea generator",
        {"database_schema": {"tables": [{"name": "ideas"}]}},
        "Generate ideas from a topic",
    )
    assert spec["app_kind"] == "llm"
    assert spec.get("gen_type") in ("linkedin_post", "ideas")
    files = apply_deterministic_fallback({}, spec)
    files = _finalize(files, spec, requirement="idea generator")
    errors = validate_against_app_spec(files, spec)
    assert errors == [], errors
    assert "apiFetch('/generate'" in files.get("frontend/src/App.jsx", "")


def test_llm_summarize_fallback():
    spec = build_app_spec(
        "Build a text summarization app",
        {},
        "Summarize long text",
    )
    assert spec["app_kind"] == "llm"
    assert spec.get("gen_type") == "summarize"
    files = apply_deterministic_fallback({}, spec)
    jsx = files.get("frontend/src/App.jsx", "")
    assert "summarize" in jsx
    assert "textarea" in jsx


def test_form_translator_fallback():
    spec = build_app_spec(
        "simple translator app",
        {"database_schema": {"tables": [{"name": "translations"}]}},
        "Translate text",
    )
    assert spec["app_kind"] == "llm"
    assert spec.get("gen_type") == "translate"
    files = apply_deterministic_fallback({}, spec)
    files = _finalize(files, spec, requirement="translator")
    errors = validate_against_app_spec(files, spec)
    assert errors == [], errors


def test_dashboard_fallback():
    spec = build_app_spec(
        "analytics dashboard with metrics",
        {"database_schema": {"tables": [{"name": "items"}]}},
        "Show totals and recent items",
    )
    assert spec["app_kind"] == "dashboard"
    files = apply_deterministic_fallback({}, spec)
    files = _finalize(files, spec, requirement="dashboard")
    errors = validate_against_app_spec(files, spec)
    assert errors == [], errors


def test_travel_planner_llm_fallback():
    spec = build_app_spec(
        "travel trip planner",
        {"database_schema": {"tables": [{"name": "trips", "columns": [{"name": "title"}]}]}},
        "Plan trips",
    )
    assert spec["app_kind"] == "llm"
    assert spec.get("gen_type") == "travel"
    base = post_process_generated_files({}, {}, requirement="travel", app_spec=spec)
    out, errs = ensure_valid_codegen_output({}, {}, requirement="travel", app_spec=spec)
    assert not errs, errs
    assert "shell-notice" not in out.get("frontend/src/App.jsx", "").lower()


if __name__ == "__main__":
    test_crud_todo_fallback()
    test_llm_generator_fallback()
    test_llm_summarize_fallback()
    test_form_translator_fallback()
    test_dashboard_fallback()
    test_travel_planner_llm_fallback()
    print("ALL OK")
