"""Tests for static quiz generator and app spec."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app_builder.services.app_spec_service import build_app_spec, detect_app_kind, validate_against_app_spec
from app_builder.services.static_app_generators import (
    _parse_static_questions,
    generate_static_quiz_app_jsx,
    apply_deterministic_fallback,
)
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

SAMPLE_PRD = """
No network calls are required. Content is static and embedded.
5 static multiple-choice questions.
"""


def test_detect_static_quiz():
    kind = detect_app_kind("create mcq quiz app", SAMPLE_PRD, {}, SAMPLE_UIUX)
    assert kind == "static_quiz"


def test_parse_questions():
    qs = _parse_static_questions(SAMPLE_UIUX, "")
    assert len(qs) >= 1
    assert "France" in qs[0]["text"] or "capital" in qs[0]["text"].lower()


def test_static_quiz_generator_validates():
    spec = build_app_spec(
        "create multiple choice questions app",
        {"database_schema": {"tables": [{"name": "questions"}]}},
        SAMPLE_PRD,
        SAMPLE_UIUX,
    )
    assert spec["app_kind"] == "static_quiz"
    assert spec["frontend_only"] is True

    files = apply_deterministic_fallback({}, spec, uiux=SAMPLE_UIUX, prd=SAMPLE_PRD)
    files = post_process_generated_files(
        files, {}, requirement="quiz", prd=SAMPLE_PRD, uiux=SAMPLE_UIUX, app_spec=spec
    )
    errors = validate_against_app_spec(files, spec)
    assert errors == [], errors


def test_ensure_valid_applies_fallback_on_empty_llm():
    spec = build_app_spec("quiz app", {}, SAMPLE_PRD, SAMPLE_UIUX)
    base = post_process_generated_files({}, {}, requirement="quiz", prd=SAMPLE_PRD, uiux=SAMPLE_UIUX, app_spec=spec)
    out, errs = ensure_valid_codegen_output(
        base, {}, uiux=SAMPLE_UIUX, prd=SAMPLE_PRD, requirement="quiz app", app_spec=spec
    )
    assert not errs, errs
    assert "QUESTIONS" in out.get("frontend/src/App.jsx", "")
    assert "shell-notice" not in out.get("frontend/src/App.jsx", "").lower()


if __name__ == "__main__":
    test_detect_static_quiz()
    test_parse_questions()
    test_static_quiz_generator_validates()
    test_ensure_valid_applies_fallback_on_empty_llm()
    print("ALL OK")
