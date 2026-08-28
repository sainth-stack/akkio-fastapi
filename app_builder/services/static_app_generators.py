"""Backward-compatible re-exports — prefer app_generators for new code."""
from app_builder.services.app_generators import (
    _parse_static_questions,
    apply_deterministic_fallback,
    generate_minimal_backend_stubs,
    generate_static_quiz_app_jsx,
    generate_static_quiz_css,
)

__all__ = [
    "_parse_static_questions",
    "apply_deterministic_fallback",
    "generate_minimal_backend_stubs",
    "generate_static_quiz_app_jsx",
    "generate_static_quiz_css",
]
