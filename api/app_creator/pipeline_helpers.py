"""Shared helpers for App Builder pipeline status and job tracking."""

from __future__ import annotations

from typing import Any, Optional

from db.app_builder import get_app_builder_db

db = get_app_builder_db()

STEP_PIPELINE = {
    "prd": ("PRD_RUNNING", "PRD_COMPLETE", "PRD_FAILED"),
    "uiux": ("UIUX_RUNNING", "UIUX_COMPLETE", "UIUX_FAILED"),
    "style": ("STYLE_RUNNING", "STYLE_COMPLETE", "STYLE_FAILED"),
    "architecture": ("ARCHITECTURE_RUNNING", "ARCHITECTURE_COMPLETE", "ARCHITECTURE_FAILED"),
    "plan": ("PLAN_RUNNING", "PLAN_COMPLETE", "PLAN_FAILED"),
}

STEP_COMPLETE_EVENT = {
    "prd": "prd_complete",
    "uiux": "uiux_complete",
    "style": "style_complete",
    "architecture": "architecture_complete",
    "plan": "plan_complete",
}

ALLOWED_MODELS = []  # re-exported from model_catalog — see list_available_models


def update_pipeline(
    app_id: str | int | None,
    user_email: str | None,
    user_id: int | None,
    status: str,
    error: str | None = None,
    **metadata: Any,
) -> None:
    if not app_id or not user_email:
        return
    try:
        db.update_app_builder_app(
            app_id=app_id,
            user_email=user_email,
            user_id=user_id,
            pipeline_status=status,
            pipeline_error=error,
            **metadata,
        )
    except Exception:
        pass


def touch_job(
    session_id: str,
    *,
    app_id: str | int | None = None,
    job_type: str = "planning",
    status: str = "running",
    step: str | None = None,
    logs: str | None = None,
    error: str | None = None,
    finished: bool = False,
) -> None:
    if not session_id:
        return
    try:
        db.create_or_update_job(
            session_id=session_id,
            app_id=app_id,
            job_type=job_type,
            status=status,
            step=step,
            logs=logs,
            error=error,
            finished=finished,
        )
    except Exception:
        pass


def public_base_url(fallback: str = "http://localhost:8000") -> str:
    import os

    return (os.environ.get("PUBLIC_BASE_URL") or fallback).rstrip("/")


def npm_install_timeout() -> int:
    import os

    try:
        return int(os.environ.get("NPM_INSTALL_TIMEOUT", "120"))
    except ValueError:
        return 120


def npm_build_timeout() -> int:
    import os

    try:
        return int(os.environ.get("NPM_BUILD_TIMEOUT", "180"))
    except ValueError:
        return 180
