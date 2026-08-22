"""Shared project ownership checks for App Builder endpoints."""

from __future__ import annotations

from fastapi import HTTPException

from api.auth.dependencies import CurrentUser
from api.auth.request_auth import user_email_from
from db.app_builder import get_app_builder_db

_db = get_app_builder_db()


def assert_project_access(project_name: str, current: CurrentUser) -> dict:
    """
    Verify the authenticated user owns a registered builder_apps row for project_name.
    Denies access if no DB row exists (orphan disk dirs) or row belongs to another user.
    Returns the app record on success.
    """
    if not project_name or not str(project_name).strip():
        raise HTTPException(status_code=400, detail="Project name is required")

    user_email = user_email_from(current)
    user_id = current.id if current.id else None
    app = _db.get_app_by_project_name(
        project_name.strip(),
        user_id=user_id,
        user_email=user_email,
    )
    if not app:
        raise HTTPException(
            status_code=403,
            detail="Project not found or you do not have access to this project",
        )
    return app


def assert_app_id_access(
    app_id: str | int | None,
    project_name: str,
    current: CurrentUser,
) -> dict | None:
    """Verify app_id belongs to the current user and matches project_name."""
    if not app_id:
        return None
    user_email = user_email_from(current)
    user_id = current.id if current.id else None
    app = _db.get_app_builder_app(app_id, user_email=user_email, user_id=user_id)
    if not app:
        raise HTTPException(status_code=404, detail="App not found")
    if (app.get("project_name") or "").strip() != (project_name or "").strip():
        raise HTTPException(status_code=403, detail="app_id does not match project_name")
    return app
