"""Shared project ownership checks for App Builder endpoints."""

from __future__ import annotations

from typing import Optional
import logging

from fastapi import HTTPException

from api.auth.dependencies import CurrentUser
from api.auth.request_auth import user_email_from
from db.app_builder import get_app_builder_db

_db = get_app_builder_db()
logger = logging.getLogger(__name__)


def assert_project_access(
    project_name: str,
    current: CurrentUser,
    *,
    auto_create: bool = True,
    app_name: Optional[str] = None,
    prompt: Optional[str] = None,
    builder_kind: Optional[str] = None,
) -> dict:
    """
    Verify the authenticated user owns a registered builder_apps row for project_name.
    If no row exists and auto_create=True, creates one so that codegen can proceed
    even when the initial save during planning failed.
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
    if app:
        return app

    # Row missing — attempt auto-create so codegen is not blocked
    if auto_create:
        try:
            app = _db.create_app_builder_app(
                user_email=user_email,
                app_name=app_name or project_name,
                prompt=prompt or "",
                project_name=project_name.strip(),
                user_id=user_id,
                builder_kind=builder_kind or "fullstack",
            )
            logger.warning(
                "assert_project_access: auto-created missing builder_apps row "
                "for project_name=%s user=%s",
                project_name,
                user_email,
            )
            return app
        except Exception as exc:
            logger.error(
                "assert_project_access: auto-create failed for project_name=%s: %s",
                project_name,
                exc,
            )

    raise HTTPException(
        status_code=403,
        detail="Project not found or you do not have access to this project",
    )


def assert_app_id_access(
    app_id: str | Optional[int],
    project_name: str,
    current: CurrentUser,
) -> Optional[dict]:
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
