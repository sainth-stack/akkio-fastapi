from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .dependencies import CurrentUser, get_current_user

bearer_scheme = HTTPBearer(auto_error=False)


async def resolve_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> CurrentUser:
    if credentials and credentials.scheme.lower() == "bearer":
        return await get_current_user(credentials)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def resolve_user_for_generated_app(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> CurrentUser:
    """
    Auth for /api/apps/* — accepts Bearer header or ?access_token= (preview iframe).
    """
    if credentials and credentials.scheme.lower() == "bearer":
        return await get_current_user(credentials)
    token = (request.query_params.get("access_token") or "").strip()
    if token:
        preview_creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        return await get_current_user(preview_creds)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def resolve_user_flexible(request: Request) -> CurrentUser:
    """JWT from Authorization header (supports multipart form uploads)."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=auth.split(" ", 1)[1])
        return await get_current_user(creds)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


def user_email_from(current: CurrentUser) -> str:
    return current.user["email"]
