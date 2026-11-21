from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException, WebSocket, status
from jose import JWTError

from .dependencies import CurrentUser
from .security import decode_access_token
from .store import auth_store


def _user_from_token(token: str) -> CurrentUser:
    try:
        payload = decode_access_token(token)
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    jti = payload.get("jti")
    if jti and auth_store.is_token_revoked(jti):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    user = auth_store.get_user_by_id(int(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    perms = auth_store.get_user_permissions(user["id"])
    return CurrentUser(user=user, permissions=perms, token_jti=jti)


async def authenticate_websocket(
    websocket: WebSocket,
    first_message: Optional[dict[str, Any]] = None,
) -> tuple[CurrentUser, dict[str, Any]]:
    """
    Authenticate WebSocket via ?token= query param or first JSON message { "token": "..." }.
    Returns (user, payload) where payload is the first message minus token.
    """
    token = websocket.query_params.get("token")
    payload: dict[str, Any] = first_message or {}

    if not token and first_message:
        token = first_message.get("token")
        payload = {k: v for k, v in first_message.items() if k != "token"}

    if not token:
        await websocket.close(code=4401, reason="Unauthorized")
        raise HTTPException(status_code=401, detail="WebSocket authentication required")

    user = _user_from_token(token)
    return user, payload
