from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from .schemas import UserOut
from .security import decode_access_token
from .store import auth_store

bearer_scheme = HTTPBearer(auto_error=False)


class CurrentUser:
    def __init__(
        self,
        user: dict,
        permissions: set[str],
        token_jti: Optional[str] = None,
        token_exp: Optional[datetime] = None,
    ):
        self.user = user
        self.permissions = permissions
        self.token_jti = token_jti
        self.token_exp = token_exp

    @property
    def id(self) -> int:
        return self.user["id"]


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> CurrentUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    try:
        payload = decode_access_token(token)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    jti = payload.get("jti")
    token_exp = payload.get("exp")
    if isinstance(token_exp, (int, float)):
        token_exp = datetime.fromtimestamp(token_exp, tz=timezone.utc)
    elif not isinstance(token_exp, datetime):
        token_exp = None

    if jti and auth_store.is_token_revoked(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    user = auth_store.get_user_by_id(int(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    permissions = auth_store.get_user_permissions(user["id"])
    return CurrentUser(
        user=user,
        permissions=permissions,
        token_jti=jti,
        token_exp=token_exp,
    )


def require_permission(permission: str) -> Callable:
    async def _checker(current: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if permission not in current.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permission: {permission}",
            )
        return current

    return _checker


def user_out_from_dict(user: dict) -> UserOut:
    return UserOut(**user)
