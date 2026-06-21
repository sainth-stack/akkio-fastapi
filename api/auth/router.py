import os

from fastapi import APIRouter, Depends, HTTPException, status
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from .dependencies import CurrentUser, get_current_user, user_out_from_dict
from .schemas import (
    AuthResponse,
    GoogleLoginRequest,
    LoginRequest,
    LogoutResponse,
    UserOut,
)
from .security import create_access_token, verify_password
from .store import auth_store

router = APIRouter(tags=["auth"])

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")


def _auth_response(user: dict) -> AuthResponse:
    access_token, _, _ = create_access_token(user_id=user["id"], email=user["email"])
    return AuthResponse(user=UserOut(**user), access_token=access_token)


@router.post("/login", response_model=AuthResponse)
async def login(body: LoginRequest):
    creds = auth_store.get_user_credentials(body.email, body.app)
    if not creds or not verify_password(body.password, creds.get("password_hash")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    user = auth_store.get_user_by_id(creds["id"])
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return _auth_response(user)


@router.post("/google-login", response_model=AuthResponse)
async def google_login(body: GoogleLoginRequest):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google login is not configured (set GOOGLE_CLIENT_ID)",
        )
    try:
        idinfo = id_token.verify_oauth2_token(
            body.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google ID token",
        ) from exc

    email = idinfo.get("email")
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Google token missing email")

    google_id = idinfo.get("sub", "")
    name = idinfo.get("name") or email.split("@")[0]
    default_org_id, default_role_id = auth_store.get_default_org_and_admin_role()
    default_role_ids = [default_role_id] if default_role_id else []

    user = auth_store.find_or_create_google_user(
        email=email,
        name=name,
        google_id=google_id,
        app=body.app,
        default_organization_id=default_org_id,
        default_role_ids=default_role_ids,
    )
    return _auth_response(user)


@router.get("/me", response_model=UserOut)
async def me(current: CurrentUser = Depends(get_current_user)):
    return user_out_from_dict(current.user)


@router.post("/logout", response_model=LogoutResponse)
async def logout(current: CurrentUser = Depends(get_current_user)):
    if current.token_jti:
        from datetime import datetime, timedelta, timezone

        expires_at = current.token_exp or (datetime.now(timezone.utc) + timedelta(days=2))
        auth_store.revoke_token(current.token_jti, expires_at)
    return LogoutResponse()
