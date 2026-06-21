from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class OrganizationOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class RoleOut(BaseModel):
    id: int
    name: str
    permissions: list[str]
    created_at: datetime
    updated_at: datetime


class UserOut(BaseModel):
    id: int
    name: Optional[str] = None
    email: str
    username: Optional[str] = None
    app: str
    organization: Optional[OrganizationOut] = None
    roles: list[RoleOut] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class AuthResponse(BaseModel):
    user: UserOut
    access_token: str
    token_type: str = "bearer"


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    username: Optional[str] = None
    app: str = "akkio"
    organization_id: Optional[int] = None
    role_ids: list[int] = Field(default_factory=list)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    app: str = "akkio"


class GoogleLoginRequest(BaseModel):
    id_token: str
    app: str = "akkio"


class LogoutResponse(BaseModel):
    message: str = "Logged out successfully"
