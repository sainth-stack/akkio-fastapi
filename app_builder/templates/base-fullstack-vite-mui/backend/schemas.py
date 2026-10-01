from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class LoginRequest(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    role: str

    class Config:
        from_attributes = True


class AuditLogOut(BaseModel):
    id: int
    user: Optional[str] = None
    action: str
    entity: Optional[str] = None
    entity_id: Optional[str] = None
    timestamp: datetime

    class Config:
        from_attributes = True
