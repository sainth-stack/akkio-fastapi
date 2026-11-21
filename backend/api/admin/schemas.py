from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class OrganizationCreate(BaseModel):
    name: str
    description: Optional[str] = None


class OrganizationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class RoleCreate(BaseModel):
    name: str
    permissions: list[str] = Field(default_factory=list)


class RoleUpdate(BaseModel):
    name: Optional[str] = None
    permissions: Optional[list[str]] = None


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    username: Optional[str] = None
    app: str = "akkio"
    organization_id: Optional[int] = None
    role_ids: list[int] = Field(default_factory=list)


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(default=None, min_length=8, max_length=128)
    username: Optional[str] = None
    app: Optional[str] = None
    organization_id: Optional[int] = None
    role_ids: Optional[list[int]] = None


class DeleteMessage(BaseModel):
    message: str
