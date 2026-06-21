from fastapi import APIRouter, Depends, HTTPException, status

from api.auth.dependencies import CurrentUser, require_permission
from api.auth.schemas import OrganizationOut, RoleOut, UserOut
from api.auth.security import hash_password
from api.auth.store import auth_store

from .schemas import (
    DeleteMessage,
    OrganizationCreate,
    OrganizationUpdate,
    RoleCreate,
    RoleUpdate,
    UserCreate,
    UserUpdate,
)

router = APIRouter(tags=["admin"])


# --- Users ---


@router.get("/users", response_model=list[UserOut])
async def list_users(_: CurrentUser = Depends(require_permission("admin_read"))):
    return auth_store.list_users()


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: UserCreate,
    _: CurrentUser = Depends(require_permission("admin_write")),
):
    user = auth_store.create_user(
        name=body.name,
        email=body.email,
        password_hash=hash_password(body.password),
        username=body.username or body.email.split("@")[0],
        app=body.app,
        organization_id=body.organization_id,
        role_ids=body.role_ids,
    )
    return user


@router.get("/users/{user_id}", response_model=UserOut)
async def get_user(
    user_id: int,
    _: CurrentUser = Depends(require_permission("admin_read")),
):
    user = auth_store.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.put("/users/{user_id}", response_model=UserOut)
async def update_user(
    user_id: int,
    body: UserUpdate,
    _: CurrentUser = Depends(require_permission("admin_write")),
):
    updates = body.model_dump(exclude_unset=True)
    if "password" in updates and updates["password"]:
        updates["password_hash"] = hash_password(updates.pop("password"))
    elif "password" in updates:
        updates.pop("password")
    user = auth_store.update_user(user_id, updates)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.delete("/users/{user_id}", response_model=DeleteMessage)
async def delete_user(
    user_id: int,
    _: CurrentUser = Depends(require_permission("admin_write")),
):
    if not auth_store.delete_user(user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return DeleteMessage(message="User deleted successfully")


# --- Organizations ---


@router.get("/organizations", response_model=list[OrganizationOut])
async def list_organizations(_: CurrentUser = Depends(require_permission("admin_read"))):
    return auth_store.list_organizations()


@router.post("/organizations", response_model=OrganizationOut, status_code=status.HTTP_201_CREATED)
async def create_organization(
    body: OrganizationCreate,
    _: CurrentUser = Depends(require_permission("admin_write")),
):
    return auth_store.create_organization(body.name, body.description)


@router.get("/organizations/{org_id}", response_model=OrganizationOut)
async def get_organization(
    org_id: int,
    _: CurrentUser = Depends(require_permission("admin_read")),
):
    org = auth_store.get_organization(org_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    return org


@router.put("/organizations/{org_id}", response_model=OrganizationOut)
async def update_organization(
    org_id: int,
    body: OrganizationUpdate,
    _: CurrentUser = Depends(require_permission("admin_write")),
):
    org = auth_store.update_organization(org_id, body.model_dump(exclude_unset=True))
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    return org


@router.delete("/organizations/{org_id}", response_model=DeleteMessage)
async def delete_organization(
    org_id: int,
    _: CurrentUser = Depends(require_permission("admin_write")),
):
    if not auth_store.delete_organization(org_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    return DeleteMessage(message="Organization deleted successfully")


# --- Roles ---


@router.get("/roles", response_model=list[RoleOut])
async def list_roles(_: CurrentUser = Depends(require_permission("admin_read"))):
    return auth_store.list_roles()


@router.post("/roles", response_model=RoleOut, status_code=status.HTTP_201_CREATED)
async def create_role(
    body: RoleCreate,
    _: CurrentUser = Depends(require_permission("admin_write")),
):
    return auth_store.create_role(body.name, body.permissions)


@router.get("/roles/{role_id}", response_model=RoleOut)
async def get_role(
    role_id: int,
    _: CurrentUser = Depends(require_permission("admin_read")),
):
    role = auth_store.get_role(role_id)
    if not role:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")
    return role


@router.put("/roles/{role_id}", response_model=RoleOut)
async def update_role(
    role_id: int,
    body: RoleUpdate,
    _: CurrentUser = Depends(require_permission("admin_write")),
):
    role = auth_store.update_role(role_id, body.model_dump(exclude_unset=True))
    if not role:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")
    return role


@router.delete("/roles/{role_id}", response_model=DeleteMessage)
async def delete_role(
    role_id: int,
    _: CurrentUser = Depends(require_permission("admin_write")),
):
    if not auth_store.delete_role(role_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")
    return DeleteMessage(message="Role deleted successfully")
