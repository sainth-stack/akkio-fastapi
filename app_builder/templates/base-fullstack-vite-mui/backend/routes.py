from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from auth import create_access_token, get_current_user, verify_password
from database import get_db
from models import User
from schemas import LoginRequest, TokenResponse, UserOut

router = APIRouter()


@router.post("/api/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user.email, user.role)
    return TokenResponse(access_token=token, role=user.role)


@router.get("/api/auth/me", response_model=UserOut)
def me(user=Depends(get_current_user)):
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@router.get("/api/dashboard/kpis")
def dashboard_kpis():
    return {
        "total_lots": 0,
        "pending_inspections": 0,
        "released": 0,
        "held": 0,
        "rejected": 0,
        "open_capa": 0,
    }
