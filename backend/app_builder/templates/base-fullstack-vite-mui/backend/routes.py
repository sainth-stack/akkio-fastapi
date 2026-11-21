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
        "total_lots": 142,
        "pending_inspections": 23,
        "released": 98,
        "held": 12,
        "rejected": 9,
        "open_capa": 7,
        "incoming_lots": 18,
        "total_items": 3420,
        "low_stock_alerts": 14,
        "total_orders": 287,
        "pending_orders": 34,
        "total_suppliers": 67,
    }


@router.get("/api/dashboard/stats")
def dashboard_stats():
    return {
        "total_lots": 142,
        "pending_inspections": 23,
        "released": 98,
        "held": 12,
        "rejected": 9,
        "open_capa": 7,
        "incoming_lots": 18,
        "total_items": 3420,
        "low_stock_alerts": 14,
        "total_orders": 287,
        "pending_orders": 34,
        "total_suppliers": 67,
    }


@router.get("/api/lots")
def list_lots():
    return {
        "items": [
            {"id": 1, "lot_number": "LOT-2024-001", "product": "Paracetamol 500mg", "quantity": 5000, "status": "Released", "created_at": "2024-01-10"},
            {"id": 2, "lot_number": "LOT-2024-002", "product": "Amoxicillin 250mg", "quantity": 2000, "status": "Pending Inspection", "created_at": "2024-01-12"},
            {"id": 3, "lot_number": "LOT-2024-003", "product": "Vitamin C 1000mg", "quantity": 8000, "status": "Released", "created_at": "2024-01-14"},
            {"id": 4, "lot_number": "LOT-2024-004", "product": "Ibuprofen 400mg", "quantity": 3000, "status": "Held", "created_at": "2024-01-15"},
            {"id": 5, "lot_number": "LOT-2024-005", "product": "Metformin 500mg", "quantity": 6000, "status": "Released", "created_at": "2024-01-16"},
        ],
        "total": 142,
    }


@router.get("/api/inspections")
def list_inspections():
    return {
        "items": [
            {"id": 1, "lot_number": "LOT-2024-002", "inspector": "Dr. Sarah Chen", "type": "Incoming", "status": "Pending", "scheduled_at": "2024-01-20"},
            {"id": 2, "lot_number": "LOT-2024-006", "inspector": "Dr. Raj Kumar", "type": "Release", "status": "In Progress", "scheduled_at": "2024-01-21"},
            {"id": 3, "lot_number": "LOT-2024-007", "inspector": "Dr. Maria Santos", "type": "Stability", "status": "Completed", "scheduled_at": "2024-01-18"},
        ],
        "total": 23,
    }


@router.get("/api/capa")
def list_capa():
    return {
        "items": [
            {"id": 1, "title": "Moisture deviation in LOT-2024-004", "severity": "Major", "status": "Open", "owner": "QA Team", "due_date": "2024-02-01"},
            {"id": 2, "title": "Supplier certificate expired", "severity": "Minor", "status": "Open", "owner": "Procurement", "due_date": "2024-01-25"},
            {"id": 3, "title": "Packaging integrity failure", "severity": "Critical", "status": "In Progress", "owner": "Production", "due_date": "2024-01-30"},
        ],
        "total": 7,
    }


@router.get("/api/suppliers")
def list_suppliers():
    return {
        "items": [
            {"id": 1, "name": "PharmaChem Ltd", "country": "India", "status": "Approved", "rating": 4.8, "last_audit": "2023-10-15"},
            {"id": 2, "name": "BioSynth Corp", "country": "Germany", "status": "Approved", "rating": 4.5, "last_audit": "2023-11-01"},
            {"id": 3, "name": "NaturalRaw Inc", "country": "USA", "status": "Conditional", "rating": 3.9, "last_audit": "2023-09-20"},
        ],
        "total": 67,
    }


@router.get("/api/orders")
def list_orders():
    return {
        "items": [
            {"id": 1, "order_number": "ORD-2024-001", "supplier": "PharmaChem Ltd", "items": 12, "total": 45000, "status": "Delivered", "created_at": "2024-01-05"},
            {"id": 2, "order_number": "ORD-2024-002", "supplier": "BioSynth Corp", "items": 5, "total": 18500, "status": "Pending", "created_at": "2024-01-08"},
            {"id": 3, "order_number": "ORD-2024-003", "supplier": "NaturalRaw Inc", "items": 8, "total": 32000, "status": "In Transit", "created_at": "2024-01-10"},
        ],
        "total": 287,
    }
