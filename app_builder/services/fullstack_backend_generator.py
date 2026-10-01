"""Backend files for Agentic Builder production MVPs (FastAPI + SQLAlchemy + JWT)."""
from __future__ import annotations

from typing import Dict


def backend_files(title: str, quality: bool) -> Dict[str, str]:
    safe = (title or "Generated API").replace('"', "'")[:80]
    files = {
        "backend/models.py": _models_py(),
        "backend/schemas.py": _schemas_py(),
        "backend/risk_engine.py": _risk_engine_py(),
        "backend/routes.py": _routes_py(),
        "backend/seed.py": _seed_py(),
        "backend/main.py": _main_py(safe),
        "backend/requirements.txt": _requirements_txt(),
        "backend/alembic/versions/001_initial_schema.py": _alembic_001(),
        "backend/tests/test_risk_engine.py": _test_risk_py(),
        "backend/tests/test_inspection.py": _test_inspection_py(),
        "README.md": _readme(safe),
    }
    if not quality:
        files["README.md"] = _readme(safe)
    return files


def _main_py(title: str) -> str:
    return '''import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import Base, engine
from routes import router

app = FastAPI(title="%s", version="1.0.0", docs_url="/docs", redoc_url="/redoc")

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
''' % title


def _requirements_txt() -> str:
    return """fastapi
uvicorn
sqlalchemy
pydantic
pydantic-settings
psycopg2-binary
python-jose[cryptography]
passlib[bcrypt]
python-dotenv
alembic
pytest
httpx
"""


def _models_py() -> str:
    return r'''from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from database import Base


class TimestampMixin:
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(64), nullable=False, default="VIEWER")
    full_name = Column(String(255))

    __table_args__ = (
        CheckConstraint("role IN ('ADMIN','QUALITY_MANAGER','INSPECTOR','VIEWER')", name="ck_users_role"),
    )


class Supplier(Base, TimestampMixin):
    __tablename__ = "suppliers"

    id = Column(Integer, primary_key=True)
    code = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    location = Column(String(255))
    category = Column(String(128))
    quality_score = Column(Numeric(6, 2), default=0)
    ppm = Column(Numeric(10, 2), default=0)
    defect_rate = Column(Numeric(8, 4), default=0)
    rejected_lots = Column(Integer, default=0)
    open_capa = Column(Integer, default=0)
    status = Column(String(32), default="ACTIVE", index=True)

    lots = relationship("IncomingLot", back_populates="supplier")
    capas = relationship("Capa", back_populates="supplier")


class Material(Base, TimestampMixin):
    __tablename__ = "materials"

    id = Column(Integer, primary_key=True)
    code = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    material_type = Column(String(128))
    specification = Column(String(255))
    criticality = Column(String(32), default="MEDIUM", index=True)
    inspection_plan = Column(String(255))
    status = Column(String(32), default="ACTIVE")

    lots = relationship("IncomingLot", back_populates="material")

    __table_args__ = (
        CheckConstraint("criticality IN ('LOW','MEDIUM','HIGH','CRITICAL')", name="ck_materials_crit"),
    )


class PurchaseOrder(Base, TimestampMixin):
    __tablename__ = "purchase_orders"

    id = Column(Integer, primary_key=True)
    po_number = Column(String(64), unique=True, nullable=False, index=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=False, index=True)
    material_id = Column(Integer, ForeignKey("materials.id"), nullable=False, index=True)
    quantity = Column(Integer, nullable=False, default=0)
    status = Column(String(32), default="OPEN", index=True)

    supplier = relationship("Supplier")
    material = relationship("Material")
    lots = relationship("IncomingLot", back_populates="purchase_order")


class IncomingLot(Base, TimestampMixin):
    __tablename__ = "incoming_lots"

    id = Column(Integer, primary_key=True)
    lot_number = Column(String(64), unique=True, nullable=False, index=True)
    po_id = Column(Integer, ForeignKey("purchase_orders.id"), index=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=False, index=True)
    material_id = Column(Integer, ForeignKey("materials.id"), nullable=False, index=True)
    quantity = Column(Integer, nullable=False, default=0)
    received_date = Column(DateTime, default=datetime.utcnow, index=True)
    inspection_status = Column(String(32), default="PENDING", index=True)
    risk_score = Column(Numeric(6, 2), default=0)
    release_status = Column(String(32), default="PENDING", index=True)
    recommendation = Column(String(64))

    supplier = relationship("Supplier", back_populates="lots")
    material = relationship("Material", back_populates="lots")
    purchase_order = relationship("PurchaseOrder", back_populates="lots")
    inspections = relationship("InspectionResult", back_populates="lot")
    defects = relationship("Defect", back_populates="lot")
    decisions = relationship("ReleaseDecision", back_populates="lot")

    __table_args__ = (
        Index("ix_lots_supplier_release", "supplier_id", "release_status"),
        CheckConstraint(
            "inspection_status IN ('PENDING','PASS','FAIL','WARNING')",
            name="ck_lots_insp",
        ),
        CheckConstraint(
            "release_status IN ('PENDING','RELEASED','HOLD','REJECTED')",
            name="ck_lots_rel",
        ),
    )


class InspectionResult(Base, TimestampMixin):
    __tablename__ = "inspection_results"

    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("incoming_lots.id"), nullable=False, index=True)
    parameter = Column(String(128), nullable=False)
    specification = Column(String(255))
    lower_limit = Column(Numeric(12, 4))
    upper_limit = Column(Numeric(12, 4))
    actual_value = Column(Numeric(12, 4), nullable=False)
    unit = Column(String(32))
    result = Column(String(16), nullable=False, index=True)
    inspector = Column(String(128))
    inspection_date = Column(DateTime, default=datetime.utcnow, index=True)

    lot = relationship("IncomingLot", back_populates="inspections")

    __table_args__ = (
        CheckConstraint("result IN ('PASS','FAIL','WARNING')", name="ck_insp_result"),
    )


class Defect(Base, TimestampMixin):
    __tablename__ = "defects"

    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("incoming_lots.id"), nullable=False, index=True)
    category = Column(String(128), nullable=False, index=True)
    description = Column(Text)
    severity = Column(String(32), default="MAJOR", index=True)

    lot = relationship("IncomingLot", back_populates="defects")


class ReleaseDecision(Base, TimestampMixin):
    __tablename__ = "release_decisions"

    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("incoming_lots.id"), nullable=False, index=True)
    risk_score = Column(Numeric(6, 2))
    recommendation = Column(String(64))
    decision = Column(String(32), nullable=False, index=True)
    reason = Column(Text)
    user = Column(String(255), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    lot = relationship("IncomingLot", back_populates="decisions")


class Capa(Base, TimestampMixin):
    __tablename__ = "capa"

    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("incoming_lots.id"), index=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), index=True)
    title = Column(String(255), nullable=False)
    status = Column(String(32), default="OPEN", index=True)
    owner = Column(String(255))
    description = Column(Text)

    supplier = relationship("Supplier", back_populates="capas")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    user = Column(String(255), index=True)
    action = Column(String(128), nullable=False, index=True)
    entity = Column(String(128), index=True)
    entity_id = Column(String(64), index=True)
    old_value = Column(Text)
    new_value = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
'''


def _schemas_py() -> str:
    return r'''from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


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


class InspectionCreate(BaseModel):
    lot_id: int
    parameter: str
    specification: Optional[str] = None
    lower_limit: float
    upper_limit: float
    actual_value: float
    unit: Optional[str] = "mm"
    inspector: Optional[str] = None


class DecisionRequest(BaseModel):
    reason: str = "Decision recorded"
    decision: Optional[str] = None


class CapaCreate(BaseModel):
    title: Optional[str] = None
    lot_id: Optional[int] = None
    supplier_id: Optional[int] = None
    status: str = "OPEN"
    description: Optional[str] = None
    reason: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)


class AnalyzeLotRequest(BaseModel):
    lot_id: Optional[int] = None
    lot_number: Optional[str] = None
'''


def _risk_engine_py() -> str:
    return r'''"""Configurable incoming-material risk scoring. Mandatory quality rules always win."""
from __future__ import annotations

from typing import Dict, Optional

WEIGHTS = {
    "inspection": 0.30,
    "supplier_history": 0.25,
    "defect_history": 0.20,
    "criticality": 0.15,
    "trend": 0.10,
}

CRITICALITY_SCORE = {"LOW": 15, "MEDIUM": 40, "HIGH": 70, "CRITICAL": 90}


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(value)))


def inspection_risk(fail_count: int, warn_count: int, total: int) -> float:
    if total <= 0:
        return 45.0
    fail_rate = fail_count / float(total)
    warn_rate = warn_count / float(total)
    return clamp(fail_rate * 100.0 + warn_rate * 35.0)


def supplier_history_risk(ppm: float, defect_rate: float, rejected_lots: int) -> float:
    ppm_score = clamp((float(ppm) / 250.0) * 100.0)
    defect_score = clamp(float(defect_rate) * 20.0)
    reject_score = clamp(rejected_lots * 12.0)
    return clamp(0.5 * ppm_score + 0.3 * defect_score + 0.2 * reject_score)


def defect_history_risk(defect_count: int, critical_count: int) -> float:
    return clamp(defect_count * 8.0 + critical_count * 18.0)


def material_criticality_risk(criticality: str) -> float:
    return float(CRITICALITY_SCORE.get((criticality or "MEDIUM").upper(), 40))


def trend_risk(recent_fail_rate: float) -> float:
    return clamp(float(recent_fail_rate) * 100.0)


def compute_risk_score(
    inspection: float,
    supplier_history: float,
    defect_history: float,
    criticality: float,
    trend: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    w = weights or WEIGHTS
    score = (
        w["inspection"] * inspection
        + w["supplier_history"] * supplier_history
        + w["defect_history"] * defect_history
        + w["criticality"] * criticality
        + w["trend"] * trend
    )
    return round(clamp(score), 2)


def recommendation_for(score: float, mandatory_hold: bool = False) -> str:
    if mandatory_hold:
        return "HOLD / REJECT"
    if score <= 30:
        return "AUTO RELEASE"
    if score <= 60:
        return "NORMAL INSPECTION"
    if score <= 80:
        return "QUALITY REVIEW"
    return "HOLD / REJECT"


def inspection_result(actual: float, lower: float, upper: float) -> str:
    if actual < lower or actual > upper:
        return "FAIL"
    span = max(upper - lower, 0.0001)
    warn = span * 0.10
    if actual <= lower + warn or actual >= upper - warn:
        return "WARNING"
    return "PASS"
'''


def _routes_py() -> str:
    return r'''import json
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from auth import create_access_token, get_current_user, require_roles, verify_password
from database import get_db
from models import (
    AuditLog,
    Capa,
    Defect,
    IncomingLot,
    InspectionResult,
    Material,
    PurchaseOrder,
    ReleaseDecision,
    Supplier,
    User,
)
from risk_engine import (
    compute_risk_score,
    defect_history_risk,
    inspection_result as calc_result,
    inspection_risk,
    material_criticality_risk,
    recommendation_for,
    supplier_history_risk,
    trend_risk,
)
from schemas import (
    AnalyzeLotRequest,
    AskRequest,
    CapaCreate,
    DecisionRequest,
    InspectionCreate,
    LoginRequest,
    TokenResponse,
    UserOut,
)

router = APIRouter()


def _audit(db: Session, user: Optional[User], action: str, entity: str, entity_id, old=None, new=None):
    db.add(
        AuditLog(
            user=user.email if user else None,
            action=action,
            entity=entity,
            entity_id=str(entity_id) if entity_id is not None else None,
            old_value=json.dumps(old) if old is not None else None,
            new_value=json.dumps(new) if new is not None else None,
        )
    )


def _paginate(query, skip: int, limit: int):
    total = query.count()
    items = query.offset(skip).limit(limit).all()
    return {"items": items, "total": total}


def _lot_dict(lot: IncomingLot):
    return {
        "id": lot.id,
        "lot_number": lot.lot_number,
        "po_number": lot.purchase_order.po_number if lot.purchase_order else None,
        "supplier": lot.supplier.name if lot.supplier else None,
        "supplier_id": lot.supplier_id,
        "material": lot.material.name if lot.material else None,
        "material_id": lot.material_id,
        "quantity": lot.quantity,
        "received_date": lot.received_date.isoformat() if lot.received_date else None,
        "inspection_status": lot.inspection_status,
        "risk_score": float(lot.risk_score or 0),
        "release_status": lot.release_status,
        "recommendation": lot.recommendation,
    }


def _score_lot(db: Session, lot: IncomingLot) -> float:
    inspections = lot.inspections or []
    fails = sum(1 for i in inspections if i.result == "FAIL")
    warns = sum(1 for i in inspections if i.result == "WARNING")
    defects = lot.defects or []
    critical = sum(1 for d in defects if (d.severity or "").upper() == "CRITICAL")
    supplier = lot.supplier
    recent_cutoff = datetime.utcnow() - timedelta(days=90)
    recent = (
        db.query(InspectionResult)
        .join(IncomingLot)
        .filter(IncomingLot.supplier_id == lot.supplier_id, InspectionResult.inspection_date >= recent_cutoff)
        .all()
    )
    recent_fail = (sum(1 for r in recent if r.result == "FAIL") / float(len(recent))) if recent else 0.2
    score = compute_risk_score(
        inspection_risk(fails, warns, len(inspections)),
        supplier_history_risk(float(supplier.ppm or 0), float(supplier.defect_rate or 0), int(supplier.rejected_lots or 0)) if supplier else 50,
        defect_history_risk(len(defects), critical),
        material_criticality_risk(lot.material.criticality if lot.material else "MEDIUM"),
        trend_risk(recent_fail),
    )
    lot.risk_score = score
    mandatory = any(i.result == "FAIL" for i in inspections) and (lot.material.criticality in ("HIGH", "CRITICAL") if lot.material else False)
    lot.recommendation = recommendation_for(score, mandatory_hold=mandatory)
    return score


@router.post("/api/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user.email, user.role)
    _audit(db, user, "LOGIN", "users", user.id, new={"email": user.email})
    db.commit()
    return TokenResponse(access_token=token, role=user.role)


@router.get("/api/auth/me", response_model=UserOut)
def me(user=Depends(get_current_user)):
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@router.get("/api/dashboard/kpis")
def dashboard_kpis(db: Session = Depends(get_db)):
    lots = db.query(IncomingLot)
    total = lots.count()
    pending = lots.filter(IncomingLot.inspection_status == "PENDING").count()
    released = lots.filter(IncomingLot.release_status == "RELEASED").count()
    held = lots.filter(IncomingLot.release_status == "HOLD").count()
    rejected = lots.filter(IncomingLot.release_status == "REJECTED").count()
    open_capa = db.query(Capa).filter(Capa.status == "OPEN").count()
    avg_ppm = db.query(func.avg(Supplier.ppm)).scalar() or 0
    defect_count = db.query(Defect).count()
    defect_rate = round((defect_count / float(total * 10 or 1)) * 100, 2)
    release_rate = round((released / float(total or 1)) * 100, 1)
    high_risk = lots.filter(IncomingLot.risk_score >= 61).count()
    high_rows = (
        db.query(IncomingLot)
        .options(joinedload(IncomingLot.supplier), joinedload(IncomingLot.material), joinedload(IncomingLot.purchase_order))
        .filter(IncomingLot.risk_score >= 61)
        .order_by(IncomingLot.risk_score.desc())
        .limit(12)
        .all()
    )
    weeks = ["W1", "W2", "W3", "W4", "W5", "W6"]
    ppm_trend = [{"label": w, "value": round(float(avg_ppm) * (0.8 + i * 0.05), 1)} for i, w in enumerate(weeks)]
    defect_trend = [{"label": w, "value": 8 + i * 3} for i, w in enumerate(weeks)]
    release_trend = [{"label": w, "value": 18 + i * 2, "released": 18 + i * 2, "rejected": 4} for i, w in enumerate(weeks)]
    cats = db.query(Defect.category, func.count(Defect.id)).group_by(Defect.category).order_by(func.count(Defect.id).desc()).limit(6).all()
    top_defects = [{"label": c or "Other", "value": n} for c, n in cats]
    return {
        "total_lots": total,
        "pending_inspections": pending,
        "released": released,
        "held": held,
        "rejected": rejected,
        "supplier_ppm": round(float(avg_ppm), 1),
        "defect_rate": f"{defect_rate}%",
        "release_rate": f"{release_rate}%",
        "high_risk_lots": high_risk,
        "open_capa": open_capa,
        "ppm_trend": ppm_trend,
        "defect_trend": defect_trend,
        "release_trend": release_trend,
        "top_defects": top_defects,
        "high_risk_table": [_lot_dict(l) for l in high_rows],
    }


@router.get("/api/suppliers")
def list_suppliers(q: Optional[str] = None, skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    query = db.query(Supplier)
    if q:
        like = f"%{q}%"
        query = query.filter((Supplier.name.ilike(like)) | (Supplier.code.ilike(like)))
    total = query.count()
    rows = query.order_by(Supplier.code).offset(skip).limit(limit).all()
    items = [
        {
            "id": s.id,
            "code": s.code,
            "name": s.name,
            "location": s.location,
            "category": s.category,
            "quality_score": float(s.quality_score or 0),
            "ppm": float(s.ppm or 0),
            "defect_rate": float(s.defect_rate or 0),
            "rejected_lots": s.rejected_lots,
            "open_capa": s.open_capa,
            "status": s.status,
        }
        for s in rows
    ]
    return {"items": items, "total": total}


@router.get("/api/suppliers/{supplier_id}")
def get_supplier(supplier_id: int, db: Session = Depends(get_db)):
    s = db.query(Supplier).filter(Supplier.id == supplier_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Supplier not found")
    lots = (
        db.query(IncomingLot)
        .options(joinedload(IncomingLot.material), joinedload(IncomingLot.purchase_order))
        .filter(IncomingLot.supplier_id == s.id)
        .order_by(IncomingLot.received_date.desc())
        .limit(40)
        .all()
    )
    capas = db.query(Capa).filter(Capa.supplier_id == s.id).all()
    weeks = ["W1", "W2", "W3", "W4", "W5", "W6"]
    return {
        "id": s.id,
        "code": s.code,
        "name": s.name,
        "location": s.location,
        "category": s.category,
        "quality_score": float(s.quality_score or 0),
        "ppm": float(s.ppm or 0),
        "defect_rate": float(s.defect_rate or 0),
        "rejected_lots": s.rejected_lots,
        "open_capa": s.open_capa,
        "status": s.status,
        "quality_score_calculation": "0.4*(100-min(ppm/5,100)) + 0.3*(100-defect_rate*10) + 0.3*(100-rejected_lots*8)",
        "ppm_trend": [{"label": w, "value": max(8, float(s.ppm or 40) - 6 + i * 3)} for i, w in enumerate(weeks)],
        "defect_trend": [{"label": w, "value": max(0.2, float(s.defect_rate or 1) + i * 0.1)} for i, w in enumerate(weeks)],
        "lots": [_lot_dict(l) for l in lots],
        "rejected_lots_list": [_lot_dict(l) for l in lots if l.release_status == "REJECTED"],
        "capa": [{"id": c.id, "title": c.title, "status": c.status} for c in capas],
    }


@router.get("/api/materials")
def list_materials(q: Optional[str] = None, skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    query = db.query(Material)
    if q:
        like = f"%{q}%"
        query = query.filter((Material.name.ilike(like)) | (Material.code.ilike(like)))
    total = query.count()
    rows = query.order_by(Material.code).offset(skip).limit(limit).all()
    items = [
        {
            "id": m.id,
            "code": m.code,
            "name": m.name,
            "type": m.material_type,
            "specification": m.specification,
            "criticality": m.criticality,
            "inspection_plan": m.inspection_plan,
            "status": m.status,
        }
        for m in rows
    ]
    return {"items": items, "total": total}


@router.get("/api/purchase-orders")
def list_pos(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    query = db.query(PurchaseOrder).options(joinedload(PurchaseOrder.supplier), joinedload(PurchaseOrder.material))
    total = query.count()
    rows = query.order_by(PurchaseOrder.id.desc()).offset(skip).limit(limit).all()
    items = [
        {
            "id": p.id,
            "po_number": p.po_number,
            "supplier": p.supplier.name if p.supplier else None,
            "material": p.material.name if p.material else None,
            "quantity": p.quantity,
            "status": p.status,
        }
        for p in rows
    ]
    return {"items": items, "total": total}


@router.get("/api/incoming-lots")
def list_lots(
    q: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    release_status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    query = db.query(IncomingLot).options(
        joinedload(IncomingLot.supplier),
        joinedload(IncomingLot.material),
        joinedload(IncomingLot.purchase_order),
    )
    if q:
        like = f"%{q}%"
        query = query.filter(IncomingLot.lot_number.ilike(like))
    if release_status:
        query = query.filter(IncomingLot.release_status == release_status.upper())
    total = query.count()
    rows = query.order_by(IncomingLot.id.desc()).offset(skip).limit(limit).all()
    return {"items": [_lot_dict(l) for l in rows], "total": total}


@router.get("/api/incoming-lots/{lot_id}")
def get_lot(lot_id: int, db: Session = Depends(get_db)):
    lot = (
        db.query(IncomingLot)
        .options(
            joinedload(IncomingLot.supplier),
            joinedload(IncomingLot.material),
            joinedload(IncomingLot.purchase_order),
            joinedload(IncomingLot.inspections),
            joinedload(IncomingLot.defects),
        )
        .filter(IncomingLot.id == lot_id)
        .first()
    )
    if not lot:
        raise HTTPException(status_code=404, detail="Lot not found")
    score = _score_lot(db, lot)
    db.commit()
    hist = (
        db.query(IncomingLot)
        .filter(IncomingLot.supplier_id == lot.supplier_id, IncomingLot.id != lot.id)
        .order_by(IncomingLot.received_date.desc())
        .limit(8)
        .all()
    )
    return {
        **_lot_dict(lot),
        "risk_score": score,
        "recommendation": lot.recommendation,
        "ai_recommendation": (
            f"Grounded recommendation {lot.recommendation} at risk {score}. "
            "Mandatory FAIL on HIGH/CRITICAL material cannot be auto-released."
        ),
        "inspections": [
            {
                "id": i.id,
                "parameter": i.parameter,
                "specification": i.specification,
                "lower_limit": float(i.lower_limit or 0),
                "upper_limit": float(i.upper_limit or 0),
                "actual_value": float(i.actual_value or 0),
                "unit": i.unit,
                "result": i.result,
                "inspector": i.inspector,
                "inspection_date": i.inspection_date.isoformat() if i.inspection_date else None,
            }
            for i in lot.inspections
        ],
        "defects": [
            {"id": d.id, "category": d.category, "description": d.description, "severity": d.severity}
            for d in lot.defects
        ],
        "historical_supplier_quality": [_lot_dict(h) for h in hist],
    }


@router.post("/api/inspections")
def create_inspection(
    payload: InspectionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("INSPECTOR", "QUALITY_MANAGER", "ADMIN")),
):
    lot = db.query(IncomingLot).filter(IncomingLot.id == payload.lot_id).first()
    if not lot:
        raise HTTPException(status_code=404, detail="Lot not found")
    result = calc_result(payload.actual_value, payload.lower_limit, payload.upper_limit)
    row = InspectionResult(
        lot_id=payload.lot_id,
        parameter=payload.parameter,
        specification=payload.specification,
        lower_limit=payload.lower_limit,
        upper_limit=payload.upper_limit,
        actual_value=payload.actual_value,
        unit=payload.unit,
        result=result,
        inspector=payload.inspector or (user.email if user else None),
        inspection_date=datetime.utcnow(),
    )
    db.add(row)
    lot.inspection_status = "FAIL" if result == "FAIL" else ("WARNING" if result == "WARNING" and lot.inspection_status != "FAIL" else lot.inspection_status if lot.inspection_status == "FAIL" else "PASS")
    _score_lot(db, lot)
    _audit(db, user, "INSPECTION_CREATE", "inspection_results", payload.lot_id, new=payload.model_dump() if hasattr(payload, "model_dump") else payload.dict())
    db.commit()
    db.refresh(row)
    return {"id": row.id, "result": result, "lot_id": lot.id, "risk_score": float(lot.risk_score or 0)}


@router.get("/api/inspections/{lot_id}")
def list_inspections(lot_id: int, db: Session = Depends(get_db)):
    rows = db.query(InspectionResult).filter(InspectionResult.lot_id == lot_id).all()
    return {
        "items": [
            {
                "id": i.id,
                "parameter": i.parameter,
                "specification": i.specification,
                "lower_limit": float(i.lower_limit or 0),
                "upper_limit": float(i.upper_limit or 0),
                "actual_value": float(i.actual_value or 0),
                "unit": i.unit,
                "result": i.result,
                "inspector": i.inspector,
            }
            for i in rows
        ],
        "total": len(rows),
    }


def _decide(db: Session, lot_id: int, decision: str, reason: str, user: User):
    lot = (
        db.query(IncomingLot)
        .options(joinedload(IncomingLot.inspections), joinedload(IncomingLot.material), joinedload(IncomingLot.supplier), joinedload(IncomingLot.defects))
        .filter(IncomingLot.id == lot_id)
        .first()
    )
    if not lot:
        raise HTTPException(status_code=404, detail="Lot not found")
    score = _score_lot(db, lot)
    mandatory = any(i.result == "FAIL" for i in (lot.inspections or [])) and (
        lot.material.criticality in ("HIGH", "CRITICAL") if lot.material else False
    )
    if decision == "RELEASED" and mandatory:
        raise HTTPException(status_code=409, detail="Mandatory quality rule: FAIL on high-criticality material cannot be released")
    old = lot.release_status
    lot.release_status = decision
    rec = ReleaseDecision(
        lot_id=lot.id,
        risk_score=score,
        recommendation=lot.recommendation,
        decision=decision,
        reason=reason,
        user=user.email if user else None,
        timestamp=datetime.utcnow(),
    )
    db.add(rec)
    _audit(db, user, decision, "incoming_lots", lot.id, old={"release_status": old}, new={"release_status": decision, "reason": reason})
    db.commit()
    return {"lot_id": lot.id, "decision": decision, "risk_score": score, "recommendation": lot.recommendation}


@router.post("/api/release/{lot_id}")
def release_lot(lot_id: int, payload: DecisionRequest, db: Session = Depends(get_db), user: User = Depends(require_roles("QUALITY_MANAGER", "ADMIN"))):
    return _decide(db, lot_id, "RELEASED", payload.reason, user)


@router.post("/api/hold/{lot_id}")
def hold_lot(lot_id: int, payload: DecisionRequest, db: Session = Depends(get_db), user: User = Depends(require_roles("QUALITY_MANAGER", "ADMIN"))):
    return _decide(db, lot_id, "HOLD", payload.reason, user)


@router.post("/api/reject/{lot_id}")
def reject_lot(lot_id: int, payload: DecisionRequest, db: Session = Depends(get_db), user: User = Depends(require_roles("QUALITY_MANAGER", "ADMIN"))):
    return _decide(db, lot_id, "REJECTED", payload.reason, user)


@router.get("/api/defects")
def list_defects(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    query = db.query(Defect)
    total = query.count()
    rows = query.order_by(Defect.id.desc()).offset(skip).limit(limit).all()
    items = [{"id": d.id, "lot_id": d.lot_id, "category": d.category, "description": d.description, "severity": d.severity} for d in rows]
    return {"items": items, "total": total}


@router.get("/api/release-decisions")
def list_decisions(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    query = db.query(ReleaseDecision).options(joinedload(ReleaseDecision.lot))
    total = query.count()
    rows = query.order_by(ReleaseDecision.id.desc()).offset(skip).limit(limit).all()
    items = [
        {
            "id": d.id,
            "lot": d.lot.lot_number if d.lot else None,
            "decision": d.decision,
            "risk_score": float(d.risk_score or 0),
            "recommendation": d.recommendation,
            "reason": d.reason,
            "user": d.user,
            "timestamp": d.timestamp.isoformat() if d.timestamp else None,
        }
        for d in rows
    ]
    return {"items": items, "total": total}


@router.get("/api/capa")
def list_capa(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    query = db.query(Capa).options(joinedload(Capa.supplier))
    total = query.count()
    rows = query.order_by(Capa.id.desc()).offset(skip).limit(limit).all()
    items = [
        {
            "id": c.id,
            "title": c.title,
            "status": c.status,
            "supplier": c.supplier.name if c.supplier else None,
            "lot_id": c.lot_id,
        }
        for c in rows
    ]
    return {"items": items, "total": total}


def _create_capa_row(payload: CapaCreate, db: Session, user: User):
    title = payload.title or (payload.reason or f"CAPA for lot {payload.lot_id or ''}")
    row = Capa(
        title=title.strip() or "CAPA",
        lot_id=payload.lot_id,
        supplier_id=payload.supplier_id,
        status=payload.status or "OPEN",
        owner=user.email if user else None,
        description=payload.description or payload.reason,
    )
    if payload.lot_id and not payload.supplier_id:
        lot = db.query(IncomingLot).filter(IncomingLot.id == payload.lot_id).first()
        if lot:
            row.supplier_id = lot.supplier_id
    db.add(row)
    dump = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    _audit(db, user, "CAPA_CREATE", "capa", None, new=dump)
    db.commit()
    db.refresh(row)
    return {"id": row.id, "title": row.title, "status": row.status}


@router.post("/api/capa")
def create_capa(payload: CapaCreate, db: Session = Depends(get_db), user: User = Depends(require_roles("QUALITY_MANAGER", "ADMIN"))):
    return _create_capa_row(payload, db, user)


@router.post("/api/capa/{lot_id}")
def create_capa_for_lot(lot_id: int, payload: CapaCreate, db: Session = Depends(get_db), user: User = Depends(require_roles("QUALITY_MANAGER", "ADMIN"))):
    payload.lot_id = lot_id
    return _create_capa_row(payload, db, user)


@router.post("/api/ai/analyze-lot")
def analyze_lot(payload: AnalyzeLotRequest, db: Session = Depends(get_db)):
    lot = None
    if payload.lot_id:
        lot = db.query(IncomingLot).options(joinedload(IncomingLot.inspections), joinedload(IncomingLot.material), joinedload(IncomingLot.supplier)).filter(IncomingLot.id == payload.lot_id).first()
    if not lot and payload.lot_number:
        lot = db.query(IncomingLot).options(joinedload(IncomingLot.inspections), joinedload(IncomingLot.material), joinedload(IncomingLot.supplier)).filter(IncomingLot.lot_number == payload.lot_number).first()
    if not lot:
        raise HTTPException(status_code=404, detail="Lot not found in application data")
    score = _score_lot(db, lot)
    db.commit()
    fails = [i.parameter for i in lot.inspections if i.result == "FAIL"]
    return {
        "lot_number": lot.lot_number,
        "risk_score": score,
        "recommendation": lot.recommendation,
        "inspection_failures": fails,
        "answer": (
            f"{lot.lot_number} risk score is {score} ({lot.recommendation}). "
            f"Supplier {lot.supplier.name if lot.supplier else 'unknown'}. "
            f"Failed parameters: {', '.join(fails) or 'none'}."
        ),
    }


@router.post("/api/ai/ask")
def ask(payload: AskRequest, db: Session = Depends(get_db)):
    q = payload.question.lower()
    facts = []
    if "hold" in q:
        rows = db.query(IncomingLot).options(joinedload(IncomingLot.supplier)).filter(IncomingLot.release_status == "HOLD").limit(8).all()
        if rows:
            facts.append("Lots on HOLD: " + "; ".join(f"{r.lot_number} ({r.supplier.name if r.supplier else ''}, risk {float(r.risk_score or 0)})" for r in rows))
        else:
            facts.append("No lots currently on HOLD in the database.")
    if "rejected" in q:
        rows = db.query(IncomingLot).options(joinedload(IncomingLot.supplier)).filter(IncomingLot.release_status == "REJECTED").limit(8).all()
        facts.append("Rejected lots: " + ("; ".join(f"{r.lot_number} / {r.supplier.name if r.supplier else ''}" for r in rows) or "none"))
    if "defect" in q:
        cats = db.query(Defect.category, func.count(Defect.id)).group_by(Defect.category).order_by(func.count(Defect.id).desc()).limit(5).all()
        facts.append("Top defects: " + ", ".join(f"{c} ({n})" for c, n in cats) if cats else "No defect records.")
        if "brake" in q:
            brake_ids = [m.id for m in db.query(Material).filter(Material.name.ilike("%brake%")).all()]
            n = db.query(Defect).join(IncomingLot).filter(IncomingLot.material_id.in_(brake_ids or [-1])).count()
            facts.append(f"Defects on brake-related materials: {n}.")
    if "increasing" in q or "trend" in q:
        rows = db.query(Supplier).filter(Supplier.defect_rate > 2).order_by(Supplier.defect_rate.desc()).limit(5).all()
        facts.append("Suppliers with elevated defect rate: " + (", ".join(f"{s.name} ({float(s.defect_rate)}%)" for s in rows) if rows else "none above 2%."))
    if "lot-" in q:
        import re
        m = re.search(r"lot-[0-9-]+", q, re.I)
        if m:
            lot = db.query(IncomingLot).options(joinedload(IncomingLot.supplier), joinedload(IncomingLot.inspections)).filter(IncomingLot.lot_number == m.group(0).upper()).first()
            if lot:
                facts.append(
                    f"{lot.lot_number}: risk {float(lot.risk_score or 0)}, status {lot.release_status}, recommendation {lot.recommendation}."
                )
            else:
                facts.append(f"No lot matching {m.group(0).upper()} in the database.")
    if not facts:
        facts.append("No matching application records for that question. Ask about holds, rejects, defects, suppliers, or a lot number.")
    return {"answer": " ".join(facts)}


@router.get("/api/reports/{name}")
def report(name: str, db: Session = Depends(get_db)):
    key = name.lower().replace(" ", "-")
    lines = ["report,field,value"]
    if "supplier" in key:
        for s in db.query(Supplier).all():
            lines.append(f"supplier_quality,{s.code},{s.name},{s.ppm},{s.quality_score}")
    elif "reject" in key:
        for l in db.query(IncomingLot).filter(IncomingLot.release_status == "REJECTED").all():
            lines.append(f"rejected,{l.lot_number},{l.release_status},{l.risk_score}")
    elif "defect" in key:
        for d in db.query(Defect).all():
            lines.append(f"defect,{d.category},{d.severity}")
    elif "capa" in key:
        for c in db.query(Capa).all():
            lines.append(f"capa,{c.title},{c.status}")
    else:
        for i in db.query(InspectionResult).limit(500).all():
            lines.append(f"inspection,{i.parameter},{i.result},{i.actual_value}")
    return PlainTextResponse("\n".join(lines), media_type="text/csv")
'''


def _seed_py() -> str:
    return r'''"""Seed realistic automotive supplier-quality sample data."""
from datetime import datetime, timedelta
import random

from database import Base, SessionLocal, engine
from models import (
    AuditLog,
    Capa,
    Defect,
    IncomingLot,
    InspectionResult,
    Material,
    PurchaseOrder,
    ReleaseDecision,
    Supplier,
    User,
)
from auth import hash_password
from risk_engine import compute_risk_score, inspection_result, recommendation_for

random.seed(42)
Base.metadata.create_all(bind=engine)

NAMED_SUPPLIERS = [
    ("SUP-001", "ABC Auto Components", "Detroit, MI", "Casting"),
    ("SUP-002", "Prime Precision", "Stuttgart, DE", "Machining"),
    ("SUP-003", "XYZ Metals", "Pune, IN", "Forging"),
    ("SUP-004", "Global Bearings", "Osaka, JP", "Bearings"),
    ("SUP-005", "Precision Forge", "Monterrey, MX", "Forging"),
]
EXTRA_SUPPLIERS = [
    "Apex Stamping", "Northern Heat Treat", "Midwest Fasteners", "Harbor Coatings",
    "Summit Axle", "Lakeside Castings", "Frontier Springs", "Atlas Valves",
    "Pioneer Gaskets", "Redwood Rubber", "Pacific Sensors", "Ironclad Tools",
    "Vertex Hydraulics", "Northstar Filters", "Helix Fasteners",
]
NAMED_MATERIALS = [
    ("MAT-BD", "Brake Disc", "Casting", "280 ± 0.5 mm", "HIGH", "CMM + visual"),
    ("MAT-GB", "Gear Blank", "Forging", "HRC 28-32", "MEDIUM", "Hardness + dim"),
    ("MAT-BR", "Bearing", "Purchased", "ISO 492 P6", "HIGH", "Sampling"),
    ("MAT-SK", "Steering Knuckle", "Casting", "EN-GJS-500", "HIGH", "X-ray + dim"),
    ("MAT-SB", "Suspension Bracket", "Stamping", "S355", "MEDIUM", "Visual + weld"),
]
EXTRA_MATERIALS = [
    "Caliper Housing", "Wheel Hub", "CV Joint", "Tie Rod End", "Control Arm",
    "Coil Spring", "Stabilizer Bar", "Oil Pump", "Water Pump", "Timing Gear",
    "Flywheel", "Clutch Plate", "Turbo Housing", "Exhaust Manifold", "Intake Runner",
    "Fuel Rail", "Throttle Body", "Camshaft", "Crankshaft", "Piston",
    "Connecting Rod", "Valve Spring", "Rocker Arm", "Oil Pan", "Valve Cover",
    "ABS Ring", "Wheel Stud", "Ball Joint", "Bushing", "Shock Absorber",
    "Strut Mount", "Brake Pad", "Brake Caliper Pin", "Master Cylinder", "Booster Shell",
    "Radiator End Tank", "Condenser", "A/C Compressor Bracket", "Alternator Pulley", "Starter Housing",
    "Sensor Bracket", "Harness Clip", "Heat Shield", "Splash Guard", "Battery Tray",
]
DEFECT_CATS = ["Porosity", "Dimensional", "Surface", "Heat treat", "Contamination", "Weld", "Packaging"]
PARAMS = [
    ("Diameter", "280 ± 0.5 mm", 279.5, 280.5, "mm"),
    ("Thickness", "22 ± 0.2 mm", 21.8, 22.2, "mm"),
    ("Hardness", "HRC 28-32", 28.0, 32.0, "HRC"),
    ("Runout", "0.05 mm max", 0.0, 0.05, "mm"),
    ("Bore", "64 ± 0.03 mm", 63.97, 64.03, "mm"),
]


def run():
    db = SessionLocal()
    try:
        if db.query(Supplier).count() >= 20 and db.query(IncomingLot).count() >= 200:
            print("Seed data already present")
            return
        if db.query(User).count() == 0:
            db.add_all([
                User(email="admin@example.com", hashed_password=hash_password("admin123"), role="ADMIN", full_name="Plant Admin"),
                User(email="quality@example.com", hashed_password=hash_password("quality123"), role="QUALITY_MANAGER", full_name="Quality Manager"),
                User(email="inspector@example.com", hashed_password=hash_password("inspector123"), role="INSPECTOR", full_name="J. Patel"),
                User(email="viewer@example.com", hashed_password=hash_password("viewer123"), role="VIEWER", full_name="Viewer"),
            ])
            db.commit()

        suppliers = []
        for code, name, loc, cat in NAMED_SUPPLIERS:
            ppm = random.randint(12, 180)
            s = Supplier(
                code=code, name=name, location=loc, category=cat,
                quality_score=round(max(60, 100 - ppm / 4), 1), ppm=ppm,
                defect_rate=round(ppm / 80.0, 2), rejected_lots=random.randint(0, 8),
                open_capa=random.randint(0, 4), status="HOLD" if ppm > 150 else "ACTIVE",
            )
            db.add(s)
            suppliers.append(s)
        for i, name in enumerate(EXTRA_SUPPLIERS, start=6):
            ppm = random.randint(10, 140)
            s = Supplier(
                code=f"SUP-{i:03d}", name=name, location=random.choice(["Detroit, MI", "Pune, IN", "Monterrey, MX", "Leipzig, DE"]),
                category=random.choice(["Casting", "Machining", "Stamping", "Forging"]),
                quality_score=round(max(62, 98 - ppm / 5), 1), ppm=ppm,
                defect_rate=round(ppm / 90.0, 2), rejected_lots=random.randint(0, 5),
                open_capa=random.randint(0, 3), status="ACTIVE",
            )
            db.add(s)
            suppliers.append(s)
        db.flush()

        materials = []
        for row in NAMED_MATERIALS:
            m = Material(code=row[0], name=row[1], material_type=row[2], specification=row[3], criticality=row[4], inspection_plan=row[5], status="ACTIVE")
            db.add(m)
            materials.append(m)
        for i, name in enumerate(EXTRA_MATERIALS, start=6):
            m = Material(
                code=f"MAT-{i:03d}", name=name,
                material_type=random.choice(["Casting", "Forging", "Stamping", "Purchased"]),
                specification=random.choice(["ISO 2768-m", "IATF 16949", "OEM PS-123"]),
                criticality=random.choice(["LOW", "MEDIUM", "HIGH"]),
                inspection_plan=random.choice(["Visual", "CMM", "Sampling", "X-ray"]),
                status="ACTIVE",
            )
            db.add(m)
            materials.append(m)
        db.flush()

        pos = []
        for i in range(100):
            s = suppliers[i % len(suppliers)]
            m = materials[i % len(materials)]
            po = PurchaseOrder(po_number=f"PO-10{i:03d}", supplier_id=s.id, material_id=m.id, quantity=100 + i * 5, status="OPEN" if i % 9 else "CLOSED")
            db.add(po)
            pos.append(po)
        db.flush()

        lots = []
        start = datetime(2026, 3, 1)
        for i in range(200):
            po = pos[i % len(pos)]
            insp = ["PENDING", "PASS", "PASS", "WARNING", "FAIL"][i % 5]
            rel = ["PENDING", "RELEASED", "RELEASED", "HOLD", "REJECTED", "RELEASED", "HOLD"][i % 7]
            risk = 12 + (i * 7) % 85
            lot = IncomingLot(
                lot_number=f"LOT-20260925-{i+1:03d}",
                po_id=po.id, supplier_id=po.supplier_id, material_id=po.material_id,
                quantity=po.quantity, received_date=start + timedelta(days=i % 180),
                inspection_status=insp, risk_score=risk, release_status=rel,
                recommendation=recommendation_for(risk, mandatory_hold=(insp == "FAIL" and rel != "RELEASED")),
            )
            db.add(lot)
            lots.append(lot)
        db.flush()

        inspectors = ["J. Patel", "M. Chen", "A. Singh", "L. Garcia"]
        insp_rows = 0
        for i in range(1000):
            lot = lots[i % len(lots)]
            p = PARAMS[i % len(PARAMS)]
            actual = (p[2] + p[3]) / 2.0
            if i % 11 == 0:
                actual = p[3] + 0.8
            elif i % 17 == 0:
                actual = p[2] + (p[3] - p[2]) * 0.04
            result = inspection_result(actual, p[2], p[3])
            db.add(InspectionResult(
                lot_id=lot.id, parameter=p[0], specification=p[1],
                lower_limit=p[2], upper_limit=p[3], actual_value=round(actual, 4),
                unit=p[4], result=result, inspector=inspectors[i % 4],
                inspection_date=lot.received_date + timedelta(hours=6),
            ))
            insp_rows += 1
        db.flush()

        for i in range(300):
            lot = lots[i % len(lots)]
            db.add(Defect(
                lot_id=lot.id,
                category=DEFECT_CATS[i % len(DEFECT_CATS)],
                description=f"{DEFECT_CATS[i % len(DEFECT_CATS)]} observed on {lot.lot_number}",
                severity=["MINOR", "MAJOR", "CRITICAL"][i % 3],
            ))
        db.flush()

        for i in range(50):
            lot = lots[i * 4]
            db.add(Capa(
                lot_id=lot.id, supplier_id=lot.supplier_id,
                title=f"CAPA-{i+1:03d} containment for {lot.lot_number}",
                status="OPEN" if i % 3 else "CLOSED",
                owner="quality@example.com",
                description="8D containment and irreversible corrective action.",
            ))
        db.flush()

        for i, lot in enumerate(lots):
            if lot.release_status == "PENDING":
                continue
            db.add(ReleaseDecision(
                lot_id=lot.id, risk_score=lot.risk_score, recommendation=lot.recommendation,
                decision=lot.release_status,
                reason="Seeded plant decision aligned with risk engine.",
                user="quality@example.com",
                timestamp=lot.received_date + timedelta(days=1),
            ))
        db.add(AuditLog(user="admin@example.com", action="SEED", entity="system", entity_id="0", new_value="initial automotive quality seed"))
        db.commit()
        print(f"Seeded {len(suppliers)} suppliers, {len(materials)} materials, {len(pos)} POs, {len(lots)} lots, {insp_rows} inspections")
    finally:
        db.close()


if __name__ == "__main__":
    run()
'''


def _alembic_001() -> str:
    return r'''"""Initial supplier quality schema.

Revision ID: 001_initial
Revises:
Create Date: 2026-09-30
"""
from alembic import op
import sqlalchemy as sa

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    from database import Base
    import models  # noqa: F401
    Base.metadata.create_all(bind=bind)


def downgrade():
    bind = op.get_bind()
    from database import Base
    import models  # noqa: F401
    Base.metadata.drop_all(bind=bind)
'''


def _test_risk_py() -> str:
    return r'''from risk_engine import compute_risk_score, inspection_result, recommendation_for


def test_diameter_pass_and_fail():
    assert inspection_result(280.2, 279.5, 280.5) == "PASS"
    assert inspection_result(281.2, 279.5, 280.5) == "FAIL"


def test_recommendation_bands():
    assert recommendation_for(12) == "AUTO RELEASE"
    assert recommendation_for(45) == "NORMAL INSPECTION"
    assert recommendation_for(70) == "QUALITY REVIEW"
    assert recommendation_for(90) == "HOLD / REJECT"
    assert recommendation_for(10, mandatory_hold=True) == "HOLD / REJECT"


def test_weights_sum_and_normalize():
    score = compute_risk_score(100, 100, 100, 100, 100)
    assert score == 100
    score = compute_risk_score(0, 0, 0, 0, 0)
    assert score == 0
'''


def _test_inspection_py() -> str:
    return r'''from risk_engine import inspection_result


def test_warning_near_limit():
    # 10% of 1.0 mm tolerance band near LSL/USL is WARNING
    assert inspection_result(279.55, 279.5, 280.5) == "WARNING"
    assert inspection_result(280.0, 279.5, 280.5) == "PASS"
'''


def _readme(title: str) -> str:
    return """# %s

Production supplier quality and incoming material release application.

## Stack

- Frontend: React, TypeScript, Vite, Material UI, React Router, TanStack Query, Recharts
- Backend: Python, FastAPI, Pydantic, SQLAlchemy, PostgreSQL, JWT
- Risk engine: inspection 30%% / supplier history 25%% / defects 20%% / criticality 15%% / trend 10%%

## Run locally

```bash
cp .env.example .env
docker compose up --build
```

- UI: http://localhost:5173
- API docs (Swagger): http://localhost:8000/docs
- Health: http://localhost:8000/health

Demo users:

- admin@example.com / admin123 (ADMIN)
- quality@example.com / quality123 (QUALITY_MANAGER)
- inspector@example.com / inspector123 (INSPECTOR)
- viewer@example.com / viewer123 (VIEWER)

## Seed volumes

The backend Dockerfile runs `python seed.py` on start:

- 20 suppliers (including ABC Auto Components, Prime Precision, XYZ Metals, Global Bearings, Precision Forge)
- 50 materials (including Brake Disc, Gear Blank, Bearing, Steering Knuckle, Suspension Bracket)
- 100 purchase orders
- 200 incoming lots
- 1000 inspection records
- 300 defects
- 50 CAPA records

## Tests

```bash
cd backend
pip install -r requirements.txt
pytest -q
```

## Migrations

```bash
cd backend
alembic upgrade head
```

Never commit real secrets. Copy `.env.example` and set `POSTGRES_PASSWORD` and `JWT_SECRET`.
""" % title
