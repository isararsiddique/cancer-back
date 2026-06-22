"""
Per-hospital data ingestion.

- Admins / hospital-admins mint scoped API keys for an organization.
- Hospitals push patient records to POST /ingest/patients with an X-API-Key
  header; records are inserted scoped to that organization (no JWT needed),
  enabling system-to-system data exchange for hospitals with their own EMR.
"""
from typing import Optional, List
from datetime import date, datetime
import hashlib
import secrets
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Header, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.deps import get_db, get_current_user
from core.rate_limit import limiter
from db.models.users import User
from db.models.onboarding import HospitalApiKey
from db.models.registry import Patient

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])
ADMIN_ROLES = {"super_admin", "ummc_admin", "hospital_admin"}


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# API key management (JWT-authenticated, admin / hospital-admin)
# ---------------------------------------------------------------------------
class ApiKeyCreate(BaseModel):
    label: Optional[str] = None
    organization_id: Optional[str] = None  # super_admin may target any org


@router.post("/hospitals/api-keys", status_code=status.HTTP_201_CREATED)
def create_api_key(payload: ApiKeyCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    roles = {r.slug for r in current_user.roles}
    if not (roles & ADMIN_ROLES):
        raise HTTPException(status_code=403, detail="Admin or hospital-admin role required")

    # Determine target org: super/ummc admin may specify; hospital_admin is locked to own org
    if "super_admin" in roles or "ummc_admin" in roles:
        org_id = payload.organization_id or (str(current_user.organization_id) if current_user.organization_id else None)
    else:
        org_id = str(current_user.organization_id) if current_user.organization_id else None
    if not org_id:
        raise HTTPException(status_code=400, detail="No organization to scope the key to")

    raw = f"ng_{secrets.token_hex(24)}"
    rec = HospitalApiKey(
        organization_id=org_id,
        tenant_id=current_user.tenant_id,
        label=payload.label or "Hospital ingestion key",
        key_prefix=raw[:10],
        key_hash=_hash_key(raw),
        is_active=True,
        created_by=current_user.id,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return {
        "id": str(rec.id),
        "organization_id": org_id,
        "label": rec.label,
        "api_key": raw,  # shown once
        "key_prefix": rec.key_prefix,
        "note": "Store this key securely. It will not be shown again.",
    }


@router.get("/hospitals/api-keys")
def list_api_keys(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    roles = {r.slug for r in current_user.roles}
    if not (roles & ADMIN_ROLES):
        raise HTTPException(status_code=403, detail="Admin or hospital-admin role required")
    q = db.query(HospitalApiKey)
    if not ({"super_admin", "ummc_admin"} & roles):
        q = q.filter(HospitalApiKey.organization_id == current_user.organization_id)
    keys = q.order_by(HospitalApiKey.created_at.desc()).all()
    return {
        "keys": [
            {
                "id": str(k.id), "organization_id": str(k.organization_id), "label": k.label,
                "key_prefix": k.key_prefix, "is_active": k.is_active,
                "created_at": k.created_at.isoformat() if k.created_at else None,
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
            } for k in keys
        ]
    }


@router.post("/hospitals/api-keys/{key_id}/revoke")
def revoke_api_key(key_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    roles = {r.slug for r in current_user.roles}
    if not (roles & ADMIN_ROLES):
        raise HTTPException(status_code=403, detail="Admin or hospital-admin role required")
    k = db.query(HospitalApiKey).filter(HospitalApiKey.id == key_id).first()
    if not k:
        raise HTTPException(status_code=404, detail="Key not found")
    if not ({"super_admin", "ummc_admin"} & roles) and str(k.organization_id) != str(current_user.organization_id):
        raise HTTPException(status_code=403, detail="Cannot revoke a key for another organization")
    k.is_active = False
    db.commit()
    return {"id": str(k.id), "is_active": False}


# ---------------------------------------------------------------------------
# Ingestion (API-key authenticated, system-to-system)
# ---------------------------------------------------------------------------
class IngestPatient(BaseModel):
    patient_name: str
    diagnosis_date: date
    icd11_main_code: str
    patient_id: Optional[str] = None
    gender: Optional[str] = None
    date_of_birth: Optional[date] = None
    age_at_diagnosis: Optional[int] = None
    nationality: Optional[str] = None
    icd11_description: Optional[str] = None
    t_category: Optional[str] = None
    n_category: Optional[str] = None
    m_category: Optional[str] = None
    vital_status: Optional[str] = None
    surgery_done: Optional[bool] = None
    chemotherapy_done: Optional[bool] = None
    radiotherapy_done: Optional[bool] = None


class IngestRequest(BaseModel):
    patients: List[IngestPatient] = Field(..., min_length=1, max_length=1000)


def _resolve_key(db: Session, x_api_key: Optional[str]) -> HospitalApiKey:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    rec = db.query(HospitalApiKey).filter(
        HospitalApiKey.key_hash == _hash_key(x_api_key.strip()),
        HospitalApiKey.is_active == True,
    ).first()
    if not rec:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")
    return rec


@router.post("/ingest/patients")
@limiter.limit("60/minute")
def ingest_patients(
    body: IngestRequest,
    db: Session = Depends(get_db),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    request: Request = None,
):
    """Insert patient records scoped to the API key's organization (no JWT)."""
    key = _resolve_key(db, x_api_key)
    created, failed = 0, []
    for idx, p in enumerate(body.patients):
        try:
            data = p.model_dump(exclude_unset=True)
            if not data.get("age_at_diagnosis") and data.get("date_of_birth"):
                dob, diag = data["date_of_birth"], data["diagnosis_date"]
                data["age_at_diagnosis"] = diag.year - dob.year - ((diag.month, diag.day) < (dob.month, dob.day))
            patient = Patient(
                id=uuid.uuid4(),
                tenant_id=key.tenant_id,
                organization_id=key.organization_id,
                data_source="HospitalAPI",
                entry_mode="API",
                is_active=True,
                **data,
            )
            db.add(patient)
            created += 1
        except Exception as e:
            failed.append({"index": idx, "error": str(e)})
    try:
        key.last_used_at = datetime.now()
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    return {
        "organization_id": str(key.organization_id),
        "received": len(body.patients),
        "created": created,
        "failed": failed,
    }
