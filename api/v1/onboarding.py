"""
Onboarding ("Join the platform") API.

Public submission for hospitals and expert clinicians to express interest, plus
admin review. Hospitals indicate whether they already have a data-entry/EMR
system: those that do are onboarded as a NextGen hospital-admin to manage their
own data; those that don't can use the platform's data-entry feature.
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc, text
import uuid
import logging

from core.deps import get_db, get_current_user
from core.rate_limit import limiter
from db.models.onboarding import OnboardingRequest
from db.models.users import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/onboarding", tags=["onboarding"])

ADMIN_ROLES = {"super_admin", "ummc_admin", "hospital_admin"}


def _require_admin(user: User):
    if not any(r.slug in ADMIN_ROLES for r in user.roles):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")


class OnboardingApply(BaseModel):
    request_type: str = Field(..., description="hospital | expert_doctor")
    contact_name: str = Field(..., min_length=2)
    email: str = Field(..., min_length=5)
    phone: Optional[str] = None
    organization: Optional[str] = None
    country: Optional[str] = None
    specialty: Optional[str] = None
    has_data_entry_system: Optional[bool] = None
    estimated_volume: Optional[str] = None
    message: Optional[str] = None


class StatusUpdate(BaseModel):
    status: str = Field(..., description="NEW | CONTACTED | APPROVED | REJECTED")


@router.post("/apply", status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
def apply(payload: OnboardingApply, db: Session = Depends(get_db), request: Request = None):
    """Public: submit a hospital or expert-doctor onboarding request."""
    rtype = payload.request_type.strip().lower()
    if rtype not in ("hospital", "expert_doctor"):
        raise HTTPException(status_code=400, detail="request_type must be 'hospital' or 'expert_doctor'")

    rec = OnboardingRequest(
        request_type=rtype,
        contact_name=payload.contact_name.strip(),
        email=payload.email.strip(),
        phone=payload.phone,
        organization=payload.organization,
        country=payload.country,
        specialty=payload.specialty,
        has_data_entry_system=payload.has_data_entry_system,
        estimated_volume=payload.estimated_volume,
        message=payload.message,
        status="NEW",
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    if rtype == "hospital":
        path = ("Your hospital will be set up with a NextGen hospital-admin account to manage your own data."
                if payload.has_data_entry_system
                else "Your hospital will be enabled with the platform data-entry feature.")
    else:
        path = "Our team will review your clinical profile for expert reviewer onboarding."

    return {
        "id": str(rec.id),
        "status": rec.status,
        "message": "Thank you. Your request has been received.",
        "next_step": path,
    }


@router.get("/admin")
def list_requests(
    request_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Admin: list onboarding requests."""
    _require_admin(current_user)
    q = db.query(OnboardingRequest)
    if request_type:
        q = q.filter(OnboardingRequest.request_type == request_type.lower())
    rows = q.order_by(desc(OnboardingRequest.created_at)).limit(500).all()
    return {
        "total": len(rows),
        "requests": [
            {
                "id": str(r.id),
                "request_type": r.request_type,
                "contact_name": r.contact_name,
                "email": r.email,
                "phone": r.phone,
                "organization": r.organization,
                "country": r.country,
                "specialty": r.specialty,
                "has_data_entry_system": r.has_data_entry_system,
                "estimated_volume": r.estimated_volume,
                "message": r.message,
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@router.post("/admin/{request_id}/status")
def update_status(
    request_id: str,
    payload: StatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Admin: update an onboarding request's status."""
    _require_admin(current_user)
    valid = {"NEW", "CONTACTED", "APPROVED", "REJECTED"}
    if payload.status not in valid:
        raise HTTPException(status_code=400, detail=f"status must be one of {sorted(valid)}")
    rec = db.query(OnboardingRequest).filter(OnboardingRequest.id == request_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Request not found")
    rec.status = payload.status
    db.commit()
    return {"id": str(rec.id), "status": rec.status}


@router.post("/admin/{request_id}/provision")
def provision_hospital(
    request_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Approve a hospital onboarding request and auto-provision it:
    creates an organization, a hospital-admin account, and (if the hospital has
    no EMR) a data-entry account. Returns one-time temporary passwords.
    """
    _require_admin(current_user)
    import secrets
    from core.security import get_password_hash
    from db.models.core import Tenant, Organization
    from db.models.rbac import Role

    rec = db.query(OnboardingRequest).filter(OnboardingRequest.id == request_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Request not found")
    if rec.request_type != "hospital":
        raise HTTPException(status_code=400, detail="Only hospital requests can be provisioned")

    org_name = (rec.organization or rec.contact_name or "Hospital").strip()
    code = "".join(ch for ch in org_name.upper() if ch.isalnum())[:12] or "HOSP"

    # Tenant + organization
    tenant = db.query(Tenant).filter(Tenant.name == org_name).first()
    if not tenant:
        tenant = Tenant(name=org_name, meta={"onboarded": True})
        db.add(tenant); db.flush()
    org = db.query(Organization).filter(Organization.code == code).first()
    if not org:
        org = Organization(tenant_id=tenant.id, name=org_name, code=code, meta={"onboarded": True})
        db.add(org); db.flush()

    def _make_user(email: str, name: str, role_slug: str) -> Optional[dict]:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            return None
        temp = f"NG-{secrets.token_hex(4)}"
        u = User(id=uuid.uuid4(), tenant_id=tenant.id, organization_id=org.id, email=email,
                 full_name=name, hashed_password=get_password_hash(temp),
                 is_active=True, is_email_verified=True)
        db.add(u); db.flush()
        role = db.query(Role).filter(Role.slug == role_slug).first()
        if role:
            db.execute(
                text("INSERT INTO rbac.user_roles (user_id, role_id) VALUES (:u,:r) ON CONFLICT DO NOTHING"),
                {"u": str(u.id), "r": str(role.id)})
        return {"email": email, "temp_password": temp, "role": role_slug}

    created_accounts = []
    admin_acct = _make_user(rec.email, rec.contact_name or "Hospital Admin", "hospital_admin")
    if admin_acct:
        created_accounts.append(admin_acct)
    if not rec.has_data_entry_system:
        de_email = f"dataentry@{code.lower()}.nextgen.health"
        de_acct = _make_user(de_email, f"{org_name} Data Entry", "registry_editor")
        if de_acct:
            created_accounts.append(de_acct)

    rec.status = "APPROVED"
    db.commit()

    return {
        "request_id": str(rec.id),
        "organization": {"id": str(org.id), "name": org.name, "code": org.code},
        "accounts": created_accounts,
        "note": "Share these one-time passwords securely; users should change them on first login.",
        "data_path": ("Self-managed: hospital-admin manages data and can mint ingestion API keys."
                      if rec.has_data_entry_system else
                      "Data-entry workspace enabled with a registry-editor account."),
    }
