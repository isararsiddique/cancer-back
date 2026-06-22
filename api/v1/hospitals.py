"""
Hospital Management API
Super admin can create, manage, and monitor multiple hospitals.
Each hospital gets auto-provisioned: tenant, organization, S3 prefix, admin user.
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, text
from pydantic import BaseModel, EmailStr
from datetime import datetime
import uuid
import logging
import os
import boto3

from core.deps import get_db, get_current_user, role_required
from core.security import get_password_hash
from core.audit import log_event
from db.models.users import User
from db.models.rbac import Role, user_roles
from db.models.core import Organization, Tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hospitals", tags=["hospitals"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class HospitalCreate(BaseModel):
    """Create a new hospital with auto-provisioning"""
    name: str
    code: str  # Short code like 'UMMC', 'HKL', 'GH_PEN'
    address: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = "Malaysia"
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    # Admin user for this hospital
    admin_email: EmailStr
    admin_password: str
    admin_name: Optional[str] = None


class HospitalUpdate(BaseModel):
    """Update hospital details"""
    name: Optional[str] = None
    code: Optional[str] = None
    address: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    is_active: Optional[bool] = None


class HospitalStaffCreate(BaseModel):
    """Add staff to a hospital"""
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role_slug: str  # 'hospital_staff', 'data_entry', 'registry_editor', 'registry_viewer'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _format_hospital(org: Organization, db: Session) -> dict:
    """Format hospital organization for API response"""
    # Count users in this org
    user_count = db.query(func.count(User.id)).filter(
        User.organization_id == org.id,
        User.is_active == True
    ).scalar() or 0

    meta = org.meta or {}

    return {
        "id": str(org.id),
        "tenant_id": str(org.tenant_id) if org.tenant_id else None,
        "name": org.name,
        "code": org.code,
        "address": meta.get("address"),
        "state": meta.get("state"),
        "country": meta.get("country", "Malaysia"),
        "contact_email": meta.get("contact_email"),
        "contact_phone": meta.get("contact_phone"),
        "is_active": meta.get("is_active", True),
        "s3_prefix": meta.get("s3_prefix"),
        "user_count": user_count,
        "created_at": meta.get("created_at"),
    }


def _provision_s3_prefix(org_id: str, org_code: str) -> str:
    """Create S3 prefix structure for hospital data isolation"""
    s3_bucket = os.getenv("ANALYTICS_S3_BUCKET", "cancer-registry-analytics")
    prefix = f"hospitals/{org_code.lower()}/"

    try:
        s3_client = boto3.client('s3', region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        # Create marker files for the hospital's S3 structure
        for sub_prefix in [f"{prefix}raw/patients/", f"{prefix}curated/patients_anonymized/"]:
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=f"{sub_prefix}.keep",
                Body=b""
            )
        logger.info(f"S3 prefix provisioned: s3://{s3_bucket}/{prefix}")
    except Exception as e:
        logger.warning(f"S3 provisioning failed (non-fatal): {e}")

    return prefix


# ============================================================================
# HOSPITAL CRUD ENDPOINTS
# ============================================================================

@router.post("/", dependencies=[Depends(role_required("super_admin"))])
def create_hospital(
    payload: HospitalCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new hospital with full auto-provisioning:
    1. Creates Tenant record
    2. Creates Organization record
    3. Provisions S3 prefix for data isolation
    4. Creates hospital admin user with hospital_admin role
    """
    try:
        # Check if org code already exists
        existing_org = db.query(Organization).filter(Organization.code == payload.code).first()
        if existing_org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Hospital with code '{payload.code}' already exists"
            )

        # Check if admin email already exists
        existing_user = db.query(User).filter(User.email == payload.admin_email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"User with email '{payload.admin_email}' already exists"
            )

        # 1. Create Tenant
        tenant = Tenant(
            name=payload.name,
            meta={"type": "hospital", "code": payload.code}
        )
        db.add(tenant)
        db.flush()

        # 2. Create Organization
        org = Organization(
            tenant_id=tenant.id,
            name=payload.name,
            code=payload.code,
            meta={
                "type": "hospital",
                "address": payload.address,
                "state": payload.state,
                "country": payload.country or "Malaysia",
                "contact_email": payload.contact_email,
                "contact_phone": payload.contact_phone,
                "is_active": True,
                "created_at": datetime.utcnow().isoformat(),
                "created_by": str(current_user.id),
            }
        )
        db.add(org)
        db.flush()

        # 3. Provision S3 prefix
        s3_prefix = _provision_s3_prefix(str(org.id), payload.code)
        org.meta = {**org.meta, "s3_prefix": s3_prefix}

        # 4. Create hospital admin user
        # Find hospital_admin role for the hospital admin user
        admin_role = db.query(Role).filter(Role.slug == "hospital_admin").first()
        if not admin_role:
            # Fallback to hospital_staff
            admin_role = db.query(Role).filter(Role.slug == "hospital_staff").first()
        if not admin_role:
            # Fallback to registry_editor
            admin_role = db.query(Role).filter(Role.slug == "registry_editor").first()

        if not admin_role:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Required role 'hospital_staff' not found in database. Please seed roles first."
            )

        admin_user = User(
            email=payload.admin_email,
            full_name=payload.admin_name or f"{payload.name} Admin",
            hashed_password=get_password_hash(payload.admin_password),
            tenant_id=tenant.id,
            organization_id=org.id,
            is_active=True,
        )
        admin_user.roles = [admin_role]
        db.add(admin_user)
        db.flush()

        # Audit log
        try:
            log_event(
                db=db,
                user_id=current_user.id,
                action_type="hospital_created",
                resource_type="organization",
                resource_id=org.id,
                change_summary=f"Hospital created: {payload.name} ({payload.code})",
                change_details={
                    "hospital_name": payload.name,
                    "hospital_code": payload.code,
                    "tenant_id": str(tenant.id),
                    "org_id": str(org.id),
                    "admin_email": payload.admin_email,
                    "s3_prefix": s3_prefix,
                    "created_by": current_user.email,
                },
                category="hospital_management"
            )
        except Exception as e:
            logger.warning(f"Audit logging failed: {e}")

        db.commit()
        db.refresh(org)

        return {
            "message": f"Hospital '{payload.name}' created successfully",
            "hospital": _format_hospital(org, db),
            "admin_user": {
                "id": str(admin_user.id),
                "email": admin_user.email,
                "full_name": admin_user.full_name,
                "role": admin_role.slug,
            },
            "provisioned": {
                "tenant_id": str(tenant.id),
                "organization_id": str(org.id),
                "s3_prefix": s3_prefix,
            }
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create hospital: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create hospital: {str(e)}"
        )


@router.get("/", dependencies=[Depends(role_required("super_admin"))])
def list_hospitals(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all hospitals with stats"""
    try:
        query = db.query(Organization).filter(
            text("core.organizations.metadata->>'type' = 'hospital'")
        )

        if search:
            search_term = f"%{search}%"
            query = query.filter(
                Organization.name.ilike(search_term) |
                Organization.code.ilike(search_term)
            )

        total = query.count()
        hospitals = query.order_by(Organization.name).offset(skip).limit(limit).all()

        result = []
        for org in hospitals:
            hospital_data = _format_hospital(org, db)

            # Get patient count for this hospital from S3/Athena
            # (lightweight: just count users for now, patient count comes from stats endpoint)
            result.append(hospital_data)

        return {
            "hospitals": result,
            "total": total,
            "skip": skip,
            "limit": limit,
        }

    except Exception as e:
        logger.error(f"Failed to list hospitals: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list hospitals: {str(e)}"
        )


@router.get("/{hospital_id}", dependencies=[Depends(role_required("super_admin", "hospital_admin", "hospital_staff"))])
def get_hospital(
    hospital_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get hospital details with stats"""
    try:
        org = db.query(Organization).filter(Organization.id == hospital_id).first()
        if not org:
            raise HTTPException(status_code=404, detail="Hospital not found")

        # Non-super-admin can only view their own hospital
        if not any(r.slug == "super_admin" for r in current_user.roles):
            if str(current_user.organization_id) != hospital_id:
                raise HTTPException(status_code=403, detail="Access denied")

        hospital = _format_hospital(org, db)

        # Get staff list
        staff = db.query(User).options(joinedload(User.roles)).filter(
            User.organization_id == org.id,
            User.is_active == True
        ).all()

        hospital["staff"] = [
            {
                "id": str(u.id),
                "email": u.email,
                "full_name": u.full_name,
                "roles": [{"id": str(r.id), "name": r.name, "slug": r.slug} for r in u.roles],
                "is_active": u.is_active,
            }
            for u in staff
        ]

        return hospital

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get hospital: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get hospital: {str(e)}")


@router.put("/{hospital_id}", dependencies=[Depends(role_required("super_admin"))])
def update_hospital(
    hospital_id: str,
    payload: HospitalUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update hospital details"""
    try:
        org = db.query(Organization).filter(Organization.id == hospital_id).first()
        if not org:
            raise HTTPException(status_code=404, detail="Hospital not found")

        if payload.name is not None:
            org.name = payload.name
            # Also update tenant name
            if org.tenant_id:
                tenant = db.query(Tenant).filter(Tenant.id == org.tenant_id).first()
                if tenant:
                    tenant.name = payload.name

        if payload.code is not None:
            org.code = payload.code

        # Update meta fields
        meta = org.meta or {}
        for field in ["address", "state", "country", "contact_email", "contact_phone", "is_active"]:
            value = getattr(payload, field, None)
            if value is not None:
                meta[field] = value

        meta["updated_at"] = datetime.utcnow().isoformat()
        meta["updated_by"] = str(current_user.id)
        org.meta = meta

        db.commit()
        db.refresh(org)

        return {
            "message": "Hospital updated successfully",
            "hospital": _format_hospital(org, db),
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update hospital: {str(e)}")


@router.delete("/{hospital_id}", dependencies=[Depends(role_required("super_admin"))])
def deactivate_hospital(
    hospital_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Deactivate a hospital (soft delete - data preserved)"""
    try:
        org = db.query(Organization).filter(Organization.id == hospital_id).first()
        if not org:
            raise HTTPException(status_code=404, detail="Hospital not found")

        meta = org.meta or {}
        meta["is_active"] = False
        meta["deactivated_at"] = datetime.utcnow().isoformat()
        meta["deactivated_by"] = str(current_user.id)
        org.meta = meta

        # Deactivate all users in this hospital
        db.query(User).filter(User.organization_id == org.id).update(
            {"is_active": False}, synchronize_session=False
        )

        try:
            log_event(
                db=db,
                user_id=current_user.id,
                action_type="hospital_deactivated",
                resource_type="organization",
                resource_id=org.id,
                change_summary=f"Hospital deactivated: {org.name}",
                change_details={"deactivated_by": current_user.email},
                category="hospital_management"
            )
        except Exception:
            pass

        db.commit()

        return {"message": f"Hospital '{org.name}' deactivated successfully"}

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to deactivate hospital: {str(e)}")


# ============================================================================
# HOSPITAL STAFF MANAGEMENT
# ============================================================================

@router.post("/{hospital_id}/staff", dependencies=[Depends(role_required("super_admin", "hospital_admin"))])
def add_hospital_staff(
    hospital_id: str,
    payload: HospitalStaffCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Add a staff member to a hospital"""
    try:
        org = db.query(Organization).filter(Organization.id == hospital_id).first()
        if not org:
            raise HTTPException(status_code=404, detail="Hospital not found")

        # Hospital admin can only add staff to their own hospital
        if not any(r.slug == "super_admin" for r in current_user.roles):
            if str(current_user.organization_id) != hospital_id:
                raise HTTPException(status_code=403, detail="You can only add staff to your own hospital")

        # Check email
        existing = db.query(User).filter(User.email == payload.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already exists")

        # Find role
        allowed_roles = ['hospital_admin', 'hospital_staff', 'data_entry', 'registry_editor', 'registry_viewer']
        if payload.role_slug not in allowed_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role. Allowed: {', '.join(allowed_roles)}"
            )

        role = db.query(Role).filter(Role.slug == payload.role_slug).first()
        if not role:
            raise HTTPException(status_code=404, detail=f"Role '{payload.role_slug}' not found")

        user = User(
            email=payload.email,
            full_name=payload.full_name,
            hashed_password=get_password_hash(payload.password),
            tenant_id=org.tenant_id,
            organization_id=org.id,
            is_active=True,
        )
        user.roles = [role]
        db.add(user)

        try:
            log_event(
                db=db,
                user_id=current_user.id,
                action_type="staff_added",
                resource_type="user",
                resource_id=user.id,
                change_summary=f"Staff added to {org.name}: {payload.email}",
                change_details={
                    "hospital": org.name,
                    "staff_email": payload.email,
                    "role": payload.role_slug,
                },
                category="hospital_management"
            )
        except Exception:
            pass

        db.commit()
        db.refresh(user)

        return {
            "message": f"Staff member added to {org.name}",
            "user": {
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "role": payload.role_slug,
                "hospital": org.name,
            }
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to add staff: {str(e)}")


@router.get("/{hospital_id}/stats", dependencies=[Depends(role_required("super_admin", "hospital_admin", "hospital_staff"))])
def get_hospital_stats(
    hospital_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get statistics for a specific hospital"""
    try:
        org = db.query(Organization).filter(Organization.id == hospital_id).first()
        if not org:
            raise HTTPException(status_code=404, detail="Hospital not found")

        # Non-super-admin can only view their own hospital stats
        if not any(r.slug == "super_admin" for r in current_user.roles):
            if str(current_user.organization_id) != hospital_id:
                raise HTTPException(status_code=403, detail="Access denied")

        # User stats
        total_staff = db.query(func.count(User.id)).filter(
            User.organization_id == org.id
        ).scalar() or 0

        active_staff = db.query(func.count(User.id)).filter(
            User.organization_id == org.id,
            User.is_active == True
        ).scalar() or 0

        # Patient count from PostgreSQL
        patient_count = 0
        try:
            from db.models.registry import Patient
            patient_count = db.query(func.count(Patient.id)).filter(
                Patient.organization_id == org.id
            ).scalar() or 0
        except Exception as e:
            logger.warning(f"Patient count query failed: {e}")

        return {
            "hospital_id": str(org.id),
            "hospital_name": org.name,
            "hospital_code": org.code,
            "total_staff": total_staff,
            "active_staff": active_staff,
            "total_patients": patient_count,
            "is_active": (org.meta or {}).get("is_active", True),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
