from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime

from core.deps import get_db, role_required, get_current_user
from db.models.core import Organization, Tenant
from db.models.users import User

router = APIRouter(prefix="/organizations", tags=["organizations"])


class OrganizationCreate(BaseModel):
    tenant_id: Optional[str] = None
    name: str
    code: Optional[str] = None
    meta: Optional[dict] = None


class OrganizationUpdate(BaseModel):
    name: Optional[str] = None
    code: Optional[str] = None
    meta: Optional[dict] = None
    tenant_id: Optional[str] = None


@router.get("/", dependencies=[Depends(role_required("super_admin", "ummc_admin"))])
def get_all_organizations(db: Session = Depends(get_db)):
    """Get all organizations"""
    orgs = db.query(Organization).all()
    return [
        {
            "id": str(org.id),
            "tenant_id": str(org.tenant_id) if org.tenant_id else None,
            "name": org.name,
            "code": org.code,
            "meta": org.meta or {},
            "created_at": org.created_at.isoformat() if hasattr(org, 'created_at') and org.created_at else None,
            "updated_at": org.updated_at.isoformat() if hasattr(org, 'updated_at') and org.updated_at else None,
        }
        for org in orgs
    ]


@router.get("/{org_id}")
def get_organization(org_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get organization by ID"""
    org = db.query(Organization).filter(Organization.id == org_id).first()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    
    return {
        "id": str(org.id),
        "tenant_id": str(org.tenant_id) if org.tenant_id else None,
        "name": org.name,
        "code": org.code,
        "meta": org.meta or {},
        "created_at": org.created_at.isoformat() if hasattr(org, 'created_at') and org.created_at else None,
        "updated_at": org.updated_at.isoformat() if hasattr(org, 'updated_at') and org.updated_at else None,
    }


@router.post("/", dependencies=[Depends(role_required("super_admin", "ummc_admin"))])
def create_organization(payload: OrganizationCreate, db: Session = Depends(get_db)):
    """Create a new organization"""
    tenant_id = None
    if payload.tenant_id:
        tenant = db.query(Tenant).filter(Tenant.id == payload.tenant_id).first()
        if not tenant:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")
        tenant_id = tenant.id
    
    org = Organization(
        tenant_id=tenant_id,
        name=payload.name,
        code=payload.code,
        meta=payload.meta or {},
    )
    db.add(org)
    db.commit()
    db.refresh(org)
    
    return {
        "id": str(org.id),
        "tenant_id": str(org.tenant_id) if org.tenant_id else None,
        "name": org.name,
        "code": org.code,
        "meta": org.meta or {},
        "created_at": org.created_at.isoformat() if hasattr(org, 'created_at') and org.created_at else None,
        "updated_at": org.updated_at.isoformat() if hasattr(org, 'updated_at') and org.updated_at else None,
    }


@router.put("/{org_id}", dependencies=[Depends(role_required("super_admin", "ummc_admin"))])
def update_organization(org_id: str, payload: OrganizationUpdate, db: Session = Depends(get_db)):
    """Update an organization"""
    org = db.query(Organization).filter(Organization.id == org_id).first()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    
    if payload.name is not None:
        org.name = payload.name
    if payload.code is not None:
        org.code = payload.code
    if payload.meta is not None:
        org.meta = payload.meta
    if payload.tenant_id is not None:
        if payload.tenant_id:
            tenant = db.query(Tenant).filter(Tenant.id == payload.tenant_id).first()
            if not tenant:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")
            org.tenant_id = tenant.id
        else:
            org.tenant_id = None
    
    db.commit()
    db.refresh(org)
    
    return {
        "id": str(org.id),
        "tenant_id": str(org.tenant_id) if org.tenant_id else None,
        "name": org.name,
        "code": org.code,
        "meta": org.meta or {},
        "created_at": org.created_at.isoformat() if hasattr(org, 'created_at') and org.created_at else None,
        "updated_at": org.updated_at.isoformat() if hasattr(org, 'updated_at') and org.updated_at else None,
    }


@router.delete("/{org_id}", dependencies=[Depends(role_required("super_admin"))])
def delete_organization(org_id: str, db: Session = Depends(get_db)):
    """Delete an organization"""
    org = db.query(Organization).filter(Organization.id == org_id).first()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    
    db.delete(org)
    db.commit()
    return {"message": "Organization deleted successfully"}


@router.get("/tenants/", dependencies=[Depends(role_required("super_admin"))])
def get_all_tenants(db: Session = Depends(get_db)):
    """Get all tenants"""
    tenants = db.query(Tenant).all()
    return [
        {
            "id": str(t.id),
            "name": t.name,
            "slug": t.name.lower().replace(" ", "-"),  # Generate slug from name
            "meta": t.meta or {},
        }
        for t in tenants
    ]
