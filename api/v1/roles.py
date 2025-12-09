from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, OperationalError
from pydantic import BaseModel
import logging
import uuid

from core.deps import get_db, get_current_user, role_required
from core.audit import log_event
from db.models.rbac import Role, Permission, user_roles
from db.models.users import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/roles", tags=["roles"])


class RoleCreate(BaseModel):
    name: str
    slug: str
    description: Optional[str] = None
    tenant_scoped: bool = False
    permission_ids: Optional[List[str]] = None


class RoleUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tenant_scoped: Optional[bool] = None
    permission_ids: Optional[List[str]] = None


class RoleResponse(BaseModel):
    id: str
    name: str
    slug: str
    description: Optional[str]
    tenant_scoped: bool
    permissions: List[dict]

    class Config:
        from_attributes = True


class PermissionResponse(BaseModel):
    id: str
    name: str
    code: str
    description: Optional[str]

    class Config:
        from_attributes = True


@router.get("/", dependencies=[Depends(role_required("super_admin"))])
def get_all_roles(db: Session = Depends(get_db)):
    """
    Get all roles with user counts.
    Requirements: 4.1, 4.6
    """
    # Validate database connection (Requirement 5.2)
    try:
        db.execute("SELECT 1")
    except OperationalError as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection unavailable"
        )
    
    roles = db.query(Role).all()
    
    result = []
    for role in roles:
        # Get user count for this role (Requirement 4.6)
        user_count = db.query(user_roles).filter(user_roles.c.role_id == role.id).count()
        
        result.append({
            "id": str(role.id),
            "name": role.name,
            "slug": role.slug,
            "description": role.description,
            "tenant_scoped": role.tenant_scoped,
            "user_count": user_count,  # Requirement 4.6
            "permissions": [
                {
                    "id": str(p.id),
                    "name": p.name,
                    "code": p.code,
                    "description": p.description,
                }
                for p in role.permissions
            ],
        })
    
    return result


@router.get("/{role_id}")
def get_role(role_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get role by ID"""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")
    
    return {
        "id": str(role.id),
        "name": role.name,
        "slug": role.slug,
        "description": role.description,
        "tenant_scoped": role.tenant_scoped,
        "permissions": [
            {
                "id": str(p.id),
                "name": p.name,
                "code": p.code,
                "description": p.description,
            }
            for p in role.permissions
        ],
    }


@router.post("/", dependencies=[Depends(role_required("super_admin"))])
def create_role(
    payload: RoleCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new role with database transaction and audit logging.
    Requirements: 4.2, 4.7, 5.1, 5.2
    """
    # Validate database connection (Requirement 5.2)
    try:
        db.execute("SELECT 1")
    except OperationalError as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection unavailable"
        )
    
    try:
        # Use database transaction (Requirement 5.1)
        # Check if slug already exists (Requirement 4.2)
        existing = db.query(Role).filter(Role.slug == payload.slug).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Role slug already exists"
            )
        
        role = Role(
            name=payload.name,
            slug=payload.slug,
            description=payload.description,
            tenant_scoped=payload.tenant_scoped,
        )
        
        # Add permissions if provided
        if payload.permission_ids:
            permissions = db.query(Permission).filter(
                Permission.id.in_(payload.permission_ids)
            ).all()
            role.permissions = permissions
        
        db.add(role)
        db.flush()  # Get role ID before commit
        
        # Audit logging (Requirement 4.7)
        try:
            log_event(
                db=db,
                user_id=current_user.id,
                action_type="role_created",
                resource_type="role",
                resource_id=role.id,
                change_summary=f"Role created: {role.name}",
                change_details={
                    "role_name": role.name,
                    "role_slug": role.slug,
                    "permissions": [str(p.id) for p in role.permissions]
                },
                category="authorization"
            )
        except Exception as e:
            logger.warning(f"Audit logging failed: {str(e)}")
        
        db.commit()
        db.refresh(role)
        
        # TODO: Clear RBAC cache (Requirement 4.5)
        
        return {
            "id": str(role.id),
            "name": role.name,
            "slug": role.slug,
            "description": role.description,
            "tenant_scoped": role.tenant_scoped,
            "permissions": [
                {
                    "id": str(p.id),
                    "name": p.name,
                    "code": p.code,
                    "description": p.description,
                }
                for p in role.permissions
            ],
        }
        
    except HTTPException:
        db.rollback()
        raise
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Database integrity error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role creation failed due to database constraint"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Role creation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create role: {str(e)}"
        )


@router.put("/{role_id}", dependencies=[Depends(role_required("super_admin"))])
def update_role(
    role_id: str,
    payload: RoleUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update a role with database transaction and audit logging.
    Requirements: 4.4, 4.7, 5.1, 5.2, 5.4
    """
    # Validate database connection (Requirement 5.2)
    try:
        db.execute("SELECT 1")
    except OperationalError as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection unavailable"
        )
    
    try:
        # Use database transaction with locking (Requirements 5.1, 5.4)
        role = db.query(Role).filter(Role.id == role_id).with_for_update().first()
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        
        # Track changes for audit log
        changes = {}
        
        if payload.name is not None and payload.name != role.name:
            changes["name"] = {"old": role.name, "new": payload.name}
            role.name = payload.name
        if payload.description is not None and payload.description != role.description:
            changes["description"] = {"old": role.description, "new": payload.description}
            role.description = payload.description
        if payload.tenant_scoped is not None and payload.tenant_scoped != role.tenant_scoped:
            changes["tenant_scoped"] = {"old": role.tenant_scoped, "new": payload.tenant_scoped}
            role.tenant_scoped = payload.tenant_scoped
        
        # Update permissions if provided (Requirement 4.4)
        if payload.permission_ids is not None:
            old_permissions = [str(p.id) for p in role.permissions]
            permissions = db.query(Permission).filter(
                Permission.id.in_(payload.permission_ids)
            ).all()
            role.permissions = permissions
            new_permissions = [str(p.id) for p in permissions]
            if old_permissions != new_permissions:
                changes["permissions"] = {"old": old_permissions, "new": new_permissions}
        
        db.flush()
        
        # Audit logging (Requirement 4.7)
        if changes:
            try:
                log_event(
                    db=db,
                    user_id=current_user.id,
                    action_type="role_updated",
                    resource_type="role",
                    resource_id=role.id,
                    change_summary=f"Role updated: {role.name}",
                    change_details={
                        "role_name": role.name,
                        "role_slug": role.slug,
                        "changes": changes
                    },
                    category="authorization"
                )
            except Exception as e:
                logger.warning(f"Audit logging failed: {str(e)}")
        
        db.commit()
        db.refresh(role)
        
        # TODO: Clear RBAC cache (Requirement 4.5)
        
        return {
            "id": str(role.id),
            "name": role.name,
            "slug": role.slug,
            "description": role.description,
            "tenant_scoped": role.tenant_scoped,
            "permissions": [
                {
                    "id": str(p.id),
                    "name": p.name,
                    "code": p.code,
                    "description": p.description,
                }
                for p in role.permissions
            ],
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Role update failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update role: {str(e)}"
        )


@router.delete("/{role_id}", dependencies=[Depends(role_required("super_admin"))])
def delete_role(
    role_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a role with cascade deletion and audit logging.
    Requirements: 4.3, 4.7, 5.1, 5.2
    """
    # Validate database connection (Requirement 5.2)
    try:
        db.execute("SELECT 1")
    except OperationalError as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection unavailable"
        )
    
    try:
        # Use database transaction (Requirement 5.1)
        role = db.query(Role).filter(Role.id == role_id).first()
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        
        # Get user count before deletion for audit log
        user_count = db.query(user_roles).filter(user_roles.c.role_id == role.id).count()
        
        # Store role info for audit log
        role_info = {
            "role_name": role.name,
            "role_slug": role.slug,
            "user_count": user_count,
            "permissions": [str(p.id) for p in role.permissions]
        }
        
        # Cascade deletion of user role assignments (Requirement 4.3)
        db.execute(user_roles.delete().where(user_roles.c.role_id == role.id))
        
        # Delete the role
        db.delete(role)
        db.flush()
        
        # Audit logging (Requirement 4.7)
        try:
            log_event(
                db=db,
                user_id=current_user.id,
                action_type="role_deleted",
                resource_type="role",
                resource_id=uuid.UUID(role_id),
                change_summary=f"Role deleted: {role_info['role_name']}",
                change_details=role_info,
                category="authorization"
            )
        except Exception as e:
            logger.warning(f"Audit logging failed: {str(e)}")
        
        db.commit()
        
        # TODO: Clear RBAC cache (Requirement 4.5)
        
        return {"message": "Role deleted successfully", "users_affected": user_count}
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Role deletion failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete role: {str(e)}"
        )


@router.get("/permissions", dependencies=[Depends(role_required("super_admin"))])
def get_all_permissions(db: Session = Depends(get_db)):
    """Get all permissions"""
    permissions = db.query(Permission).all()
    return [
        {
            "id": str(p.id),
            "name": p.name,
            "code": p.code,
            "description": p.description,
        }
        for p in permissions
    ]
