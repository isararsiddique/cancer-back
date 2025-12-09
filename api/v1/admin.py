from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, func
from pydantic import BaseModel, EmailStr
from datetime import datetime
import logging

from core.deps import get_db, get_current_user, role_required
from core.security import get_password_hash
from core.audit import log_event
from db.models.users import User
from db.models.rbac import Role, user_roles
from db.models.core import Organization, Tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class UserCreateRequest(BaseModel):
    """Create new user - super admin only"""
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role_ids: List[str]  # List of role IDs to assign
    tenant_id: Optional[str] = None
    organization_id: Optional[str] = None
    is_active: bool = True


class UserUpdateRequest(BaseModel):
    """Update existing user - super admin only"""
    email: Optional[EmailStr] = None
    password: Optional[str] = None  # If provided, will update password
    full_name: Optional[str] = None
    role_ids: Optional[List[str]] = None
    tenant_id: Optional[str] = None
    organization_id: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """User response with all details including password hash"""
    id: str
    email: str
    full_name: Optional[str]
    is_active: bool
    tenant_id: Optional[str]
    organization_id: Optional[str]
    created_at: Optional[datetime]
    roles: List[dict]
    # Note: We don't expose hashed_password in API for security
    
    class Config:
        from_attributes = True


class AssignRoleRequest(BaseModel):
    """Assign role to user"""
    user_id: str
    role_id: str


# ============================================================================
# USER MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/users", dependencies=[Depends(role_required("super_admin"))])
def list_all_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None, description="Search by email or name"),
    role_slug: Optional[str] = Query(None, description="Filter by role slug"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all users with their roles and credentials.
    Super admin can see all user details including login credentials.
    
    Features:
    - Search by email or name
    - Filter by role
    - Filter by active status
    - Pagination support
    """
    try:
        # Base query with eager loading
        query = db.query(User).options(
            joinedload(User.roles),
            joinedload(User.tenant),
            joinedload(User.organization)
        )
        
        # Search filter
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    User.email.ilike(search_term),
                    User.full_name.ilike(search_term)
                )
            )
        
        # Role filter
        if role_slug:
            query = query.join(User.roles).filter(Role.slug == role_slug)
        
        # Active status filter
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        users = query.order_by(User.created_at.desc()).offset(skip).limit(limit).all()
        
        # Format response
        result = []
        for user in users:
            user_data = {
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "tenant_id": str(user.tenant_id) if user.tenant_id else None,
                "tenant_name": user.tenant.name if user.tenant else None,
                "organization_id": str(user.organization_id) if user.organization_id else None,
                "organization_name": user.organization.name if user.organization else None,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "roles": [
                    {
                        "id": str(role.id),
                        "name": role.name,
                        "slug": role.slug,
                        "description": role.description
                    }
                    for role in user.roles
                ],
                # Note: We don't expose the actual password hash for security
                # Super admin can reset passwords using the update endpoint
                "has_password": bool(user.hashed_password)
            }
            result.append(user_data)
        
        return {
            "users": result,
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Failed to list users: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list users: {str(e)}"
        )


@router.get("/users/{user_id}", dependencies=[Depends(role_required("super_admin"))])
def get_user_details(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a specific user.
    Super admin can see all user details.
    """
    try:
        user = db.query(User).options(
            joinedload(User.roles),
            joinedload(User.tenant),
            joinedload(User.organization)
        ).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "tenant_id": str(user.tenant_id) if user.tenant_id else None,
            "tenant_name": user.tenant.name if user.tenant else None,
            "organization_id": str(user.organization_id) if user.organization_id else None,
            "organization_name": user.organization.name if user.organization else None,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "roles": [
                {
                    "id": str(role.id),
                    "name": role.name,
                    "slug": role.slug,
                    "description": role.description,
                    "permissions": [
                        {
                            "id": str(p.id),
                            "name": p.name,
                            "code": p.code
                        }
                        for p in role.permissions
                    ]
                }
                for role in user.roles
            ],
            "has_password": bool(user.hashed_password)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user details: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user details: {str(e)}"
        )


@router.post("/users", dependencies=[Depends(role_required("super_admin"))])
def create_user(
    payload: UserCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new user with specified roles.
    Super admin can create users for any role except super_admin.
    
    Supports creating:
    - Hospital staff (doctors, nurses, admins)
    - UMMC admins
    - Researchers
    """
    try:
        # Check if email already exists
        existing = db.query(User).filter(User.email == payload.email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )
        
        # Validate roles exist and prevent super_admin assignment
        roles = db.query(Role).filter(Role.id.in_(payload.role_ids)).all()
        if len(roles) != len(payload.role_ids):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="One or more role IDs are invalid"
            )
        
        # Prevent creating super_admin users
        for role in roles:
            if role.slug == "super_admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot assign super_admin role to new users"
                )
        
        # Create user
        user = User(
            email=payload.email,
            full_name=payload.full_name,
            hashed_password=get_password_hash(payload.password),
            tenant_id=payload.tenant_id,
            organization_id=payload.organization_id,
            is_active=payload.is_active
        )
        
        # Assign roles
        user.roles = roles
        
        db.add(user)
        db.flush()
        
        # Audit log
        try:
            log_event(
                db=db,
                user_id=current_user.id,
                action_type="user_created",
                resource_type="user",
                resource_id=user.id,
                change_summary=f"User created: {user.email}",
                change_details={
                    "email": user.email,
                    "roles": [r.slug for r in roles],
                    "created_by": current_user.email
                },
                category="user_management"
            )
        except Exception as e:
            logger.warning(f"Audit logging failed: {str(e)}")
        
        db.commit()
        db.refresh(user)
        
        return {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "roles": [
                {
                    "id": str(role.id),
                    "name": role.name,
                    "slug": role.slug
                }
                for role in user.roles
            ],
            "message": "User created successfully"
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create user: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )


@router.put("/users/{user_id}", dependencies=[Depends(role_required("super_admin"))])
def update_user(
    user_id: str,
    payload: UserUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update user details including password and roles.
    Super admin can update any user except other super_admins.
    """
    try:
        user = db.query(User).options(joinedload(User.roles)).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Prevent modifying super_admin users (except self)
        if any(r.slug == "super_admin" for r in user.roles) and str(user.id) != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot modify other super admin users"
            )
        
        changes = {}
        
        # Update email
        if payload.email and payload.email != user.email:
            # Check if new email already exists
            existing = db.query(User).filter(User.email == payload.email, User.id != user_id).first()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already exists"
                )
            changes["email"] = {"old": user.email, "new": payload.email}
            user.email = payload.email
        
        # Update password
        if payload.password:
            user.hashed_password = get_password_hash(payload.password)
            changes["password"] = "updated"
        
        # Update full name
        if payload.full_name is not None:
            changes["full_name"] = {"old": user.full_name, "new": payload.full_name}
            user.full_name = payload.full_name
        
        # Update tenant
        if payload.tenant_id is not None:
            changes["tenant_id"] = {"old": str(user.tenant_id) if user.tenant_id else None, "new": payload.tenant_id}
            user.tenant_id = payload.tenant_id
        
        # Update organization
        if payload.organization_id is not None:
            changes["organization_id"] = {"old": str(user.organization_id) if user.organization_id else None, "new": payload.organization_id}
            user.organization_id = payload.organization_id
        
        # Update active status
        if payload.is_active is not None:
            changes["is_active"] = {"old": user.is_active, "new": payload.is_active}
            user.is_active = payload.is_active
        
        # Update roles
        if payload.role_ids is not None:
            roles = db.query(Role).filter(Role.id.in_(payload.role_ids)).all()
            if len(roles) != len(payload.role_ids):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="One or more role IDs are invalid"
                )
            
            # Prevent assigning super_admin role
            for role in roles:
                if role.slug == "super_admin":
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Cannot assign super_admin role"
                    )
            
            old_roles = [r.slug for r in user.roles]
            user.roles = roles
            new_roles = [r.slug for r in roles]
            changes["roles"] = {"old": old_roles, "new": new_roles}
        
        db.flush()
        
        # Audit log
        if changes:
            try:
                log_event(
                    db=db,
                    user_id=current_user.id,
                    action_type="user_updated",
                    resource_type="user",
                    resource_id=user.id,
                    change_summary=f"User updated: {user.email}",
                    change_details={
                        "email": user.email,
                        "changes": changes,
                        "updated_by": current_user.email
                    },
                    category="user_management"
                )
            except Exception as e:
                logger.warning(f"Audit logging failed: {str(e)}")
        
        db.commit()
        db.refresh(user)
        
        return {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "roles": [
                {
                    "id": str(role.id),
                    "name": role.name,
                    "slug": role.slug
                }
                for role in user.roles
            ],
            "message": "User updated successfully"
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update user: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}"
        )


@router.delete("/users/{user_id}", dependencies=[Depends(role_required("super_admin"))])
def delete_user(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a user (soft delete by setting is_active=False).
    Super admin cannot delete other super_admins or themselves.
    """
    try:
        user = db.query(User).options(joinedload(User.roles)).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Prevent deleting super_admin users
        if any(r.slug == "super_admin" for r in user.roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot delete super admin users"
            )
        
        # Prevent self-deletion
        if str(user.id) == str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot delete your own account"
            )
        
        # Soft delete
        user.is_active = False
        
        db.flush()
        
        # Audit log
        try:
            log_event(
                db=db,
                user_id=current_user.id,
                action_type="user_deleted",
                resource_type="user",
                resource_id=user.id,
                change_summary=f"User deleted: {user.email}",
                change_details={
                    "email": user.email,
                    "roles": [r.slug for r in user.roles],
                    "deleted_by": current_user.email
                },
                category="user_management"
            )
        except Exception as e:
            logger.warning(f"Audit logging failed: {str(e)}")
        
        db.commit()
        
        return {
            "message": "User deleted successfully",
            "user_id": str(user.id),
            "email": user.email
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete user: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}"
        )


# ============================================================================
# ROLE ASSIGNMENT ENDPOINTS
# ============================================================================

@router.post("/users/{user_id}/roles/{role_id}", dependencies=[Depends(role_required("super_admin"))])
def assign_role_to_user(
    user_id: str,
    role_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Assign a role to a user.
    Super admin can assign any role except super_admin.
    """
    try:
        user = db.query(User).options(joinedload(User.roles)).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        role = db.query(Role).filter(Role.id == role_id).first()
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        
        # Prevent assigning super_admin role
        if role.slug == "super_admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot assign super_admin role"
            )
        
        # Check if user already has this role
        if role in user.roles:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already has this role"
            )
        
        user.roles.append(role)
        db.flush()
        
        # Audit log
        try:
            log_event(
                db=db,
                user_id=current_user.id,
                action_type="role_assigned",
                resource_type="user",
                resource_id=user.id,
                change_summary=f"Role '{role.name}' assigned to {user.email}",
                change_details={
                    "user_email": user.email,
                    "role_name": role.name,
                    "role_slug": role.slug,
                    "assigned_by": current_user.email
                },
                category="user_management"
            )
        except Exception as e:
            logger.warning(f"Audit logging failed: {str(e)}")
        
        db.commit()
        
        return {
            "message": "Role assigned successfully",
            "user_id": str(user.id),
            "role_id": str(role.id),
            "role_name": role.name
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to assign role: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assign role: {str(e)}"
        )


@router.delete("/users/{user_id}/roles/{role_id}", dependencies=[Depends(role_required("super_admin"))])
def remove_role_from_user(
    user_id: str,
    role_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Remove a role from a user.
    Super admin cannot remove super_admin role.
    """
    try:
        user = db.query(User).options(joinedload(User.roles)).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        role = db.query(Role).filter(Role.id == role_id).first()
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        
        # Prevent removing super_admin role
        if role.slug == "super_admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot remove super_admin role"
            )
        
        # Check if user has this role
        if role not in user.roles:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User does not have this role"
            )
        
        user.roles.remove(role)
        db.flush()
        
        # Audit log
        try:
            log_event(
                db=db,
                user_id=current_user.id,
                action_type="role_removed",
                resource_type="user",
                resource_id=user.id,
                change_summary=f"Role '{role.name}' removed from {user.email}",
                change_details={
                    "user_email": user.email,
                    "role_name": role.name,
                    "role_slug": role.slug,
                    "removed_by": current_user.email
                },
                category="user_management"
            )
        except Exception as e:
            logger.warning(f"Audit logging failed: {str(e)}")
        
        db.commit()
        
        return {
            "message": "Role removed successfully",
            "user_id": str(user.id),
            "role_id": str(role.id),
            "role_name": role.name
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to remove role: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove role: {str(e)}"
        )

