from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from db.session import SessionLocal
from core.security import decode_token
from core.config import settings
from db.session import set_session_vars
from db.models.users import User
from db.models.rbac import Role, Permission


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# HTTP Bearer scheme for JWT token authentication
# Users login via /api/v1/auth/login to get JWT tokens, then use them in Authorization header
security = HTTPBearer(scheme_name="HTTPBearer", auto_error=True)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    # Extract token from Bearer credentials
    token = credentials.credentials
    
    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive user")

    # Eagerly load roles and permissions
    user.roles  # access relationship
    # Build roles CSV for RLS
    roles_csv = ",".join([r.slug for r in user.roles])
    is_super_admin = any(r.slug == "super_admin" for r in user.roles)

    set_session_vars(
        db,
        user_id=str(user.id),
        tenant_id=str(user.tenant_id) if user.tenant_id else None,
        organization_id=str(user.organization_id) if user.organization_id else None,
        roles_csv=roles_csv,
        is_super_admin=is_super_admin,
        enc_key=settings.db_enc_key,
    )

    return user

def permission_required(permission_code: str) -> Callable:
    """
    Check if user has required permission.
    Permission code format: "module:action" (e.g., "patients:read", "users:create")
    """
    def _checker(current_user=Depends(get_current_user), db: Session = Depends(get_db)):
        # Super admin bypass
        if any(r.slug == "super_admin" for r in current_user.roles):
            return True
        
        # Get all user permissions from their roles
        user_perms = {p.code for role in current_user.roles for p in role.permissions}
        
        # Check if user has the required permission
        if permission_code not in user_perms:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail=f"Insufficient permissions. Required: {permission_code}"
            )
        return True

    return _checker


def role_required(*role_slugs: str) -> Callable:
    def _checker(current_user=Depends(get_current_user)):
        if any(r.slug == "super_admin" for r in current_user.roles):
            return True
        if not any(r.slug in role_slugs for r in current_user.roles):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return True

    return _checker


def dashboard_required(required_dashboard: str) -> Callable:
    """
    Check if user's token is locked to the required dashboard type.
    This prevents users from switching between hospital and researcher dashboards without re-login.
    
    Args:
        required_dashboard: "hospital" or "researcher"
    """
    def _checker(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        current_user=Depends(get_current_user)
    ):
        token = credentials.credentials
        try:
            payload = decode_token(token)
            token_dashboard = payload.get("dashboard_type")
            
            # If token has no dashboard type (old tokens), require re-login
            if token_dashboard is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Please login again to access this dashboard"
                )
            
            # If token dashboard doesn't match required dashboard, require re-login
            if token_dashboard != required_dashboard:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"This session is for {token_dashboard} dashboard. Please login to access {required_dashboard} dashboard."
                )
            
            return True
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    return _checker
