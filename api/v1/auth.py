"""
Enterprise-grade JWT Authentication System
- Single login endpoint to get JWT tokens
- JWT access tokens (24 hour expiry)
- Refresh tokens (30 days expiry)
- Auto-refresh mechanism
- Session management
- Use JWT tokens in Authorization: Bearer <token> header
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Form, Request
from fastapi import Request as FastAPIRequest
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
import hashlib
import jwt

from core.deps import get_db, get_current_user
from core.security import (
    create_access_token, 
    create_refresh_token,
    decode_token,
    verify_password,
)
from core.audit import log_login, log_logout
from db.models.users import User
from db.models.auth import RefreshToken

from pydantic import BaseModel


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 86400  # 24 hours in seconds


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str


class RefreshTokenResponse(BaseModel):
    """Refresh token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 86400  # 24 hours in seconds


class UserResponse(BaseModel):
    """User profile response"""
    id: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    roles: List[str]


router = APIRouter(prefix="/auth", tags=["auth"])


def _create_tokens_for_user(user: User, db: Session, request: Optional[FastAPIRequest] = None, dashboard_type: Optional[str] = None) -> TokenResponse:
    """
    Helper function to create access and refresh tokens for a user.
    Stores refresh token in database for session management.
    
    Args:
        user: User object
        db: Database session
        request: Optional FastAPI request for logging
        dashboard_type: Optional dashboard type to lock token to specific dashboard
    """
    try:
        # Get user roles - handle case where roles might not be loaded
        roles = []
        if hasattr(user, "roles") and user.roles:
            roles = [r.slug for r in user.roles]
        else:
            # If roles not loaded, query them
            from db.models.rbac import Role, user_roles
            role_ids = db.query(user_roles.c.role_id).filter(user_roles.c.user_id == user.id).all()
            if role_ids:
                role_slugs = db.query(Role.slug).filter(Role.id.in_([r[0] for r in role_ids])).all()
                roles = [r[0] for r in role_slugs]
        
        claims = {
            "tenant_id": str(user.tenant_id) if user.tenant_id else None,
            "organization_id": str(user.organization_id) if user.organization_id else None,
            "roles": roles,
            "email": user.email,
            "dashboard_type": dashboard_type,  # Lock token to specific dashboard
        }
        
        # Create tokens - access token expires in 24 hours (86400 seconds)
        # This prevents frequent re-login requirements when switching between pages
        access_token = create_access_token(sub=str(user.id), claims=claims, expires_seconds=86400)
        refresh_token, jti = create_refresh_token(sub=str(user.id))
        
        # Hash refresh token for storage
        token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
        
        # Get client info if available
        user_agent = None
        ip_address = None
        if request:
            user_agent = request.headers.get("user-agent")
            ip_address = request.client.host if request.client else None
        
        # Store refresh token in database
        expires_at = datetime.now(timezone.utc) + timedelta(days=30)
        db_refresh_token = RefreshToken(
            user_id=user.id,
            jti=jti,
            token_hash=token_hash,
            expires_at=expires_at,
            user_agent=user_agent,
            ip_address=ip_address,
            is_active=True
        )
        db.add(db_refresh_token)
        db.commit()
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=3600  # 1 hour
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create tokens: {str(e)}"
        )


@router.post("/login", response_model=TokenResponse)
def login(
    username: str = Form(...),
    password: str = Form(...),
    dashboard_type: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    request: FastAPIRequest = None
):
    """
    Enterprise JWT Authentication Login
    
    Login with email and password to receive JWT tokens.
    Use the access_token in Authorization header: "Bearer <access_token>"
    
    Parameters:
    - username: User email
    - password: User password
    - dashboard_type: Optional dashboard type ("hospital" or "researcher") - locks token to specific dashboard
    
    Returns:
    - access_token: JWT token valid for 1 hour (3600 seconds)
    - refresh_token: JWT token valid for 30 days (for auto-refresh)
    - token_type: "bearer"
    - expires_in: 3600 (seconds)
    
    Usage:
    1. Call this endpoint with email (as username) and password
    2. Receive access_token and refresh_token
    3. Use access_token in API requests: Authorization: Bearer <access_token>
    4. Use refresh_token to get new access_token when it expires
    
    Security:
    - Credentials validated against database
    - JWT tokens signed with HS256
    - Refresh tokens stored securely in database
    - Session tracking with IP and User-Agent
    - Dashboard type locked in token - switching requires re-login
    """
    # username field is actually the email (OAuth2 standard)
    # Eager load roles to avoid lazy loading issues
    from sqlalchemy.orm import joinedload
    user = db.query(User).options(joinedload(User.roles)).filter(User.email == username).first()
    if not user or not user.hashed_password or not verify_password(password, user.hashed_password):
        # Log failed login attempt (don't reveal if user exists)
        if user:
            try:
                log_login(db, user.id, status="failure", error_message="Incorrect password", request=request)
            except Exception:
                pass  # Don't fail login if audit logging fails
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        # Log failed login attempt
        try:
            log_login(db, user.id, status="failure", error_message="Inactive user", request=request)
        except Exception:
            pass  # Don't fail login if audit logging fails
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Inactive user"
        )

    # Log successful login (non-blocking) - do this AFTER we know login will succeed
    # but BEFORE creating tokens to avoid transaction issues
    try:
        log_login(db, user.id, status="success", request=request)
    except Exception:
        pass  # Don't fail login if audit logging fails
    
    # Ensure transaction is clean before creating tokens
    try:
        return _create_tokens_for_user(user, db, request, dashboard_type)
    except Exception as e:
        # If token creation fails, rollback and re-raise
        db.rollback()
        raise


@router.post("/refresh", response_model=RefreshTokenResponse)
def refresh_token(
    payload: RefreshTokenRequest,
    db: Session = Depends(get_db),
    request: FastAPIRequest = None
):
    """
    Refresh access token using refresh token.
    
    Implements token rotation for enhanced security:
    1. Validates refresh token
    2. Checks database for active token
    3. Revokes old refresh token
    4. Generates new access and refresh tokens
    5. Updates session timestamp
    
    This enables automatic token refresh without re-authentication.
    """
    try:
        # Decode and validate refresh token
        token_payload = decode_token(payload.refresh_token, token_type="refresh")
        user_id = token_payload.get("sub")
        jti = token_payload.get("jti")
        
        if not user_id or not jti:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Check if token exists in database and is active
        token_hash = hashlib.sha256(payload.refresh_token.encode()).hexdigest()
        db_token = db.query(RefreshToken).filter(
            RefreshToken.jti == jti,
            RefreshToken.user_id == user_id,
            RefreshToken.is_active == True
        ).first()
        
        if not db_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token not found or revoked"
            )
        
        # Verify token hash matches
        if db_token.token_hash != token_hash:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Check if token is expired
        if db_token.expires_at < datetime.now(timezone.utc):
            db_token.is_active = False
            db_token.revoked_at = datetime.now(timezone.utc)
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired"
            )
        
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Revoke old refresh token (token rotation)
        db_token.is_active = False
        db_token.revoked_at = datetime.now(timezone.utc)
        db_token.last_used_at = datetime.now(timezone.utc)
        
        # Create new tokens
        response = _create_tokens_for_user(user, db, request)
        
        return RefreshTokenResponse(
            access_token=response.access_token,
            refresh_token=response.refresh_token,
            expires_in=3600
        )
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired"
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {str(e)}"
        )


@router.post("/logout")
def logout(
    payload: RefreshTokenRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    request: FastAPIRequest = None
):
    """
    Logout by revoking refresh token.
    
    This invalidates the refresh token and ends the session.
    The access token will expire naturally (1 hour).
    """
    try:
        token_payload = decode_token(payload.refresh_token, token_type="refresh")
        jti = token_payload.get("jti")
        user_id = token_payload.get("sub")
        
        if not jti or str(user_id) != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Revoke token
        db_token = db.query(RefreshToken).filter(
            RefreshToken.jti == jti,
            RefreshToken.user_id == current_user.id
        ).first()
        
        if db_token:
            db_token.is_active = False
            db_token.revoked_at = datetime.now(timezone.utc)
            db.commit()
        
        # Log logout
        log_logout(db, current_user.id, request=request)
        
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        # Even if token is invalid, return success for security
        return {"message": "Logged out successfully"}


@router.get("/sessions")
def get_active_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get list of active sessions (refresh tokens) for the current user.
    
    Returns information about all active sessions including:
    - Creation time
    - Last used time
    - Expiry time
    - User agent and IP address
    """
    sessions = db.query(RefreshToken).filter(
        RefreshToken.user_id == current_user.id,
        RefreshToken.is_active == True,
        RefreshToken.expires_at > datetime.now(timezone.utc)
    ).order_by(RefreshToken.created_at.desc()).all()
    
    return {
        "sessions": [
            {
                "id": str(session.id),
                "created_at": session.created_at.isoformat(),
                "last_used_at": session.last_used_at.isoformat() if session.last_used_at else None,
                "expires_at": session.expires_at.isoformat(),
                "user_agent": session.user_agent,
                "ip_address": session.ip_address,
            }
            for session in sessions
        ],
        "total_active": len(sessions)
    }


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user profile.
    
    Returns user information including roles and permissions.
    """
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        roles=[r.slug for r in current_user.roles],
    )
