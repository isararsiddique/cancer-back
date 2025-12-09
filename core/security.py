import jwt
from datetime import datetime, timedelta, timezone
import bcrypt
from typing import Tuple, Optional
import secrets

from core.config import settings


def create_access_token(sub: str, claims: dict, expires_seconds: int = 86400) -> str:
    """
    Create JWT access token with 24 hour expiry (86400 seconds).
    Used for API authentication.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=expires_seconds)).timestamp()),
        "type": "access",  # Token type identifier
        **claims,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def create_refresh_token(sub: str, jti: Optional[str] = None) -> Tuple[str, str]:
    """
    Create JWT refresh token with 30 days expiry.
    Returns (refresh_token, jti) tuple.
    jti (JWT ID) is used for token revocation and session management.
    """
    now = datetime.now(timezone.utc)
    jti = jti or secrets.token_urlsafe(32)  # Generate unique token ID
    
    payload = {
        "sub": sub,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=30)).timestamp()),  # 30 days expiry
        "type": "refresh",  # Token type identifier
        "jti": jti,  # JWT ID for session management
    }
    refresh_token = jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    return refresh_token, jti


def decode_token(token: str, token_type: Optional[str] = None) -> dict:
    """
    Decode and validate JWT token.
    Optionally validate token type (access or refresh).
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
        )
        
        # Validate token type if specified
        if token_type and payload.get("type") != token_type:
            raise jwt.InvalidTokenError(f"Invalid token type. Expected {token_type}")
        
        return payload
    except jwt.ExpiredSignatureError:
        raise jwt.ExpiredSignatureError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise jwt.InvalidTokenError(f"Invalid token: {str(e)}")


def verify_token(token: str, token_type: Optional[str] = None) -> bool:
    """
    Verify if token is valid without raising exceptions.
    Returns True if valid, False otherwise.
    """
    try:
        decode_token(token, token_type)
        return True
    except Exception:
        return False


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a bcrypt hash"""
    try:
        # Convert to bytes
        password_bytes = plain_password.encode('utf-8')
        hash_bytes = hashed_password.encode('utf-8')
        # Verify using bcrypt
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except Exception as e:
        print(f"Password verification error: {e}")
        return False


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt"""
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')
