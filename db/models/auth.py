"""
Authentication models for JWT refresh tokens and session management.
"""
from sqlalchemy import Column, String, Boolean, ForeignKey, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from db.base import Base


class RefreshToken(Base):
    """
    Stores refresh tokens for JWT session management.
    Allows token revocation and session tracking.
    """
    __tablename__ = "refresh_tokens"
    __table_args__ = (
        Index("idx_refresh_tokens_user_active", "user_id", "is_active"),
        Index("idx_refresh_tokens_expires", "expires_at"),
        {"schema": "rbac"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id", ondelete="CASCADE"), nullable=False, index=True)
    jti = Column(String, unique=True, nullable=False, index=True)  # JWT ID from refresh token
    token_hash = Column(String, nullable=False)  # Hashed refresh token for verification
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    user_agent = Column(String, nullable=True)  # Browser/client info
    ip_address = Column(String, nullable=True)  # Client IP
    
    # Relationship
    user = relationship("User", backref="refresh_tokens")

