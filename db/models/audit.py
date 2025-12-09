"""
Enterprise-Grade Immutable Audit Log Models
"""
from sqlalchemy import Column, String, ForeignKey, Boolean, Text, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP, ARRAY, INET
from sqlalchemy.orm import relationship
import uuid

from db.base import Base


class AuditLog(Base):
    """
    Enterprise-grade immutable audit log entry.
    
    Tracks all critical operations with full context for compliance,
    security, and debugging purposes.
    """
    __tablename__ = "logs"
    __table_args__ = (
        Index("idx_logs_timestamp", "timestamp"),
        Index("idx_logs_user_id", "user_id"),
        Index("idx_logs_resource", "resource_type", "resource_id"),
        Index("idx_logs_action", "action_type", "timestamp"),
        {"schema": "public"}
    )
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Timestamp (immutable)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, server_default="now()")
    
    # User Context
    user_id = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id", ondelete="SET NULL"), nullable=True)
    user_email = Column(String, nullable=True)
    user_name = Column(String, nullable=True)
    user_roles = Column(ARRAY(String), nullable=True)
    
    # Tenant/Organization Context
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("core.tenants.id", ondelete="SET NULL"), nullable=True)
    tenant_name = Column(String, nullable=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("core.organizations.id", ondelete="SET NULL"), nullable=True)
    organization_name = Column(String, nullable=True)
    
    # Action Details
    action_type = Column(String, nullable=False)  # login, create, update, delete, etc.
    resource_type = Column(String, nullable=False)  # user, patient, role, etc.
    resource_id = Column(UUID(as_uuid=True), nullable=True)
    resource_identifier = Column(String, nullable=True)  # Human-readable identifier
    
    # Change Details
    change_summary = Column(Text, nullable=False)  # Human-readable summary
    change_details = Column(JSONB, nullable=True)  # Detailed change information
    old_values = Column(JSONB, nullable=True)  # Previous values (for updates)
    new_values = Column(JSONB, nullable=True)  # New values (for creates/updates)
    
    # Request Context
    ip_address = Column(INET, nullable=True)
    user_agent = Column(Text, nullable=True)
    request_id = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    
    # Status & Outcome
    status = Column(String, nullable=False, default="success")  # success, failure, error
    error_message = Column(Text, nullable=True)
    error_code = Column(String, nullable=True)
    
    # Metadata
    severity = Column(String, default="info")  # info, warning, error, critical
    category = Column(String, nullable=True)  # authentication, authorization, data_access, etc.
    tags = Column(ARRAY(String), nullable=True)
    
    # Compliance & Retention
    retention_until = Column(TIMESTAMP(timezone=True), nullable=True)
    compliance_flags = Column(ARRAY(String), nullable=True)  # GDPR, HIPAA, SOX, etc.
    is_sensitive = Column(Boolean, default=False)
    
    # Immutability Protection
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default="now()")
    checksum = Column(String, nullable=True)  # Hash for integrity verification

