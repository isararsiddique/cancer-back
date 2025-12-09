from sqlalchemy import Column, String, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid

from db.base import Base
from db.models.rbac import user_roles


class User(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "rbac"}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("core.tenants.id", ondelete="SET NULL"), nullable=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("core.organizations.id", ondelete="SET NULL"), nullable=True)
    email = Column(String, nullable=False, unique=True)
    full_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_email_verified = Column(Boolean, default=False)
    meta = Column("metadata", JSONB, default=dict)
    roles = relationship("Role", secondary=user_roles)
