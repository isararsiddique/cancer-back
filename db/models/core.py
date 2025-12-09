from sqlalchemy import Column, String, ForeignKey, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from db.base import Base


class Tenant(Base):
    __tablename__ = "tenants"
    __table_args__ = {"schema": "core"}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, unique=True)
    meta = Column("metadata", JSON, default=dict)


class Organization(Base):
    __tablename__ = "organizations"
    __table_args__ = {"schema": "core"}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("core.tenants.id", ondelete="SET NULL"), nullable=True)
    name = Column(String, nullable=False)
    code = Column(String, nullable=True)
    meta = Column("metadata", JSON, default=dict)
