from sqlalchemy import Column, String, Boolean, ForeignKey, Table, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from db.base import Base


# association tables first
role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column("role_id", UUID(as_uuid=True), ForeignKey("rbac.roles.id", ondelete="CASCADE")),
    Column("permission_id", UUID(as_uuid=True), ForeignKey("rbac.permissions.id", ondelete="CASCADE")),
    schema="rbac",
)

# Clean user_roles - no organization_id or tenant_id
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", UUID(as_uuid=True), ForeignKey("rbac.users.id", ondelete="CASCADE")),
    Column("role_id", UUID(as_uuid=True), ForeignKey("rbac.roles.id", ondelete="CASCADE")),
    schema="rbac",
)


class Module(Base):
    """Modules represent features/app sections (users, organizations, patients, etc.)"""
    __tablename__ = "modules"
    __table_args__ = {"schema": "rbac"}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, unique=True)
    description = Column(String, nullable=True)
    permissions = relationship("Permission", back_populates="module")


class Role(Base):
    __tablename__ = "roles"
    __table_args__ = {"schema": "rbac"}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("core.tenants.id", ondelete="CASCADE"), nullable=True)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False)
    description = Column(String, nullable=True)
    tenant_scoped = Column(Boolean, default=False)  # Kept for backward compatibility
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")


class Permission(Base):
    """Enterprise-grade permissions: Module + Action (CRUD)"""
    __tablename__ = "permissions"
    __table_args__ = (
        CheckConstraint("action IN ('create', 'read', 'update', 'delete')", name="permission_action_check"),
        {"schema": "rbac"}
    )
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    module_id = Column(UUID(as_uuid=True), ForeignKey("rbac.modules.id", ondelete="CASCADE"), nullable=False)
    action = Column(String, nullable=False)  # create, read, update, delete
    code = Column(String, nullable=False)  # Generated: module_name:action (e.g., "patients:read")
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    module = relationship("Module", back_populates="permissions")
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")
