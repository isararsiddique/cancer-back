"""
Collaboration module — bidirectional data requests between researchers and hospitals
with protocol, document sharing, messaging, and audit.
"""
from sqlalchemy import Column, String, Boolean, Text, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP, JSONB
from sqlalchemy import text
import uuid

from db.base import Base


class DataCollaboration(Base):
    """A governed data-sharing request — either researcher->hospital or hospital->researcher."""
    __tablename__ = "data_collaborations"
    __table_args__ = {"schema": "registry"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collab_id = Column(String, unique=True, nullable=False)  # COL-NG-YYYYMMDD-XXXX

    # Direction
    initiated_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"), nullable=False)
    initiated_by_role = Column(String, nullable=False)  # researcher | hospital_admin
    target_organization_id = Column(UUID(as_uuid=True), ForeignKey("core.organizations.id"))

    # Protocol
    title = Column(String, nullable=False)
    purpose = Column(Text, nullable=False)
    protocol_version = Column(String, default="v1.0")
    data_requirements = Column(Text)  # what data is needed
    ethical_justification = Column(Text)
    estimated_records = Column(Integer)
    icd11_codes = Column(JSONB)  # list of ICD-11 codes requested
    year_range = Column(JSONB)  # {"from": 2020, "to": 2025}

    # Status flow: DRAFT -> SUBMITTED -> UNDER_REVIEW -> ETHICS_PENDING ->
    #              ETHICS_APPROVED -> DATA_SHARED -> COMPLETED | REJECTED | WITHDRAWN
    status = Column(String, nullable=False, default="DRAFT")
    rejection_reason = Column(Text)

    # Ethics
    ethics_approval_ref = Column(String)
    ethics_approved_at = Column(TIMESTAMP(timezone=True))
    ethics_approved_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))

    # Data sharing
    data_shared_at = Column(TIMESTAMP(timezone=True))
    data_record_count = Column(Integer)

    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))


class CollabDocument(Base):
    """Shared documents (protocols, ethics approval letters, consent forms, etc.)."""
    __tablename__ = "collab_documents"
    __table_args__ = {"schema": "registry"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collaboration_id = Column(UUID(as_uuid=True), ForeignKey("registry.data_collaborations.id", ondelete="CASCADE"), nullable=False)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String)
    file_size = Column(Integer)
    document_category = Column(String)  # protocol | ethics_approval | consent_form | data_dictionary | other
    description = Column(Text)
    uploaded_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))


class CollabMessage(Base):
    """Audited conversation thread on a collaboration."""
    __tablename__ = "collab_messages"
    __table_args__ = {"schema": "registry"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collaboration_id = Column(UUID(as_uuid=True), ForeignKey("registry.data_collaborations.id", ondelete="CASCADE"), nullable=False)
    sender_id = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"), nullable=False)
    sender_name = Column(String)
    sender_role = Column(String)
    message = Column(Text, nullable=False)
    is_system = Column(Boolean, default=False)  # system-generated audit messages
    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))


class CollabAuditEntry(Base):
    """Immutable audit log per collaboration action."""
    __tablename__ = "collab_audit"
    __table_args__ = {"schema": "registry"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collaboration_id = Column(UUID(as_uuid=True), ForeignKey("registry.data_collaborations.id", ondelete="CASCADE"), nullable=False)
    actor_id = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    actor_name = Column(String)
    action = Column(String, nullable=False)  # created, submitted, reviewed, ethics_approved, data_shared, messaged, document_uploaded, rejected, withdrawn
    detail = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))
