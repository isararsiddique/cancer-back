from sqlalchemy import Column, String, Boolean, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy import text
import uuid

from db.base import Base


class OnboardingRequest(Base):
    """A 'join the platform' request from a hospital or an expert clinician."""
    __tablename__ = "onboarding_requests"
    __table_args__ = {"schema": "registry"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_type = Column(String, nullable=False)          # hospital | expert_doctor
    contact_name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    phone = Column(String)
    organization = Column(String)                          # hospital / institution name
    country = Column(String)
    specialty = Column(String)                             # expert doctor specialty
    has_data_entry_system = Column(Boolean)                # hospital: already has EMR/data entry?
    estimated_volume = Column(String)                      # approx records/year
    message = Column(Text)
    status = Column(String, nullable=False, default="NEW")  # NEW, CONTACTED, APPROVED, REJECTED
    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))


class HospitalApiKey(Base):
    """A scoped API key that lets an onboarded hospital push data to its own organization."""
    __tablename__ = "hospital_api_keys"
    __table_args__ = {"schema": "registry"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("core.organizations.id", ondelete="CASCADE"), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("core.tenants.id", ondelete="SET NULL"), nullable=True)
    label = Column(String)
    key_prefix = Column(String, nullable=False)   # first chars, shown for identification
    key_hash = Column(String, nullable=False)      # sha256 of the full key
    is_active = Column(Boolean, nullable=False, default=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))
    last_used_at = Column(TIMESTAMP(timezone=True))
