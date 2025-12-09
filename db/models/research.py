from sqlalchemy import Column, String, ForeignKey, Date, Integer, Boolean, Text, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP
from sqlalchemy import text
import uuid

from db.base import Base


class ResearchRequest(Base):
    __tablename__ = "research_requests"
    __table_args__ = {"schema": "registry"}
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Request Identifier
    request_id = Column(String, unique=True, nullable=False)  # REQ-UMMC-YYYYMMDD-HHMMSS-####
    
    # Researcher Information
    researcher_name = Column(String, nullable=False)
    researcher_email = Column(String, nullable=False)
    researcher_affiliation = Column(String)  # Institution/Organization
    purpose_of_study = Column(Text, nullable=False)
    
    # Request Filters (stored as JSONB for flexibility)
    filters = Column(JSONB, nullable=False)  # All filter criteria
    
    # Approval Workflow
    status = Column(String, nullable=False, default='PENDING')  # PENDING, APPROVED, REJECTED, EXPIRED
    rejection_reason = Column(Text)  # If rejected
    approved_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    approved_at = Column(TIMESTAMP(timezone=True))
    
    # Data Extraction
    record_count = Column(Integer)  # Number of records matching filters
    extraction_date = Column(TIMESTAMP(timezone=True))
    download_token = Column(String, unique=True)  # UMMC-TOKEN-<32-char>
    token_expires_at = Column(TIMESTAMP(timezone=True))
    
    # Metadata
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    updated_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'), onupdate=text('now()'))
    created_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    
    # Tenant/Organization (for multi-tenant support)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("core.tenants.id", ondelete="SET NULL"), nullable=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("core.organizations.id", ondelete="SET NULL"), nullable=True)


class ResearchRequestFilter(Base):
    """Structured filter definition for research requests"""
    __tablename__ = "research_request_filters"
    __table_args__ = {"schema": "registry"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(UUID(as_uuid=True), ForeignKey("registry.research_requests.id", ondelete="CASCADE"), nullable=False)
    
    # Filter Fields
    icd11_main_code = Column(String)  # Cancer type filter
    icd11_description = Column(String)
    diagnosis_year_from = Column(Integer)
    diagnosis_year_to = Column(Integer)
    age_from = Column(Integer)
    age_to = Column(Integer)
    gender = Column(String)  # Male, Female, Other, Unknown
    t_category = Column(String)
    n_category = Column(String)
    m_category = Column(String)
    icd11_morphology_code = Column(String)
    icd11_topography_code = Column(String)
    surgery_done = Column(Boolean)
    chemotherapy_done = Column(Boolean)
    radiotherapy_done = Column(Boolean)
    hormonal_therapy = Column(Boolean)
    immunotherapy = Column(Boolean)
    recurrence = Column(Boolean)
    metastasis = Column(Boolean)
    vital_status = Column(String)  # Alive, Dead, Unknown
    treatment_intent = Column(String)
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))

