from sqlalchemy import Column, String, ForeignKey, Date, Integer, Boolean, Text, CheckConstraint, text
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP
import uuid

from db.base import Base


class Patient(Base):
    __tablename__ = "patients"
    __table_args__ = {"schema": "registry"}
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Tenant & Organization
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("core.tenants.id", ondelete="SET NULL"), nullable=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("core.organizations.id", ondelete="SET NULL"), nullable=True)
    
    # PATIENT DEMOGRAPHICS
    patient_id = Column(String, unique=True)  # External patient/medical record number
    patient_name = Column(String, nullable=False)
    gender = Column(String)  # CHECK constraint handled in migration
    date_of_birth = Column(Date)
    nationality = Column(String)
    address = Column(JSONB)  # e.g. {"line1": "...", "city": "...", "state": "...", "postcode": "...", "country": "..."}
    
    # DIAGNOSIS DETAILS
    diagnosis_date = Column(Date, nullable=False)
    age_at_diagnosis = Column(Integer)  # CHECK constraint handled in migration
    
    # ICD-11 OFFICIAL DISEASE CODING (FULL STRUCTURE)
    icd11_main_code = Column(String, nullable=False)  # Required by registry standards
    icd11_description = Column(String)  # WHO title
    icd11_composite_expression = Column(String)  # Post-coordination expression
    icd11_manifestation_code = Column(String)
    manifestation = Column(String)
    icd11_topography_code = Column(String)
    icd11_topography = Column(String)
    icd11_morphology_code = Column(String)
    icd11_morphology = Column(String)
    icd11_behavior_code = Column(String)  # /0, /1, /2, /3
    icd11_stage_code = Column(String)
    laterality = Column(String)  # CHECK constraint handled in migration
    
    # TNM STAGING (AJCC)
    t_category = Column(String)  # CHECK constraint handled in migration
    n_category = Column(String)  # CHECK constraint handled in migration
    m_category = Column(String)  # CHECK constraint handled in migration
    
    # ADDITIONAL CANCER REGISTRY FIELDS
    multiple_primary_flag = Column(Boolean)
    basis_of_diagnosis = Column(String)  # CHECK constraint handled in migration
    primary_site_confirmed = Column(Boolean)
    
    # TREATMENT INFORMATION
    surgery_done = Column(Boolean)
    surgery_date = Column(Date)
    chemotherapy_done = Column(Boolean)
    chemo_start_date = Column(Date)
    radiotherapy_done = Column(Boolean)
    hormonal_therapy = Column(Boolean)
    immunotherapy = Column(Boolean)
    treatment_intent = Column(String)  # CHECK constraint handled in migration
    treatment_notes = Column(Text)
    
    # FOLLOW-UP DATA
    followup_date = Column(Date)
    vital_status = Column(String)  # CHECK constraint handled in migration
    cause_of_death_icd11 = Column(String)
    recurrence = Column(Boolean)
    recurrence_date = Column(Date)
    metastasis = Column(Boolean)
    survival_months = Column(Integer)  # CHECK constraint handled in migration
    followup_notes = Column(Text)
    
    # REGISTRY METADATA
    data_source = Column(String)  # CHECK constraint handled in migration
    entry_mode = Column(String)  # e.g., Web, BulkCSV, Imported
    entered_by = Column(String)  # Text field from form (e.g., "Registrar")
    updated_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"), nullable=True)  # Who last updated (system field)
    validation_status = Column(String, default='Pending')  # CHECK constraint handled in migration
    
    # SYSTEM FIELDS (not in form, but needed for system operation)
    is_active = Column(Boolean, default=True)
    entry_timestamp = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    last_modified = Column(TIMESTAMP(timezone=True), server_default=text('now()'), onupdate=text('now()'))
