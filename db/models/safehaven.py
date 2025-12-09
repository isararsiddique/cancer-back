"""
UM-HDSH (University of Malaya - Health Data Safe Haven) Models
Comprehensive models for all 10 modules of the Data Safehaven system
"""
from sqlalchemy import Column, String, ForeignKey, Date, Integer, Boolean, Text, CheckConstraint, Numeric, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP, ARRAY
from sqlalchemy import text
from sqlalchemy.orm import relationship
import uuid

from db.base import Base


# ============================================================================
# MODULE 1: EMR → Registry Connector
# ============================================================================

class EMRSyncStatus(Base):
    """Tracks EMR synchronization status and CDC events"""
    __tablename__ = "emr_sync_status"
    __table_args__ = {"schema": "registry"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    emr_source = Column(String, nullable=False)  # e.g., "iPesakit"
    last_sync_at = Column(TIMESTAMP(timezone=True))
    last_sync_status = Column(String, default='success')  # success, failed, partial
    records_synced = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    error_message = Column(Text)
    sync_config = Column(JSONB)  # Sync interval, filters, etc.
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    updated_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'), onupdate=text('now()'))


class EMREvent(Base):
    """CDC events from EMR system"""
    __tablename__ = "emr_events"
    __table_args__ = (
        Index("idx_emr_events_timestamp", "event_timestamp"),
        Index("idx_emr_events_status", "status"),
        {"schema": "registry"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    emr_source = Column(String, nullable=False)
    event_type = Column(String, nullable=False)  # created, updated, deleted
    patient_id = Column(String)  # External EMR patient ID
    registry_patient_id = Column(UUID(as_uuid=True), ForeignKey("registry.patients.id", ondelete="SET NULL"))
    event_data = Column(JSONB, nullable=False)  # Raw event payload
    event_timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    processed = Column(Boolean, default=False)
    processed_at = Column(TIMESTAMP(timezone=True))
    status = Column(String, default='pending')  # pending, processed, failed
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))


# ============================================================================
# MODULE 2: Registry Standardisation Engine
# ============================================================================

class StandardizationJob(Base):
    """Tracks standardization/validation jobs"""
    __tablename__ = "standardization_jobs"
    __table_args__ = {"schema": "registry"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type = Column(String, nullable=False)  # validation, icd11_mapping, staging_conversion
    patient_id = Column(UUID(as_uuid=True), ForeignKey("registry.patients.id", ondelete="CASCADE"))
    input_data = Column(JSONB, nullable=False)
    output_data = Column(JSONB)
    status = Column(String, default='pending')  # pending, processing, completed, failed
    validation_errors = Column(JSONB)  # Array of validation errors
    icd11_mappings = Column(JSONB)  # Mapped ICD-11 codes
    ajcc_staging = Column(JSONB)  # Converted AJCC staging
    error_message = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    completed_at = Column(TIMESTAMP(timezone=True))


class ICD11MappingCache(Base):
    """Cache for ICD-11 term mappings"""
    __tablename__ = "icd11_mapping_cache"
    __table_args__ = (
        Index("idx_icd11_cache_term", "diagnosis_term"),
        {"schema": "registry"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    diagnosis_term = Column(String, nullable=False)
    icd11_code = Column(String, nullable=False)
    icd11_description = Column(String)
    confidence_score = Column(Numeric(5, 2))  # 0.00 to 1.00
    mapping_method = Column(String)  # exact_match, fuzzy_match, manual
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    updated_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'), onupdate=text('now()'))


# ============================================================================
# MODULE 3: Governance & Request Management (Enhanced)
# ============================================================================

class ResearchProject(Base):
    """Enhanced research project with full workflow"""
    __tablename__ = "research_projects"
    __table_args__ = {"schema": "registry"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_code = Column(String, unique=True, nullable=False)  # PROJ-YYYYMMDD-####
    
    # Project Details
    project_title = Column(String, nullable=False)
    project_description = Column(Text, nullable=False)
    researcher_name = Column(String, nullable=False)
    researcher_email = Column(String, nullable=False)
    researcher_affiliation = Column(String)
    clinician_name = Column(String)  # Supervising clinician
    clinician_email = Column(String)
    mrec_number = Column(String)  # Medical Research Ethics Committee number
    iexplore_id = Column(String)  # Integration with iExplore system
    
    # Workflow State
    status = Column(String, nullable=False, default='draft')  # draft, submitted, ethics_review, steering_review, approved, rejected, expired, archived
    current_step = Column(String)  # Current workflow step
    workflow_state = Column(JSONB)  # Full state machine state
    
    # Approval Chain
    submitted_at = Column(TIMESTAMP(timezone=True))
    ethics_approved_at = Column(TIMESTAMP(timezone=True))
    ethics_approved_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    steering_approved_at = Column(TIMESTAMP(timezone=True))
    steering_approved_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    approved_at = Column(TIMESTAMP(timezone=True))
    approved_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    rejected_at = Column(TIMESTAMP(timezone=True))
    rejected_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    rejection_reason = Column(Text)
    
    # Project Scope
    requested_variables = Column(JSONB, nullable=False)  # List of requested data fields
    date_range_from = Column(Date)
    date_range_to = Column(Date)
    filters = Column(JSONB)  # Additional filter criteria
    
    # Access Management
    access_token = Column(String, unique=True)  # Time-limited access token
    token_issued_at = Column(TIMESTAMP(timezone=True))
    token_expires_at = Column(TIMESTAMP(timezone=True))
    access_granted_at = Column(TIMESTAMP(timezone=True))
    
    # Project Timeline
    project_start_date = Column(Date)
    project_end_date = Column(Date)
    extension_requested = Column(Boolean, default=False)
    extension_reason = Column(Text)
    
    # Metadata
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    updated_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'), onupdate=text('now()'))
    created_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("core.tenants.id", ondelete="SET NULL"))
    organization_id = Column(UUID(as_uuid=True), ForeignKey("core.organizations.id", ondelete="SET NULL"))


class ProjectWorkflowHistory(Base):
    """Audit trail for project workflow state changes"""
    __tablename__ = "project_workflow_history"
    __table_args__ = {"schema": "registry"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("registry.research_projects.id", ondelete="CASCADE"), nullable=False)
    from_status = Column(String)
    to_status = Column(String, nullable=False)
    changed_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    change_reason = Column(Text)
    workflow_metadata = Column("metadata", JSONB)  # Use Column name parameter to keep DB column as "metadata"
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))


# ============================================================================
# MODULE 4: Extraction & Data Curation Pipeline
# ============================================================================

class ExtractionJob(Base):
    """Tracks data extraction jobs for approved projects"""
    __tablename__ = "extraction_jobs"
    __table_args__ = {"schema": "registry"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("registry.research_projects.id", ondelete="CASCADE"), nullable=False)
    job_type = Column(String, nullable=False)  # initial_extraction, reprocess, update
    status = Column(String, default='pending')  # pending, running, completed, failed
    requested_variables = Column(JSONB, nullable=False)
    filters = Column(JSONB)
    date_range_from = Column(Date)
    date_range_to = Column(Date)
    
    # Extraction Results
    records_extracted = Column(Integer, default=0)
    extraction_query = Column(Text)  # SQL query used
    extraction_metadata = Column(JSONB)
    
    # Data Quality
    qa_status = Column(String)  # pending, passed, failed, warning
    qa_report = Column(JSONB)  # Great Expectations report
    qa_errors = Column(JSONB)  # Quality check failures
    
    # Output
    curated_dataset_path = Column(String)  # Path to curated dataset (S3/MinIO)
    curated_dataset_size = Column(Integer)  # Size in bytes
    curated_dataset_hash = Column(String)  # SHA-256 hash for integrity
    
    # Timestamps
    started_at = Column(TIMESTAMP(timezone=True))
    completed_at = Column(TIMESTAMP(timezone=True))
    error_message = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    created_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))


# ============================================================================
# MODULE 5: Anonymisation & Re-identification Risk Assessment
# ============================================================================

class AnonymizationJob(Base):
    """Tracks anonymization jobs and risk assessments"""
    __tablename__ = "anonymization_jobs"
    __table_args__ = {"schema": "registry"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("registry.research_projects.id", ondelete="CASCADE"), nullable=False)
    extraction_job_id = Column(UUID(as_uuid=True), ForeignKey("registry.extraction_jobs.id", ondelete="SET NULL"))
    status = Column(String, default='pending')  # pending, running, completed, failed
    
    # Input
    input_dataset_path = Column(String, nullable=False)
    anonymization_rules = Column(JSONB, nullable=False)  # Configuration for masking, generalization, etc.
    
    # Processing
    records_processed = Column(Integer, default=0)
    transformations_applied = Column(JSONB)  # List of transformations
    
    # Risk Assessment
    k_anonymity = Column(Integer)  # k-anonymity value
    l_diversity = Column(Numeric(5, 2))  # l-diversity value
    risk_score = Column(Numeric(5, 2))  # Overall risk score (0-100)
    risk_level = Column(String)  # low, medium, high, critical
    risk_report = Column(JSONB)  # Detailed risk assessment
    
    # Output
    anonymized_dataset_path = Column(String)  # Path to anonymized dataset
    anonymized_dataset_size = Column(Integer)
    anonymized_dataset_hash = Column(String)
    anonymization_report_path = Column(String)  # Path to signed report
    anonymization_report_hash = Column(String)
    
    # Approval
    approved = Column(Boolean, default=False)
    approved_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    approved_at = Column(TIMESTAMP(timezone=True))
    
    # Timestamps
    started_at = Column(TIMESTAMP(timezone=True))
    completed_at = Column(TIMESTAMP(timezone=True))
    error_message = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    created_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))


# ============================================================================
# MODULE 6: Secure Upload & Storage (Safe Haven)
# ============================================================================

class SafeHavenStorage(Base):
    """Tracks datasets stored in Safe Haven"""
    __tablename__ = "safehaven_storage"
    __table_args__ = {"schema": "registry"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("registry.research_projects.id", ondelete="CASCADE"), nullable=False)
    anonymization_job_id = Column(UUID(as_uuid=True), ForeignKey("registry.anonymization_jobs.id", ondelete="SET NULL"))
    
    # Storage Details
    storage_type = Column(String, nullable=False)  # minio, s3, local
    bucket_name = Column(String, nullable=False)
    object_key = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)  # Full path
    
    # Encryption
    encryption_method = Column(String, default='AES-256')
    encryption_key_id = Column(String)  # KMS key ID if used
    encrypted = Column(Boolean, default=True)
    
    # Data Residency
    data_residency_country = Column(String, default='Malaysia')
    data_residency_region = Column(String)
    storage_location = Column(String)  # Physical location
    
    # Access Control
    access_policy = Column(JSONB)  # Bucket/object ACLs
    project_acl = Column(JSONB)  # Project-specific access rules
    
    # Metadata
    file_size = Column(Integer)
    file_hash = Column(String)  # SHA-256
    mime_type = Column(String)
    uploaded_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    
    # Audit
    upload_event_id = Column(UUID(as_uuid=True))  # Reference to audit log
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))


# ============================================================================
# MODULE 7: In-Safe-Haven Research Environment (Compute + Notebooks)
# ============================================================================

class ComputeWorkspace(Base):
    """Tracks JupyterHub/RStudio workspaces in Safe Haven"""
    __tablename__ = "compute_workspaces"
    __table_args__ = {"schema": "registry"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("registry.research_projects.id", ondelete="CASCADE"), nullable=False)
    workspace_name = Column(String, nullable=False)
    workspace_type = Column(String, nullable=False)  # jupyterlab, rstudio, custom
    
    # Kubernetes/Infrastructure
    namespace = Column(String)  # Kubernetes namespace
    pod_name = Column(String)
    node_name = Column(String)
    resource_quota = Column(JSONB)  # CPU, memory, GPU limits
    
    # Access
    access_token = Column(String, unique=True)  # Time-limited workspace token
    access_url = Column(String)  # URL to access workspace
    token_issued_at = Column(TIMESTAMP(timezone=True))
    token_expires_at = Column(TIMESTAMP(timezone=True))
    
    # Dataset Mount
    dataset_mount_path = Column(String)  # Read-only mount path
    dataset_storage_id = Column(UUID(as_uuid=True), ForeignKey("registry.safehaven_storage.id", ondelete="SET NULL"))
    
    # Security Configuration
    network_restrictions = Column(JSONB)  # Egress restrictions
    download_disabled = Column(Boolean, default=True)
    export_disabled = Column(Boolean, default=True)
    session_recording_enabled = Column(Boolean, default=True)
    
    # Status
    status = Column(String, default='pending')  # pending, running, stopped, terminated, expired
    started_at = Column(TIMESTAMP(timezone=True))
    stopped_at = Column(TIMESTAMP(timezone=True))
    last_activity_at = Column(TIMESTAMP(timezone=True))
    
    # Metadata
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    created_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))


class WorkspaceSession(Base):
    """Logs workspace sessions and activities"""
    __tablename__ = "workspace_sessions"
    __table_args__ = (
        Index("idx_workspace_sessions_workspace", "workspace_id"),
        Index("idx_workspace_sessions_timestamp", "session_start"),
        {"schema": "registry"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("registry.compute_workspaces.id", ondelete="CASCADE"), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("registry.research_projects.id", ondelete="CASCADE"), nullable=False)
    
    session_id = Column(String, unique=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    
    session_start = Column(TIMESTAMP(timezone=True), nullable=False)
    session_end = Column(TIMESTAMP(timezone=True))
    duration_seconds = Column(Integer)
    
    # Activity Logging
    queries_executed = Column(Integer, default=0)
    queries_log = Column(JSONB)  # Log of SQL/queries executed
    files_accessed = Column(ARRAY(String))  # List of files accessed
    actions_log = Column(JSONB)  # Other actions (plot generation, etc.)
    
    # Security Events
    blocked_actions = Column(JSONB)  # Attempted downloads, exports, etc.
    security_alerts = Column(JSONB)
    
    # Recording
    session_recording_path = Column(String)  # Path to session recording
    session_log_path = Column(String)  # Path to activity log
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))


# ============================================================================
# MODULE 8: Pre-built Dashboards & Analytics Tools
# ============================================================================

class DashboardQuery(Base):
    """Logs all dashboard queries for audit"""
    __tablename__ = "dashboard_queries"
    __table_args__ = (
        Index("idx_dashboard_queries_project", "project_id"),
        Index("idx_dashboard_queries_timestamp", "query_timestamp"),
        {"schema": "registry"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("registry.research_projects.id", ondelete="CASCADE"), nullable=False)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("registry.compute_workspaces.id", ondelete="SET NULL"))
    
    dashboard_type = Column(String, nullable=False)  # survival, trend, predictive, cohort
    query_type = Column(String, nullable=False)  # sql, aggregate, plot
    
    # Query Details
    query_text = Column(Text)  # SQL or query definition
    query_filters = Column(JSONB)  # Applied filters
    query_params = Column(JSONB)  # Query parameters
    
    # Results (aggregate only, no raw data)
    result_type = Column(String)  # plot_image, aggregate_stats, summary
    result_metadata = Column(JSONB)  # Metadata about results (not raw data)
    result_size = Column(Integer)  # Size of result in bytes
    
    # User Context
    user_id = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    query_timestamp = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))
    
    # Audit
    execution_time_ms = Column(Integer)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))


# ============================================================================
# MODULE 9: IAM, Audit & Compliance Engine (Enhanced)
# ============================================================================
# Note: AuditLog already exists in audit.py, but we add project-specific audit

class ProjectAuditLog(Base):
    """Project-specific audit log entries"""
    __tablename__ = "project_audit_logs"
    __table_args__ = (
        Index("idx_project_audit_project", "project_id"),
        Index("idx_project_audit_timestamp", "timestamp"),
        {"schema": "registry"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("registry.research_projects.id", ondelete="CASCADE"), nullable=False)
    
    # Link to main audit log
    audit_log_id = Column(UUID(as_uuid=True), ForeignKey("public.logs.id", ondelete="CASCADE"))
    
    # Project-specific context
    action_type = Column(String, nullable=False)
    resource_type = Column(String, nullable=False)
    resource_id = Column(UUID(as_uuid=True))
    
    # User context
    user_id = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    user_email = Column(String)
    
    # Details
    change_summary = Column(Text, nullable=False)
    change_details = Column(JSONB)
    
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))
    ip_address = Column(String)
    user_agent = Column(Text)
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))


# ============================================================================
# MODULE 10: Project Closure, Archival & Extension Handler
# ============================================================================

class ProjectArchive(Base):
    """Archived project data and metadata"""
    __tablename__ = "project_archives"
    __table_args__ = {"schema": "registry"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("registry.research_projects.id", ondelete="CASCADE"), nullable=False)
    
    # Archive Details
    archive_reason = Column(String, nullable=False)  # expired, completed, terminated, extension_denied
    archived_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))
    archived_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    
    # Archived Data
    dataset_archive_path = Column(String)  # Path to archived dataset
    metadata_archive_path = Column(String)  # Path to archived metadata
    iexplore_metadata = Column(JSONB)  # Metadata sent to iExplore
    
    # Access Revocation
    access_tokens_revoked = Column(ARRAY(String))  # List of revoked tokens
    active_sessions_terminated = Column(Integer, default=0)
    revocation_timestamp = Column(TIMESTAMP(timezone=True))
    
    # Retention
    retention_until = Column(TIMESTAMP(timezone=True))  # When archive can be deleted
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))


class ProjectExtensionRequest(Base):
    """Extension requests for projects"""
    __tablename__ = "project_extension_requests"
    __table_args__ = {"schema": "registry"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("registry.research_projects.id", ondelete="CASCADE"), nullable=False)
    
    # Extension Details
    requested_extension_days = Column(Integer, nullable=False)
    requested_new_end_date = Column(Date, nullable=False)
    extension_reason = Column(Text, nullable=False)
    additional_justification = Column(Text)
    
    # Workflow
    status = Column(String, default='pending')  # pending, approved, rejected
    submitted_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    reviewed_at = Column(TIMESTAMP(timezone=True))
    reviewed_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))
    review_notes = Column(Text)
    
    # Outcome
    approved_extension_days = Column(Integer)
    approved_new_end_date = Column(Date)
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    created_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"))


# ============================================================================
# ML TRAINING & MODEL MANAGEMENT
# ============================================================================

class MLTrainingResult(Base):
    """Stores ML model training results and metadata"""
    __tablename__ = "ml_training_results"
    __table_args__ = (
        Index("idx_ml_training_user", "created_by"),
        Index("idx_ml_training_request", "research_request_id"),
        Index("idx_ml_training_created", "created_at"),
        {"schema": "registry"}
    )
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String, unique=True, nullable=False)  # Client-generated model ID
    
    # Association with research request
    research_request_id = Column(UUID(as_uuid=True), ForeignKey("registry.research_requests.id", ondelete="SET NULL"), nullable=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("registry.research_projects.id", ondelete="SET NULL"), nullable=True)
    
    # Model Configuration
    algorithm = Column(String, nullable=False)  # xgboost, random_forest, neural_network
    target_variable = Column(String, nullable=False)
    features = Column(JSONB, nullable=False)  # List of feature column names
    hyperparameters = Column(JSONB)  # Model hyperparameters
    
    # Training Configuration
    test_size = Column(Numeric(5, 4))  # Test split ratio (e.g., 0.2)
    random_state = Column(Integer)
    custom_pipeline = Column(Text)  # Custom Python pipeline code if used
    
    # Training Results
    metrics = Column(JSONB, nullable=False)  # All metrics (accuracy, precision, R², MSE, etc.)
    feature_importance = Column(JSONB)  # Feature importance scores
    predictions = Column(JSONB)  # Sample predictions for visualization
    confusion_matrix = Column(JSONB)  # Confusion matrix (for classification)
    roc_curve = Column(JSONB)  # ROC curve data (for binary classification)
    cv_scores = Column(JSONB)  # Cross-validation scores
    
    # Training Metadata
    training_status = Column(String, default='completed')  # completed, failed, training
    error_message = Column(Text)  # Error if training failed
    training_duration_seconds = Column(Integer)  # How long training took
    resource_metrics = Column(JSONB)  # CPU and memory usage metrics
    
    # Data Statistics
    n_samples = Column(Integer)  # Number of training samples
    n_features = Column(Integer)  # Number of features
    n_train = Column(Integer)  # Training set size
    n_test = Column(Integer)  # Test set size
    n_val = Column(Integer)  # Validation set size (if used)
    
    # Model Artifacts (optional - for server-side model storage)
    model_artifact_path = Column(String)  # Path to saved model file (pickle/joblib)
    model_artifact_size = Column(Integer)  # Size in bytes
    model_artifact_hash = Column(String)  # SHA-256 hash
    
    # Metadata
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    created_by = Column(UUID(as_uuid=True), ForeignKey("rbac.users.id"), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'), onupdate=text('now()'))
    
    # Tags and Notes
    tags = Column(ARRAY(String))  # User-defined tags for organization
    notes = Column(Text)  # User notes about this model

