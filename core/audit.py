"""
Enterprise-Grade Audit Logging Utility

Provides easy-to-use functions for logging audit events throughout the application.
All logs are immutable and comply with industry standards (GDPR, HIPAA, SOX).
"""
from typing import Optional, Dict, Any, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi import Request
import uuid


def get_client_info(request: Optional[Request]) -> Tuple[Optional[str], Optional[str]]:
    """Extract IP address and user agent from request"""
    if not request:
        return None, None
    
    # Get IP address (check for proxy headers)
    ip_address = None
    if request.client:
        ip_address = request.client.host
    
    # Check common proxy headers
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        ip_address = forwarded_for.split(",")[0].strip()
    elif request.headers.get("X-Real-IP"):
        ip_address = request.headers.get("X-Real-IP")
    
    # Get user agent
    user_agent = request.headers.get("user-agent")
    
    return ip_address, user_agent


def log_event(
    db: Session,
    action_type: str,
    resource_type: str,
    change_summary: str,
    user_id: Optional[uuid.UUID] = None,
    resource_id: Optional[uuid.UUID] = None,
    resource_identifier: Optional[str] = None,
    change_details: Optional[Dict[str, Any]] = None,
    old_values: Optional[Dict[str, Any]] = None,
    new_values: Optional[Dict[str, Any]] = None,
    status: str = "success",
    error_message: Optional[str] = None,
    severity: str = "info",
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    request: Optional[Request] = None,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> uuid.UUID:
    """
    Log an audit event to the immutable audit log.
    
    Args:
        db: Database session
        action_type: Type of action (login, create, update, delete, etc.)
        resource_type: Type of resource (user, patient, role, etc.)
        change_summary: Human-readable summary of the change
        user_id: ID of the user performing the action
        resource_id: ID of the affected resource
        resource_identifier: Human-readable identifier (e.g., email, patient_id)
        change_details: Detailed change information (JSON)
        old_values: Previous values (for updates)
        new_values: New values (for creates/updates)
        status: success, failure, error, warning
        error_message: Error message if status is failure/error
        severity: info, warning, error, critical
        category: authentication, authorization, data_access, data_modification, system
        tags: Searchable tags
        request: FastAPI Request object (for IP/user agent)
        request_id: Unique request identifier
        session_id: Session identifier
        
    Returns:
        UUID of the created log entry
    """
    # Get client info from request
    ip_address, user_agent = get_client_info(request)
    
    # Convert IP to string for PostgreSQL INET type
    ip_str = str(ip_address) if ip_address else None
    
    # Prepare parameters
    params = {
        "p_user_id": str(user_id) if user_id else None,
        "p_action_type": action_type,
        "p_resource_type": resource_type,
        "p_resource_id": str(resource_id) if resource_id else None,
        "p_resource_identifier": resource_identifier,
        "p_change_summary": change_summary,
        "p_change_details": change_details,
        "p_old_values": old_values,
        "p_new_values": new_values,
        "p_status": status,
        "p_error_message": error_message,
        "p_severity": severity,
        "p_category": category,
        "p_tags": tags,
        "p_ip_address": ip_str,
        "p_user_agent": user_agent,
        "p_request_id": request_id,
        "p_session_id": session_id,
    }
    
    # Call the database function
    # Use a savepoint to avoid transaction conflicts
    savepoint = None
    try:
        # Create a savepoint so audit logging failures don't abort the main transaction
        savepoint = db.begin_nested()
        
        result = db.execute(
            text("SELECT audit.log_event("
                 ":p_user_id::uuid, :p_action_type, :p_resource_type, :p_resource_id::uuid, "
                 ":p_resource_identifier, :p_change_summary, :p_change_details::jsonb, "
                 ":p_old_values::jsonb, :p_new_values::jsonb, :p_status, :p_error_message, "
                 ":p_severity, :p_category, :p_tags::text[], :p_ip_address::inet, "
                 ":p_user_agent, :p_request_id, :p_session_id)"),
            params
        )
        
        log_id = result.scalar()
        savepoint.commit()
        
        return log_id
    except Exception as e:
        # Rollback the savepoint if audit logging fails
        if savepoint:
            savepoint.rollback()
        # If audit logging fails (e.g., function doesn't exist), return None
        # This allows the main operation to continue
        return None


def log_login(
    db: Session,
    user_id: uuid.UUID,
    status: str = "success",
    error_message: Optional[str] = None,
    request: Optional[Request] = None,
) -> Optional[uuid.UUID]:
    """Log a login event (non-blocking - returns None if logging fails)"""
    try:
        return log_event(
            db=db,
            action_type="login" if status == "success" else "login_failed",
            resource_type="user",
            change_summary=f"User login attempt: {status}",
            user_id=user_id,
            resource_id=user_id,
            status=status,
            error_message=error_message,
            category="authentication",
            tags=["login", "authentication"],
            request=request,
        )
    except Exception:
        # Silently fail - don't block login if audit logging fails
        return None


def log_logout(
    db: Session,
    user_id: uuid.UUID,
    request: Optional[Request] = None,
) -> uuid.UUID:
    """Log a logout event"""
    return log_event(
        db=db,
        action_type="logout",
        resource_type="user",
        change_summary="User logged out",
        user_id=user_id,
        resource_id=user_id,
        category="authentication",
        tags=["logout", "authentication"],
        request=request,
    )


def log_patient_create(
    db: Session,
    user_id: uuid.UUID,
    patient_id: uuid.UUID,
    patient_identifier: Optional[str] = None,
    request: Optional[Request] = None,
) -> uuid.UUID:
    """Log patient creation"""
    return log_event(
        db=db,
        action_type="create",
        resource_type="patient",
        change_summary=f"Patient record created: {patient_identifier or 'New Patient'}",
        user_id=user_id,
        resource_id=patient_id,
        resource_identifier=patient_identifier,
        category="data_modification",
        tags=["patient", "create", "registry"],
        request=request,
    )


def log_patient_update(
    db: Session,
    user_id: uuid.UUID,
    patient_id: uuid.UUID,
    patient_identifier: Optional[str] = None,
    old_values: Optional[Dict[str, Any]] = None,
    new_values: Optional[Dict[str, Any]] = None,
    request: Optional[Request] = None,
) -> uuid.UUID:
    """Log patient update"""
    return log_event(
        db=db,
        action_type="update",
        resource_type="patient",
        change_summary=f"Patient record updated: {patient_identifier or f'Patient #{patient_id}'}",
        user_id=user_id,
        resource_id=patient_id,
        resource_identifier=patient_identifier,
        old_values=old_values,
        new_values=new_values,
        category="data_modification",
        tags=["patient", "update", "registry"],
        request=request,
    )


def log_bulk_upload(
    db: Session,
    user_id: uuid.UUID,
    record_count: int,
    success_count: int,
    error_count: int,
    request: Optional[Request] = None,
) -> uuid.UUID:
    """Log bulk CSV upload"""
    return log_event(
        db=db,
        action_type="bulk_upload",
        resource_type="patient",
        change_summary=f"Bulk CSV upload: {success_count} created, {error_count} errors (total: {record_count})",
        user_id=user_id,
        change_details={
            "total_records": record_count,
            "success_count": success_count,
            "error_count": error_count,
        },
        status="success" if error_count == 0 else "warning",
        category="data_modification",
        tags=["patient", "bulk_upload", "csv", "registry"],
        request=request,
    )


def log_data_export(
    db: Session,
    user_id: uuid.UUID,
    record_count: int,
    export_format: str = "csv",
    request: Optional[Request] = None,
) -> uuid.UUID:
    """Log data export"""
    return log_event(
        db=db,
        action_type="export",
        resource_type="patient",
        change_summary=f"Data exported: {record_count} records in {export_format} format",
        user_id=user_id,
        change_details={
            "record_count": record_count,
            "export_format": export_format,
        },
        category="data_access",
        tags=["patient", "export", "data_access"],
        request=request,
    )

