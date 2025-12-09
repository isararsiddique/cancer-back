"""
Enterprise-Grade Audit Log API Endpoints

Provides access to immutable audit logs with filtering, search, and export capabilities.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import text, and_, or_, desc, func
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel
import csv
from io import StringIO

from core.deps import get_db, get_current_user
from db.models.users import User
from db.models.audit import AuditLog
from core.audit import log_event

router = APIRouter(prefix="/audit", tags=["Audit Logs"])


# ============================================
# Response Models
# ============================================
class AuditLogResponse(BaseModel):
    """Audit log entry response"""
    id: str
    timestamp: datetime
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    user_name: Optional[str] = None
    user_roles: Optional[List[str]] = None
    tenant_id: Optional[str] = None
    tenant_name: Optional[str] = None
    organization_id: Optional[str] = None
    organization_name: Optional[str] = None
    action_type: str
    resource_type: str
    resource_id: Optional[str] = None
    resource_identifier: Optional[str] = None
    change_summary: str
    change_details: Optional[Dict[str, Any]] = None
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    severity: str
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    retention_until: Optional[datetime] = None
    compliance_flags: Optional[List[str]] = None
    is_sensitive: bool
    created_at: datetime
    checksum: Optional[str] = None

    @classmethod
    def from_orm(cls, log: AuditLog):
        """Convert AuditLog model to response"""
        return cls(
            id=str(log.id),
            timestamp=log.timestamp,
            user_id=str(log.user_id) if log.user_id else None,
            user_email=log.user_email,
            user_name=log.user_name,
            user_roles=log.user_roles,
            tenant_id=str(log.tenant_id) if log.tenant_id else None,
            tenant_name=log.tenant_name,
            organization_id=str(log.organization_id) if log.organization_id else None,
            organization_name=log.organization_name,
            action_type=log.action_type,
            resource_type=log.resource_type,
            resource_id=str(log.resource_id) if log.resource_id else None,
            resource_identifier=log.resource_identifier,
            change_summary=log.change_summary,
            change_details=log.change_details,
            old_values=log.old_values,
            new_values=log.new_values,
            ip_address=str(log.ip_address) if log.ip_address else None,
            user_agent=log.user_agent,
            request_id=log.request_id,
            session_id=log.session_id,
            status=log.status,
            error_message=log.error_message,
            error_code=log.error_code,
            severity=log.severity,
            category=log.category,
            tags=log.tags,
            retention_until=log.retention_until,
            compliance_flags=log.compliance_flags,
            is_sensitive=log.is_sensitive,
            created_at=log.created_at,
            checksum=log.checksum
        )

    class Config:
        from_attributes = True


class AuditLogListResponse(BaseModel):
    """Paginated audit log list response"""
    total: int
    skip: int
    limit: int
    logs: List[AuditLogResponse]


# ============================================
# Helper Functions
# ============================================
def _build_log_query(
    db: Session,
    user_id: Optional[str] = None,
    action_type: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    category: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    search: Optional[str] = None,
    current_user: Optional[User] = None,
):
    """Build query with filters and access control"""
    query = db.query(AuditLog)
    
    # Access control: Super admins see all, others see their own/tenant logs
    is_super_admin = any(r.slug == "super_admin" for r in current_user.roles) if current_user else False
    
    if not is_super_admin:
        # Users can see their own logs or logs from their tenant
        if current_user:
            query = query.filter(
                or_(
                    AuditLog.user_id == current_user.id,
                    AuditLog.tenant_id == current_user.tenant_id
                )
            )
    
    # Apply filters
    if user_id:
        query = query.filter(AuditLog.user_id == user_id)
    
    if action_type:
        query = query.filter(AuditLog.action_type == action_type)
    
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)
    
    if resource_id:
        query = query.filter(AuditLog.resource_id == resource_id)
    
    if status:
        query = query.filter(AuditLog.status == status)
    
    if severity:
        query = query.filter(AuditLog.severity == severity)
    
    if category:
        query = query.filter(AuditLog.category == category)
    
    if start_date:
        query = query.filter(AuditLog.timestamp >= start_date)
    
    if end_date:
        query = query.filter(AuditLog.timestamp <= end_date)
    
    if search:
        search_filter = or_(
            AuditLog.change_summary.ilike(f"%{search}%"),
            AuditLog.user_email.ilike(f"%{search}%"),
            AuditLog.user_name.ilike(f"%{search}%"),
            AuditLog.resource_identifier.ilike(f"%{search}%")
        )
        query = query.filter(search_filter)
    
    return query


# ============================================
# API Endpoints
# ============================================
@router.get("/logs", response_model=AuditLogListResponse)
def list_audit_logs(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    action_type: Optional[str] = Query(None, description="Filter by action type"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    resource_id: Optional[str] = Query(None, description="Filter by resource ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    category: Optional[str] = Query(None, description="Filter by category"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    search: Optional[str] = Query(None, description="Search in summary, email, name, identifier"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List audit logs with filtering and pagination.
    
    Access control:
    - Super admins: See all logs
    - Other users: See only their own logs and logs from their tenant
    """
    query = _build_log_query(
        db=db,
        user_id=user_id,
        action_type=action_type,
        resource_type=resource_type,
        resource_id=resource_id,
        status=status,
        severity=severity,
        category=category,
        start_date=start_date,
        end_date=end_date,
        search=search,
        current_user=current_user
    )
    
    # Get total count
    total = query.count()
    
    # Apply pagination and ordering
    logs = query.order_by(desc(AuditLog.timestamp)).offset(skip).limit(limit).all()
    
    return AuditLogListResponse(
        total=total,
        skip=skip,
        limit=limit,
        logs=[AuditLogResponse.from_orm(log) for log in logs]
    )


@router.get("/logs/{log_id}", response_model=AuditLogResponse)
def get_audit_log(
    log_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific audit log entry by ID.
    
    Access control:
    - Super admins: Can access any log
    - Other users: Can only access their own logs or logs from their tenant
    """
    log = db.query(AuditLog).filter(AuditLog.id == log_id).first()
    
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audit log not found"
        )
    
    # Access control
    is_super_admin = any(r.slug == "super_admin" for r in current_user.roles)
    
    if not is_super_admin:
        if log.user_id != current_user.id and log.tenant_id != current_user.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this audit log"
            )
    
    return AuditLogResponse.from_orm(log)


@router.get("/logs/export/csv")
def export_audit_logs_csv(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    action_type: Optional[str] = Query(None, description="Filter by action type"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    category: Optional[str] = Query(None, description="Filter by category"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    search: Optional[str] = Query(None, description="Search in summary, email, name, identifier"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    request: Request = None
):
    """
    Export audit logs to CSV format.
    
    Access control:
    - Super admins: Can export all logs
    - Other users: Can only export their own logs or logs from their tenant
    """
    query = _build_log_query(
        db=db,
        user_id=user_id,
        action_type=action_type,
        resource_type=resource_type,
        status=status,
        severity=severity,
        category=category,
        start_date=start_date,
        end_date=end_date,
        search=search,
        current_user=current_user
    )
    
    # Get all matching logs (no pagination for export)
    logs = query.order_by(desc(AuditLog.timestamp)).all()
    
    # Create CSV
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        "ID", "Timestamp", "User Email", "User Name", "User Roles",
        "Tenant Name", "Organization Name",
        "Action Type", "Resource Type", "Resource ID", "Resource Identifier",
        "Change Summary", "Status", "Severity", "Category",
        "IP Address", "User Agent", "Error Message",
        "Tags", "Compliance Flags", "Created At"
    ])
    
    # Write data
    for log in logs:
        writer.writerow([
            str(log.id),
            log.timestamp.isoformat() if log.timestamp else "",
            log.user_email or "",
            log.user_name or "",
            ", ".join(log.user_roles) if log.user_roles else "",
            log.tenant_name or "",
            log.organization_name or "",
            log.action_type,
            log.resource_type,
            str(log.resource_id) if log.resource_id else "",
            log.resource_identifier or "",
            log.change_summary,
            log.status,
            log.severity,
            log.category or "",
            str(log.ip_address) if log.ip_address else "",
            log.user_agent or "",
            log.error_message or "",
            ", ".join(log.tags) if log.tags else "",
            ", ".join(log.compliance_flags) if log.compliance_flags else "",
            log.created_at.isoformat() if log.created_at else ""
        ])
    
    # Log the export
    log_event(
        db=db,
        action_type="export",
        resource_type="audit_log",
        change_summary=f"Audit logs exported to CSV by {current_user.email}",
        user_id=current_user.id,
        status="success",
        category="data_access",
        tags=["export", "csv", "audit_log"],
        request=request
    )
    
    from fastapi.responses import Response
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=audit_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    )


@router.get("/stats/summary")
def get_audit_stats(
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get audit log statistics summary.
    
    Returns counts by action type, status, severity, and category.
    Access control applies: users only see stats for their accessible logs.
    """
    # Build base query with access control
    base_query = _build_log_query(
        db=db,
        start_date=start_date,
        end_date=end_date,
        current_user=current_user
    )
    
    # Get total count
    total = base_query.count()
    
    # Get counts by action type (using the filtered query)
    action_query = base_query.with_entities(AuditLog.action_type, func.count(AuditLog.id).label('count'))
    action_query = action_query.group_by(AuditLog.action_type).order_by(func.count(AuditLog.id).desc())
    action_counts = action_query.all()
    
    # Get counts by status
    status_query = base_query.with_entities(AuditLog.status, func.count(AuditLog.id).label('count'))
    status_query = status_query.group_by(AuditLog.status).order_by(func.count(AuditLog.id).desc())
    status_counts = status_query.all()
    
    # Get counts by severity
    severity_query = base_query.with_entities(AuditLog.severity, func.count(AuditLog.id).label('count'))
    severity_query = severity_query.group_by(AuditLog.severity).order_by(func.count(AuditLog.id).desc())
    severity_counts = severity_query.all()
    
    return {
        "total": total,
        "by_action_type": {row[0]: row[1] for row in action_counts},
        "by_status": {row[0]: row[1] for row in status_counts},
        "by_severity": {row[0]: row[1] for row in severity_counts},
        "period": {
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None
        }
    }
