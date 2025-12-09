"""
UM-HDSH Research Project Management API
Enhanced governance workflow with full project lifecycle management
"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import text, and_, or_
from datetime import datetime, timedelta, date
from pydantic import BaseModel, EmailStr, Field
import secrets
import uuid

from core.deps import get_db, get_current_user
from db.models.safehaven import (
    ResearchProject, ProjectWorkflowHistory,
    ExtractionJob, AnonymizationJob,
    SafeHavenStorage, ComputeWorkspace,
    ProjectExtensionRequest, ProjectArchive
)
from db.models.users import User
from db.models.rbac import Role

router = APIRouter(prefix="/projects", tags=["projects"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ProjectCreate(BaseModel):
    """Create a new research project"""
    project_title: str
    project_description: str
    researcher_name: str
    researcher_email: EmailStr
    researcher_affiliation: Optional[str] = None
    clinician_name: Optional[str] = None
    clinician_email: Optional[EmailStr] = None
    mrec_number: Optional[str] = None
    iexplore_id: Optional[str] = None
    requested_variables: List[str] = Field(..., description="List of requested data fields")
    date_range_from: Optional[date] = None
    date_range_to: Optional[date] = None
    filters: Optional[Dict[str, Any]] = None
    project_start_date: Optional[date] = None
    project_end_date: Optional[date] = None


class ProjectUpdate(BaseModel):
    """Update project details (only in draft status)"""
    project_title: Optional[str] = None
    project_description: Optional[str] = None
    clinician_name: Optional[str] = None
    clinician_email: Optional[EmailStr] = None
    mrec_number: Optional[str] = None
    requested_variables: Optional[List[str]] = None
    date_range_from: Optional[date] = None
    date_range_to: Optional[date] = None
    filters: Optional[Dict[str, Any]] = None


class ProjectSubmit(BaseModel):
    """Submit project for review"""
    pass  # No additional fields needed


class ProjectApprove(BaseModel):
    """Approve project at a workflow step"""
    step: str = Field(..., description="Workflow step: ethics_review, steering_review, or final")
    notes: Optional[str] = None


class ProjectReject(BaseModel):
    """Reject project"""
    reason: str = Field(..., description="Rejection reason")


class ExtensionRequest(BaseModel):
    """Request project extension"""
    requested_extension_days: int = Field(..., gt=0, description="Number of days to extend")
    extension_reason: str = Field(..., description="Reason for extension")
    additional_justification: Optional[str] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_project_code() -> str:
    """Generate unique project code: PROJ-YYYYMMDD-####"""
    date_part = datetime.now().strftime("%Y%m%d")
    random_part = secrets.token_hex(2)  # 4 hex characters
    return f"PROJ-{date_part}-{random_part.upper()}"


def generate_access_token() -> str:
    """Generate secure access token for approved projects"""
    random_part = secrets.token_hex(16)  # 32 hex characters
    return f"SAFEHAVEN-TOKEN-{random_part}"


def check_project_permission(project: ResearchProject, user: User, db: Session) -> bool:
    """Check if user has permission to access project"""
    # Creator can always access
    if project.created_by == user.id:
        return True
    
    # Researcher email match
    if project.researcher_email.lower() == user.email.lower():
        return True
    
    # Admin roles can access
    user_roles = [role.slug for role in user.roles]
    if 'super_admin' in user_roles or 'ummc_admin' in user_roles:
        return True
    
    return False


# ============================================================================
# PROJECT CRUD OPERATIONS
# ============================================================================

@router.post("/", status_code=status.HTTP_201_CREATED)
def create_project(
    project_data: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new research project (draft status)"""
    # Generate unique project code
    project_code = generate_project_code()
    while db.query(ResearchProject).filter(ResearchProject.project_code == project_code).first():
        project_code = generate_project_code()
    
    project = ResearchProject(
        project_code=project_code,
        project_title=project_data.project_title,
        project_description=project_data.project_description,
        researcher_name=project_data.researcher_name,
        researcher_email=project_data.researcher_email,
        researcher_affiliation=project_data.researcher_affiliation,
        clinician_name=project_data.clinician_name,
        clinician_email=project_data.clinician_email,
        mrec_number=project_data.mrec_number,
        iexplore_id=project_data.iexplore_id,
        status='draft',
        current_step='draft',
        requested_variables=project_data.requested_variables,
        date_range_from=project_data.date_range_from,
        date_range_to=project_data.date_range_to,
        filters=project_data.filters or {},
        project_start_date=project_data.project_start_date,
        project_end_date=project_data.project_end_date,
        created_by=current_user.id,
        tenant_id=current_user.tenant_id,
        organization_id=current_user.organization_id,
        workflow_state={
            "current_step": "draft",
            "history": []
        }
    )
    
    db.add(project)
    db.commit()
    db.refresh(project)
    
    return {
        "project_id": str(project.id),
        "project_code": project.project_code,
        "status": project.status,
        "message": "Project created successfully. Submit when ready for review."
    }


@router.get("/{project_id}")
def get_project(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get project details"""
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid project ID")
    
    project = db.query(ResearchProject).filter(ResearchProject.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    
    if not check_project_permission(project, current_user, db):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    
    return {
        "project_id": str(project.id),
        "project_code": project.project_code,
        "project_title": project.project_title,
        "project_description": project.project_description,
        "researcher_name": project.researcher_name,
        "researcher_email": project.researcher_email,
        "researcher_affiliation": project.researcher_affiliation,
        "clinician_name": project.clinician_name,
        "clinician_email": project.clinician_email,
        "mrec_number": project.mrec_number,
        "iexplore_id": project.iexplore_id,
        "status": project.status,
        "current_step": project.current_step,
        "requested_variables": project.requested_variables,
        "date_range_from": project.date_range_from.isoformat() if project.date_range_from else None,
        "date_range_to": project.date_range_to.isoformat() if project.date_range_to else None,
        "filters": project.filters,
        "project_start_date": project.project_start_date.isoformat() if project.project_start_date else None,
        "project_end_date": project.project_end_date.isoformat() if project.project_end_date else None,
        "access_token": project.access_token if project.status == 'approved' else None,
        "token_expires_at": project.token_expires_at.isoformat() if project.token_expires_at else None,
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "submitted_at": project.submitted_at.isoformat() if project.submitted_at else None,
        "approved_at": project.approved_at.isoformat() if project.approved_at else None
    }


@router.patch("/{project_id}")
def update_project(
    project_id: str,
    project_data: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update project (only in draft status)"""
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid project ID")
    
    project = db.query(ResearchProject).filter(ResearchProject.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    
    if project.status != 'draft':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot update project in {project.status} status"
        )
    
    if not check_project_permission(project, current_user, db):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    
    # Update fields
    if project_data.project_title is not None:
        project.project_title = project_data.project_title
    if project_data.project_description is not None:
        project.project_description = project_data.project_description
    if project_data.clinician_name is not None:
        project.clinician_name = project_data.clinician_name
    if project_data.clinician_email is not None:
        project.clinician_email = project_data.clinician_email
    if project_data.mrec_number is not None:
        project.mrec_number = project_data.mrec_number
    if project_data.requested_variables is not None:
        project.requested_variables = project_data.requested_variables
    if project_data.date_range_from is not None:
        project.date_range_from = project_data.date_range_from
    if project_data.date_range_to is not None:
        project.date_range_to = project_data.date_range_to
    if project_data.filters is not None:
        project.filters = project_data.filters
    
    db.commit()
    db.refresh(project)
    
    return {
        "project_id": str(project.id),
        "project_code": project.project_code,
        "status": project.status,
        "message": "Project updated successfully"
    }


# ============================================================================
# WORKFLOW OPERATIONS
# ============================================================================

@router.post("/{project_id}/submit")
def submit_project(
    project_id: str,
    submit_data: ProjectSubmit,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Submit project for ethics review"""
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid project ID")
    
    project = db.query(ResearchProject).filter(ResearchProject.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    
    if project.status != 'draft':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Project is already {project.status}"
        )
    
    if not check_project_permission(project, current_user, db):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    
    # Update status
    project.status = 'submitted'
    project.current_step = 'ethics_review'
    project.submitted_at = datetime.now()
    if project.workflow_state:
        project.workflow_state["current_step"] = "ethics_review"
        project.workflow_state["history"].append({
            "from": "draft",
            "to": "submitted",
            "timestamp": datetime.now().isoformat(),
            "changed_by": str(current_user.id)
        })
    
    db.commit()
    
    return {
        "project_id": str(project.id),
        "project_code": project.project_code,
        "status": project.status,
        "current_step": project.current_step,
        "message": "Project submitted for ethics review"
    }


@router.post("/{project_id}/approve")
def approve_project_step(
    project_id: str,
    approval_data: ProjectApprove,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Approve project at workflow step (admin only)"""
    # Check admin role
    user_roles = [role.slug for role in current_user.roles]
    if 'super_admin' not in user_roles and 'ummc_admin' not in user_roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid project ID")
    
    project = db.query(ResearchProject).filter(ResearchProject.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    
    step = approval_data.step.lower()
    
    if step == 'ethics_review':
        if project.status != 'submitted' and project.status != 'ethics_review':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot approve ethics review for project in {project.status} status"
            )
        project.status = 'ethics_review'
        project.current_step = 'steering_review'
        project.ethics_approved_at = datetime.now()
        project.ethics_approved_by = current_user.id
        
    elif step == 'steering_review':
        if project.status != 'ethics_review':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot approve steering review for project in {project.status} status"
            )
        project.status = 'steering_review'
        project.current_step = 'final_approval'
        project.steering_approved_at = datetime.now()
        project.steering_approved_by = current_user.id
        
    elif step == 'final':
        if project.status != 'steering_review':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot final approve project in {project.status} status"
            )
        # Generate access token
        access_token = generate_access_token()
        token_expires_at = datetime.now() + timedelta(days=365)  # 1 year default
        
        project.status = 'approved'
        project.current_step = 'approved'
        project.approved_at = datetime.now()
        project.approved_by = current_user.id
        project.access_token = access_token
        project.token_issued_at = datetime.now()
        project.token_expires_at = token_expires_at
        project.access_granted_at = datetime.now()
        
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid step: {step}. Must be 'ethics_review', 'steering_review', or 'final'"
        )
    
    # Update workflow state
    if project.workflow_state:
        old_status = project.workflow_state.get("current_step", project.status)
        project.workflow_state["current_step"] = project.current_step
        project.workflow_state["history"].append({
            "from": old_status,
            "to": project.status,
            "timestamp": datetime.now().isoformat(),
            "changed_by": str(current_user.id),
            "notes": approval_data.notes
        })
    
    db.commit()
    db.refresh(project)
    
    response = {
        "project_id": str(project.id),
        "project_code": project.project_code,
        "status": project.status,
        "current_step": project.current_step,
        "message": f"Project approved at {step} step"
    }
    
    if project.status == 'approved':
        response["access_token"] = project.access_token
        response["token_expires_at"] = project.token_expires_at.isoformat()
        response["message"] = "Project fully approved. Access token generated."
    
    return response


@router.post("/{project_id}/reject")
def reject_project(
    project_id: str,
    rejection_data: ProjectReject,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Reject project (admin only)"""
    # Check admin role
    user_roles = [role.slug for role in current_user.roles]
    if 'super_admin' not in user_roles and 'ummc_admin' not in user_roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid project ID")
    
    project = db.query(ResearchProject).filter(ResearchProject.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    
    if project.status in ['rejected', 'archived']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Project is already {project.status}"
        )
    
    project.status = 'rejected'
    project.current_step = 'rejected'
    project.rejected_at = datetime.now()
    project.rejected_by = current_user.id
    project.rejection_reason = rejection_data.reason
    
    # Update workflow state
    if project.workflow_state:
        old_status = project.workflow_state.get("current_step", project.status)
        project.workflow_state["current_step"] = "rejected"
        project.workflow_state["history"].append({
            "from": old_status,
            "to": "rejected",
            "timestamp": datetime.now().isoformat(),
            "changed_by": str(current_user.id),
            "reason": rejection_data.reason
        })
    
    db.commit()
    
    return {
        "project_id": str(project.id),
        "project_code": project.project_code,
        "status": project.status,
        "rejection_reason": project.rejection_reason,
        "message": "Project rejected"
    }


# ============================================================================
# PROJECT LISTING
# ============================================================================

@router.get("/")
def list_projects(
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List projects (user sees own, admin sees all)"""
    user_roles = [role.slug for role in current_user.roles]
    is_admin = 'super_admin' in user_roles or 'ummc_admin' in user_roles
    
    query = db.query(ResearchProject)
    
    # Filter by user if not admin
    if not is_admin:
        query = query.filter(
            or_(
                ResearchProject.created_by == current_user.id,
                ResearchProject.researcher_email == current_user.email
            )
        )
    
    # Filter by status
    if status_filter:
        query = query.filter(ResearchProject.status == status_filter)
    
    # Order by created_at desc
    query = query.order_by(ResearchProject.created_at.desc())
    
    total = query.count()
    projects = query.offset(skip).limit(limit).all()
    
    return {
        "projects": [
            {
                "project_id": str(p.id),
                "project_code": p.project_code,
                "project_title": p.project_title,
                "researcher_name": p.researcher_name,
                "researcher_email": p.researcher_email,
                "status": p.status,
                "current_step": p.current_step,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "submitted_at": p.submitted_at.isoformat() if p.submitted_at else None
            }
            for p in projects
        ],
        "total": total,
        "skip": skip,
        "limit": limit
    }


# ============================================================================
# EXTENSION REQUESTS
# ============================================================================

@router.post("/{project_id}/extend")
def request_extension(
    project_id: str,
    extension_data: ExtensionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Request project extension"""
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid project ID")
    
    project = db.query(ResearchProject).filter(ResearchProject.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    
    if project.status != 'approved':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only request extension for approved projects"
        )
    
    if not check_project_permission(project, current_user, db):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    
    # Calculate new end date
    if project.project_end_date:
        from datetime import timedelta
        requested_new_end_date = project.project_end_date + timedelta(days=extension_data.requested_extension_days)
    else:
        requested_new_end_date = date.today() + timedelta(days=extension_data.requested_extension_days)
    
    # Create extension request
    extension_request = ProjectExtensionRequest(
        project_id=project.id,
        requested_extension_days=extension_data.requested_extension_days,
        requested_new_end_date=requested_new_end_date,
        extension_reason=extension_data.extension_reason,
        additional_justification=extension_data.additional_justification,
        status='pending',
        created_by=current_user.id
    )
    
    project.extension_requested = True
    project.extension_reason = extension_data.extension_reason
    
    db.add(extension_request)
    db.commit()
    db.refresh(extension_request)
    
    return {
        "extension_request_id": str(extension_request.id),
        "project_id": str(project.id),
        "requested_extension_days": extension_request.requested_extension_days,
        "requested_new_end_date": extension_request.requested_new_end_date.isoformat(),
        "status": extension_request.status,
        "message": "Extension request submitted for review"
    }

