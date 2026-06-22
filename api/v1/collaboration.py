"""
Collaboration API — bidirectional data requests between researchers and hospitals.

Either party can initiate. Supports:
- Protocol/request creation and submission
- Document sharing (protocols, ethics letters, consent forms)
- Messaging thread (fully audited)
- Ethics approval flow
- Data sharing confirmation
- Full audit trail per collaboration
"""
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc
import secrets
import uuid
import os
import logging

from core.deps import get_db, get_current_user
from core.rate_limit import limiter
from db.models.users import User
from db.models.collaboration import DataCollaboration, CollabDocument, CollabMessage, CollabAuditEntry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/collaborations", tags=["collaborations"])

UPLOAD_DIR = os.environ.get("COLLAB_UPLOAD_DIR", "/app/uploads/collaborations")
ALLOWED_ROLES = {"researcher", "hospital_admin", "ummc_admin", "super_admin", "registry_editor"}


def _user_role(user: User) -> str:
    slugs = [r.slug for r in user.roles]
    if "researcher" in slugs:
        return "researcher"
    if "hospital_admin" in slugs:
        return "hospital_admin"
    if "super_admin" in slugs or "ummc_admin" in slugs:
        return "admin"
    return slugs[0] if slugs else "unknown"


def _check_access(user: User):
    if not any(r.slug in ALLOWED_ROLES for r in user.roles):
        raise HTTPException(status_code=403, detail="Insufficient role for collaboration")


def _audit(db: Session, collab_id, actor: User, action: str, detail: str = ""):
    db.add(CollabAuditEntry(
        collaboration_id=collab_id, actor_id=actor.id,
        actor_name=actor.full_name or actor.email, action=action, detail=detail,
    ))


def _sys_msg(db: Session, collab_id, text: str):
    db.add(CollabMessage(
        collaboration_id=collab_id, sender_id=None, sender_name="System",
        sender_role="system", message=text, is_system=True,
    ))


def _collab_view(c: DataCollaboration, docs=None, messages=None, audit=None):
    return {
        "id": str(c.id),
        "collab_id": c.collab_id,
        "initiated_by": str(c.initiated_by),
        "initiated_by_role": c.initiated_by_role,
        "target_organization_id": str(c.target_organization_id) if c.target_organization_id else None,
        "title": c.title,
        "purpose": c.purpose,
        "protocol_version": c.protocol_version,
        "data_requirements": c.data_requirements,
        "ethical_justification": c.ethical_justification,
        "estimated_records": c.estimated_records,
        "icd11_codes": c.icd11_codes,
        "year_range": c.year_range,
        "status": c.status,
        "rejection_reason": c.rejection_reason,
        "ethics_approval_ref": c.ethics_approval_ref,
        "ethics_approved_at": c.ethics_approved_at.isoformat() if c.ethics_approved_at else None,
        "data_shared_at": c.data_shared_at.isoformat() if c.data_shared_at else None,
        "data_record_count": c.data_record_count,
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        "documents": docs,
        "messages": messages,
        "audit_trail": audit,
    }


# ---- CREATE ----
class CreateCollabRequest(BaseModel):
    title: str = Field(..., min_length=5)
    purpose: str = Field(..., min_length=10)
    data_requirements: Optional[str] = None
    ethical_justification: Optional[str] = None
    estimated_records: Optional[int] = None
    icd11_codes: Optional[List[str]] = None
    year_range: Optional[dict] = None
    target_organization_id: Optional[str] = None


@router.post("/", status_code=201)
@limiter.limit("20/minute")
def create_collaboration(
    body: CreateCollabRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    request: Request = None,
):
    """Create a new data collaboration request (researcher or hospital can initiate)."""
    _check_access(current_user)
    role = _user_role(current_user)
    ts = datetime.now().strftime("%Y%m%d")
    collab_id = f"COL-NG-{ts}-{secrets.token_hex(3).upper()}"

    c = DataCollaboration(
        collab_id=collab_id,
        initiated_by=current_user.id,
        initiated_by_role=role,
        target_organization_id=body.target_organization_id,
        title=body.title,
        purpose=body.purpose,
        data_requirements=body.data_requirements,
        ethical_justification=body.ethical_justification,
        estimated_records=body.estimated_records,
        icd11_codes=body.icd11_codes,
        year_range=body.year_range,
        status="SUBMITTED",
    )
    db.add(c)
    db.flush()
    _audit(db, c.id, current_user, "created", f"Collaboration initiated by {role}")
    _sys_msg(db, c.id, f"Collaboration request created by {current_user.full_name or current_user.email}.")
    db.commit()
    return {"collab_id": collab_id, "id": str(c.id), "status": "SUBMITTED"}


# ---- LIST ----
@router.get("/")
def list_collaborations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List collaborations visible to the current user."""
    _check_access(current_user)
    role = _user_role(current_user)
    q = db.query(DataCollaboration)
    if role == "researcher":
        q = q.filter(DataCollaboration.initiated_by == current_user.id)
    elif role == "hospital_admin":
        q = q.filter(
            (DataCollaboration.target_organization_id == current_user.organization_id) |
            (DataCollaboration.initiated_by == current_user.id)
        )
    # admin sees all
    rows = q.order_by(desc(DataCollaboration.created_at)).limit(200).all()
    return {"total": len(rows), "collaborations": [_collab_view(r) for r in rows]}


# ---- DETAIL ----
@router.get("/{collab_id}")
def get_collaboration(
    collab_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Full detail with documents, messages, and audit trail."""
    _check_access(current_user)
    c = db.query(DataCollaboration).filter(DataCollaboration.collab_id == collab_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Collaboration not found")

    docs = [{"id": str(d.id), "filename": d.filename, "category": d.document_category,
             "description": d.description, "file_size": d.file_size,
             "uploaded_by": str(d.uploaded_by), "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None}
            for d in db.query(CollabDocument).filter(CollabDocument.collaboration_id == c.id).order_by(CollabDocument.uploaded_at).all()]

    msgs = [{"id": str(m.id), "sender_name": m.sender_name, "sender_role": m.sender_role,
             "message": m.message, "is_system": m.is_system,
             "created_at": m.created_at.isoformat() if m.created_at else None}
            for m in db.query(CollabMessage).filter(CollabMessage.collaboration_id == c.id).order_by(CollabMessage.created_at).all()]

    trail = [{"action": a.action, "actor_name": a.actor_name, "detail": a.detail,
              "created_at": a.created_at.isoformat() if a.created_at else None}
             for a in db.query(CollabAuditEntry).filter(CollabAuditEntry.collaboration_id == c.id).order_by(CollabAuditEntry.created_at).all()]

    return _collab_view(c, docs=docs, messages=msgs, audit=trail)


# ---- MESSAGING ----
class SendMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)


@router.post("/{collab_id}/messages")
def send_message(
    collab_id: str,
    body: SendMessage,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Send a message in the collaboration thread (audited)."""
    _check_access(current_user)
    c = db.query(DataCollaboration).filter(DataCollaboration.collab_id == collab_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Collaboration not found")

    m = CollabMessage(
        collaboration_id=c.id, sender_id=current_user.id,
        sender_name=current_user.full_name or current_user.email,
        sender_role=_user_role(current_user),
        message=body.message,
    )
    db.add(m)
    _audit(db, c.id, current_user, "messaged", body.message[:100])
    db.commit()
    return {"id": str(m.id), "created_at": m.created_at.isoformat() if m.created_at else None}


# ---- DOCUMENTS ----
@router.post("/{collab_id}/documents")
async def upload_document(
    collab_id: str,
    file: UploadFile = File(...),
    category: str = Form("other"),
    description: str = Form(""),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a document (protocol, ethics letter, consent form, etc.)."""
    _check_access(current_user)
    c = db.query(DataCollaboration).filter(DataCollaboration.collab_id == collab_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Collaboration not found")

    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 15 MB limit")

    os.makedirs(os.path.join(UPLOAD_DIR, str(c.id)), exist_ok=True)
    safe = f"{uuid.uuid4().hex[:8]}_{os.path.basename(file.filename or 'doc')}"
    path = os.path.join(UPLOAD_DIR, str(c.id), safe)
    with open(path, "wb") as f:
        f.write(contents)

    doc = CollabDocument(
        collaboration_id=c.id, uploaded_by=current_user.id,
        filename=file.filename or safe, file_path=path,
        file_type=file.content_type, file_size=len(contents),
        document_category=category, description=description,
    )
    db.add(doc)
    _audit(db, c.id, current_user, "document_uploaded", f"{category}: {file.filename}")
    _sys_msg(db, c.id, f"{current_user.full_name or current_user.email} uploaded '{file.filename}' ({category}).")
    db.commit()
    return {"id": str(doc.id), "filename": doc.filename, "category": category}


@router.get("/{collab_id}/documents/{doc_id}/download")
def download_document(
    collab_id: str, doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _check_access(current_user)
    doc = db.query(CollabDocument).filter(CollabDocument.id == doc_id).first()
    if not doc or not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(path=doc.file_path, filename=doc.filename, media_type=doc.file_type or "application/octet-stream")


# ---- STATUS TRANSITIONS ----
class StatusTransition(BaseModel):
    action: str  # review, approve_ethics, share_data, reject, withdraw
    reason: Optional[str] = None
    ethics_reference: Optional[str] = None
    record_count: Optional[int] = None


TRANSITIONS = {
    "review": {"from": ["SUBMITTED"], "to": "UNDER_REVIEW"},
    "request_ethics": {"from": ["UNDER_REVIEW"], "to": "ETHICS_PENDING"},
    "approve_ethics": {"from": ["ETHICS_PENDING"], "to": "ETHICS_APPROVED"},
    "share_data": {"from": ["ETHICS_APPROVED"], "to": "DATA_SHARED"},
    "complete": {"from": ["DATA_SHARED"], "to": "COMPLETED"},
    "reject": {"from": ["SUBMITTED", "UNDER_REVIEW", "ETHICS_PENDING"], "to": "REJECTED"},
    "withdraw": {"from": ["SUBMITTED", "UNDER_REVIEW", "ETHICS_PENDING", "ETHICS_APPROVED"], "to": "WITHDRAWN"},
}


@router.post("/{collab_id}/transition")
def transition_status(
    collab_id: str,
    body: StatusTransition,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Advance or reject a collaboration through its lifecycle."""
    _check_access(current_user)
    c = db.query(DataCollaboration).filter(DataCollaboration.collab_id == collab_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Collaboration not found")

    rule = TRANSITIONS.get(body.action)
    if not rule:
        raise HTTPException(status_code=400, detail=f"Unknown action: {body.action}")
    if c.status not in rule["from"]:
        raise HTTPException(status_code=400, detail=f"Cannot {body.action} from status {c.status}")

    c.status = rule["to"]
    c.updated_at = datetime.now()

    if body.action == "reject":
        c.rejection_reason = body.reason
    if body.action == "approve_ethics":
        c.ethics_approval_ref = body.ethics_reference or f"ETH-COL-{secrets.token_hex(3).upper()}"
        c.ethics_approved_at = datetime.now()
        c.ethics_approved_by = current_user.id
    if body.action == "share_data":
        c.data_shared_at = datetime.now()
        c.data_record_count = body.record_count

    _audit(db, c.id, current_user, body.action, body.reason or "")
    _sys_msg(db, c.id, f"Status changed to {rule['to']} by {current_user.full_name or current_user.email}.")
    db.commit()
    return {"collab_id": c.collab_id, "status": c.status}
