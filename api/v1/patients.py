"""
PostgreSQL-Based Patient API (replaces S3/Athena)
Direct database storage for patient data.

Endpoints:
- POST /patients/           - Create patient
- POST /patients/bulk-upload - Bulk upload CSV
- GET  /patients/statistics  - Patient statistics
- GET  /patients/public-analytics - Public analytics (no auth)
- GET  /patients/raw         - List patients (with PII)
- GET  /patients/            - List patients
- GET  /patients/all         - Get all patients
- GET  /patients/all/export  - Export CSV
- GET  /patients/{id}        - Get patient
- PUT  /patients/{id}        - Update patient
- POST /patients/{id}/followup - Add followup
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import or_, func, extract, case, cast, String
from datetime import date, datetime
import csv
import io
import uuid
import logging
import json

from core.deps import permission_required, get_current_user, get_db
from db.models.users import User
from db.models.registry import Patient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patients", tags=["patients"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PatientCreate(BaseModel):
    patient_id: Optional[str] = None
    patient_name: str
    diagnosis_date: date
    icd11_main_code: str
    gender: Optional[str] = None
    date_of_birth: Optional[date] = None
    nationality: Optional[str] = None
    address: Optional[dict] = None
    icd11_description: Optional[str] = None
    icd11_composite_expression: Optional[str] = None
    icd11_manifestation_code: Optional[str] = None
    manifestation: Optional[str] = None
    icd11_topography_code: Optional[str] = None
    icd11_topography: Optional[str] = None
    icd11_morphology_code: Optional[str] = None
    icd11_morphology: Optional[str] = None
    icd11_behavior_code: Optional[str] = None
    icd11_stage_code: Optional[str] = None
    laterality: Optional[str] = None
    t_category: Optional[str] = None
    n_category: Optional[str] = None
    m_category: Optional[str] = None
    multiple_primary_flag: Optional[bool] = None
    basis_of_diagnosis: Optional[str] = None
    primary_site_confirmed: Optional[bool] = None
    surgery_done: Optional[bool] = None
    surgery_date: Optional[date] = None
    chemotherapy_done: Optional[bool] = None
    chemo_start_date: Optional[date] = None
    radiotherapy_done: Optional[bool] = None
    hormonal_therapy: Optional[bool] = None
    immunotherapy: Optional[bool] = None
    treatment_intent: Optional[str] = None
    treatment_notes: Optional[str] = None
    followup_date: Optional[date] = None
    vital_status: Optional[str] = None
    cause_of_death_icd11: Optional[str] = None
    recurrence: Optional[bool] = None
    recurrence_date: Optional[date] = None
    metastasis: Optional[bool] = None
    survival_months: Optional[int] = None
    followup_notes: Optional[str] = None
    data_source: Optional[str] = None
    entry_mode: Optional[str] = None
    entered_by: Optional[str] = None
    age_at_diagnosis: Optional[int] = None
    validation_status: Optional[str] = "Pending"


class PatientUpdate(BaseModel):
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    gender: Optional[str] = None
    date_of_birth: Optional[date] = None
    nationality: Optional[str] = None
    address: Optional[dict] = None
    diagnosis_date: Optional[date] = None
    age_at_diagnosis: Optional[int] = None
    icd11_main_code: Optional[str] = None
    icd11_description: Optional[str] = None
    icd11_composite_expression: Optional[str] = None
    icd11_manifestation_code: Optional[str] = None
    manifestation: Optional[str] = None
    icd11_topography_code: Optional[str] = None
    icd11_topography: Optional[str] = None
    icd11_morphology_code: Optional[str] = None
    icd11_morphology: Optional[str] = None
    icd11_behavior_code: Optional[str] = None
    icd11_stage_code: Optional[str] = None
    laterality: Optional[str] = None
    t_category: Optional[str] = None
    n_category: Optional[str] = None
    m_category: Optional[str] = None
    multiple_primary_flag: Optional[bool] = None
    basis_of_diagnosis: Optional[str] = None
    primary_site_confirmed: Optional[bool] = None
    surgery_done: Optional[bool] = None
    surgery_date: Optional[date] = None
    chemotherapy_done: Optional[bool] = None
    chemo_start_date: Optional[date] = None
    radiotherapy_done: Optional[bool] = None
    hormonal_therapy: Optional[bool] = None
    immunotherapy: Optional[bool] = None
    treatment_intent: Optional[str] = None
    treatment_notes: Optional[str] = None
    followup_date: Optional[date] = None
    vital_status: Optional[str] = None
    cause_of_death_icd11: Optional[str] = None
    recurrence: Optional[bool] = None
    recurrence_date: Optional[date] = None
    metastasis: Optional[bool] = None
    survival_months: Optional[int] = None
    followup_notes: Optional[str] = None
    validation_status: Optional[str] = None


class FollowupUpdate(BaseModel):
    followup_date: Optional[date] = None
    vital_status: Optional[str] = None
    cause_of_death_icd11: Optional[str] = None
    recurrence: Optional[bool] = None
    recurrence_date: Optional[date] = None
    metastasis: Optional[bool] = None
    survival_months: Optional[int] = None
    followup_notes: Optional[str] = None


# ============================================================================
# HELPERS
# ============================================================================

def patient_to_dict(p: Patient) -> dict:
    """Convert Patient ORM object to dict."""
    return {
        "id": str(p.id),
        "tenant_id": str(p.tenant_id) if p.tenant_id else None,
        "organization_id": str(p.organization_id) if p.organization_id else None,
        "patient_id": p.patient_id,
        "patient_name": p.patient_name,
        "gender": p.gender,
        "date_of_birth": p.date_of_birth.isoformat() if p.date_of_birth else None,
        "nationality": p.nationality,
        "address": p.address,
        "diagnosis_date": p.diagnosis_date.isoformat() if p.diagnosis_date else None,
        "age_at_diagnosis": p.age_at_diagnosis,
        "icd11_main_code": p.icd11_main_code,
        "icd11_description": p.icd11_description,
        "icd11_composite_expression": p.icd11_composite_expression,
        "icd11_manifestation_code": p.icd11_manifestation_code,
        "manifestation": p.manifestation,
        "icd11_topography_code": p.icd11_topography_code,
        "icd11_topography": p.icd11_topography,
        "icd11_morphology_code": p.icd11_morphology_code,
        "icd11_morphology": p.icd11_morphology,
        "icd11_behavior_code": p.icd11_behavior_code,
        "icd11_stage_code": p.icd11_stage_code,
        "laterality": p.laterality,
        "t_category": p.t_category,
        "n_category": p.n_category,
        "m_category": p.m_category,
        "multiple_primary_flag": p.multiple_primary_flag,
        "basis_of_diagnosis": p.basis_of_diagnosis,
        "primary_site_confirmed": p.primary_site_confirmed,
        "surgery_done": p.surgery_done,
        "surgery_date": p.surgery_date.isoformat() if p.surgery_date else None,
        "chemotherapy_done": p.chemotherapy_done,
        "chemo_start_date": p.chemo_start_date.isoformat() if p.chemo_start_date else None,
        "radiotherapy_done": p.radiotherapy_done,
        "hormonal_therapy": p.hormonal_therapy,
        "immunotherapy": p.immunotherapy,
        "treatment_intent": p.treatment_intent,
        "treatment_notes": p.treatment_notes,
        "followup_date": p.followup_date.isoformat() if p.followup_date else None,
        "vital_status": p.vital_status,
        "cause_of_death_icd11": p.cause_of_death_icd11,
        "recurrence": p.recurrence,
        "recurrence_date": p.recurrence_date.isoformat() if p.recurrence_date else None,
        "metastasis": p.metastasis,
        "survival_months": p.survival_months,
        "followup_notes": p.followup_notes,
        "data_source": p.data_source,
        "entry_mode": p.entry_mode,
        "entered_by": p.entered_by,
        "validation_status": p.validation_status,
        "is_active": p.is_active,
        "entry_timestamp": p.entry_timestamp.isoformat() if p.entry_timestamp else None,
        "last_modified": p.last_modified.isoformat() if p.last_modified else None,
    }


def apply_tenant_filter(query, current_user: User):
    """Apply tenant/organization filter based on user's scope."""
    if any(r.slug == "super_admin" for r in current_user.roles):
        return query  # Super admin sees all
    if current_user.organization_id:
        return query.filter(Patient.organization_id == current_user.organization_id)
    if current_user.tenant_id:
        return query.filter(Patient.tenant_id == current_user.tenant_id)
    return query


# ============================================================================
# CREATE ENDPOINTS
# ============================================================================

@router.post("/", dependencies=[Depends(permission_required("patients.create"))])
def create_patient(
    payload: PatientCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new patient record in PostgreSQL."""
    try:
        data = payload.model_dump(exclude_unset=True)

        # Calculate age at diagnosis if not provided
        if not data.get("age_at_diagnosis") and data.get("date_of_birth"):
            dob = data["date_of_birth"]
            diag = data["diagnosis_date"]
            age = diag.year - dob.year - ((diag.month, diag.day) < (dob.month, dob.day))
            data["age_at_diagnosis"] = age

        patient = Patient(
            id=uuid.uuid4(),
            tenant_id=current_user.tenant_id,
            organization_id=current_user.organization_id,
            **data,
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

        return {
            "id": str(patient.id),
            "patient_id": patient.patient_id,
            "message": "Patient created successfully",
            "storage": "PostgreSQL",
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error creating patient: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create patient: {e}")


@router.post("/bulk-upload", dependencies=[Depends(permission_required("patients.create"))])
async def bulk_upload_patients(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Bulk upload patients from CSV file."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV file")

    try:
        contents = await file.read()
        csv_data = contents.decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_data))

        successful = 0
        failed = 0
        failed_records = []

        def parse_date(s):
            if not s:
                return None
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
                try:
                    return datetime.strptime(s.strip(), fmt).date()
                except ValueError:
                    continue
            return None

        def parse_bool(s):
            return s.strip().lower() in ("true", "1", "yes") if s else None

        for idx, row in enumerate(reader, 1):
            if not row.get("patient_name") or not row.get("diagnosis_date") or not row.get("icd11_main_code"):
                failed += 1
                failed_records.append({"row": idx, "error": "Missing required fields"})
                continue
            try:
                p = Patient(
                    id=uuid.uuid4(),
                    tenant_id=current_user.tenant_id,
                    organization_id=current_user.organization_id,
                    patient_id=row.get("patient_id") or None,
                    patient_name=row["patient_name"],
                    diagnosis_date=parse_date(row["diagnosis_date"]),
                    icd11_main_code=row["icd11_main_code"],
                    gender=row.get("gender"),
                    date_of_birth=parse_date(row.get("date_of_birth")),
                    age_at_diagnosis=int(row["age_at_diagnosis"]) if row.get("age_at_diagnosis") else None,
                    nationality=row.get("nationality"),
                    icd11_description=row.get("icd11_description"),
                    icd11_composite_expression=row.get("icd11_composite_expression"),
                    icd11_topography_code=row.get("icd11_topography_code"),
                    icd11_topography=row.get("icd11_topography"),
                    icd11_morphology_code=row.get("icd11_morphology_code"),
                    icd11_morphology=row.get("icd11_morphology"),
                    icd11_behavior_code=row.get("icd11_behavior_code"),
                    icd11_stage_code=row.get("icd11_stage_code"),
                    icd11_manifestation_code=row.get("icd11_manifestation_code"),
                    manifestation=row.get("manifestation"),
                    laterality=row.get("laterality"),
                    t_category=row.get("t_category"),
                    n_category=row.get("n_category"),
                    m_category=row.get("m_category"),
                    multiple_primary_flag=parse_bool(row.get("multiple_primary_flag")),
                    basis_of_diagnosis=row.get("basis_of_diagnosis"),
                    surgery_done=parse_bool(row.get("surgery_done")),
                    surgery_date=parse_date(row.get("surgery_date")),
                    chemotherapy_done=parse_bool(row.get("chemotherapy_done")),
                    chemo_start_date=parse_date(row.get("chemo_start_date")),
                    radiotherapy_done=parse_bool(row.get("radiotherapy_done")),
                    hormonal_therapy=parse_bool(row.get("hormonal_therapy")),
                    immunotherapy=parse_bool(row.get("immunotherapy")),
                    treatment_intent=row.get("treatment_intent"),
                    treatment_notes=row.get("treatment_notes"),
                    followup_date=parse_date(row.get("followup_date")),
                    vital_status=row.get("vital_status"),
                    cause_of_death_icd11=row.get("cause_of_death_icd11"),
                    recurrence=parse_bool(row.get("recurrence")),
                    recurrence_date=parse_date(row.get("recurrence_date")),
                    metastasis=parse_bool(row.get("metastasis")),
                    survival_months=int(row["survival_months"]) if row.get("survival_months") else None,
                    followup_notes=row.get("followup_notes"),
                    data_source=row.get("data_source", "Bulk CSV Upload"),
                    entry_mode="Bulk",
                    validation_status=row.get("validation_status", "Pending"),
                )
                db.add(p)
                successful += 1
            except Exception as e:
                failed += 1
                failed_records.append({"row": idx, "error": str(e)})

        db.commit()
        return {
            "message": "Bulk upload completed",
            "storage": "PostgreSQL",
            "total_rows": successful + failed,
            "successful": successful,
            "failed": failed,
            "successful_imports": successful,
            "failed_imports": failed_records,
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Bulk upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Bulk upload failed: {e}")


# ============================================================================
# STATISTICS (must be before /{patient_id} to avoid route conflict)
# ============================================================================

@router.get("/statistics")
def get_patient_statistics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get patient statistics."""
    try:
        q = db.query(Patient).filter(Patient.is_active == True)
        q = apply_tenant_filter(q, current_user)

        total = q.count()
        gender_dist = dict(
            db.query(Patient.gender, func.count())
            .filter(Patient.is_active == True, Patient.gender.isnot(None))
            .group_by(Patient.gender)
            .all()
        )
        top_cancers = (
            db.query(Patient.icd11_main_code, Patient.icd11_description, func.count().label("count"))
            .filter(Patient.is_active == True, Patient.icd11_main_code.isnot(None))
            .group_by(Patient.icd11_main_code, Patient.icd11_description)
            .order_by(func.count().desc())
            .limit(10)
            .all()
        )

        return {
            "total_patients": total,
            "gender_distribution": gender_dist,
            "top_cancers": [
                {"code": c[0], "description": c[1] or c[0], "count": c[2]}
                for c in top_cancers
            ],
        }
    except Exception as e:
        logger.error(f"Statistics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/public-analytics")
def get_public_analytics(db: Session = Depends(get_db)):
    """Public analytics (no auth required). Returns only aggregate statistics."""
    try:
        total = db.query(func.count(Patient.id)).filter(Patient.is_active == True).scalar()

        # Gender distribution
        gender_rows = (
            db.query(Patient.gender, func.count())
            .filter(Patient.is_active == True, Patient.gender.isnot(None))
            .group_by(Patient.gender).all()
        )
        by_gender = [{"gender": g, "patient_count": c} for g, c in gender_rows]

        # Top cancer types
        cancer_rows = (
            db.query(Patient.icd11_main_code, Patient.icd11_description, func.count().label("cnt"))
            .filter(Patient.is_active == True, Patient.icd11_main_code.isnot(None))
            .group_by(Patient.icd11_main_code, Patient.icd11_description)
            .order_by(func.count().desc()).limit(10).all()
        )
        by_cancer_type = [
            {"icd11_main_code": r[0], "description": r[1] or r[0], "patient_count": r[2]}
            for r in cancer_rows
        ]

        # Year distribution
        year_rows = (
            db.query(extract("year", Patient.diagnosis_date).label("year"), func.count())
            .filter(Patient.is_active == True, Patient.diagnosis_date.isnot(None))
            .group_by("year").order_by("year").all()
        )
        by_year = [{"year": int(y), "patient_count": c} for y, c in year_rows]

        # Age group distribution
        age_case = case(
            (Patient.age_at_diagnosis < 20, "0-19"),
            (Patient.age_at_diagnosis < 40, "20-39"),
            (Patient.age_at_diagnosis < 60, "40-59"),
            (Patient.age_at_diagnosis < 80, "60-79"),
            else_="80+",
        )
        age_rows = (
            db.query(age_case.label("age_group"), func.count())
            .filter(Patient.is_active == True, Patient.age_at_diagnosis.isnot(None))
            .group_by("age_group").order_by("age_group").all()
        )
        by_age_group = [{"age_group": a, "patient_count": c} for a, c in age_rows]

        return {
            "summary": {"total_anonymized_patients": total, "data_available": total > 0},
            "statistics": {
                "by_cancer_type": by_cancer_type,
                "by_year": by_year,
                "by_age_group": by_age_group,
                "by_gender": by_gender,
            },
        }
    except Exception as e:
        logger.error(f"Public analytics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# READ ENDPOINTS
# ============================================================================

@router.get("/raw", dependencies=[Depends(permission_required("patients.read"))])
def list_raw_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List patients with full PII from PostgreSQL."""
    try:
        q = db.query(Patient).filter(Patient.is_active == True)
        q = apply_tenant_filter(q, current_user)

        if search:
            term = f"%{search}%"
            q = q.filter(
                or_(
                    Patient.patient_name.ilike(term),
                    Patient.patient_id.ilike(term),
                    cast(Patient.id, String).ilike(term),
                    Patient.icd11_main_code.ilike(term),
                    Patient.icd11_description.ilike(term),
                    Patient.gender.ilike(term),
                    Patient.nationality.ilike(term),
                    Patient.t_category.ilike(term),
                    Patient.n_category.ilike(term),
                    Patient.m_category.ilike(term),
                )
            )

        total = q.count()
        patients = q.order_by(Patient.entry_timestamp.desc()).offset(skip).limit(limit).all()

        return {
            "items": [patient_to_dict(p) for p in patients],
            "total": total,
            "skip": skip,
            "limit": limit,
        }
    except Exception as e:
        logger.error(f"List patients error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", dependencies=[Depends(permission_required("patients.read"))])
def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None),
    validation_status: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List patients with pagination."""
    try:
        q = db.query(Patient).filter(Patient.is_active == True)
        q = apply_tenant_filter(q, current_user)

        if validation_status:
            q = q.filter(Patient.validation_status == validation_status)

        if search:
            term = f"%{search}%"
            q = q.filter(
                or_(
                    Patient.patient_name.ilike(term),
                    Patient.patient_id.ilike(term),
                    Patient.icd11_main_code.ilike(term),
                    Patient.icd11_description.ilike(term),
                )
            )

        total = q.count()
        patients = q.order_by(Patient.entry_timestamp.desc()).offset(skip).limit(limit).all()

        return {
            "patients": [patient_to_dict(p) for p in patients],
            "total": total,
            "skip": skip,
            "limit": limit,
        }
    except Exception as e:
        logger.error(f"List error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all", dependencies=[Depends(permission_required("patients.read"))])
def get_all_patients(
    mode: str = Query("all"),
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all patients."""
    try:
        q = db.query(Patient).filter(Patient.is_active == True)
        q = apply_tenant_filter(q, current_user)

        if status:
            q = q.filter(Patient.validation_status == status)
        if search:
            term = f"%{search}%"
            q = q.filter(
                or_(
                    Patient.patient_name.ilike(term),
                    Patient.patient_id.ilike(term),
                    Patient.icd11_main_code.ilike(term),
                    Patient.icd11_description.ilike(term),
                )
            )

        total = q.count()
        query = q.order_by(Patient.entry_timestamp.desc()).offset(skip)
        if limit:
            query = query.limit(limit)
        patients = query.all()

        if mode == "export":
            return {"patients": [patient_to_dict(p) for p in patients], "total": total}

        return {
            "patients": [patient_to_dict(p) for p in patients],
            "total": total,
            "skip": skip,
            "limit": limit,
        }
    except Exception as e:
        logger.error(f"Get all error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all/export", dependencies=[Depends(permission_required("patients.read"))])
def export_patients_csv(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export patients as CSV download."""
    try:
        q = db.query(Patient).filter(Patient.is_active == True)
        q = apply_tenant_filter(q, current_user)
        patients = q.order_by(Patient.entry_timestamp.desc()).all()

        output = io.StringIO()
        if patients:
            fields = list(patient_to_dict(patients[0]).keys())
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for p in patients:
                writer.writerow(patient_to_dict(p))

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=patients_export.csv"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/excel", dependencies=[Depends(permission_required("patients.read"))])
def export_patients_excel(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export as CSV (Excel-compatible)."""
    return export_patients_csv(current_user=current_user, db=db)


# ============================================================================
# SINGLE PATIENT ENDPOINTS
# ============================================================================

@router.get("/{patient_id}", dependencies=[Depends(permission_required("patients.read"))])
def get_patient(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a single patient by ID (UUID or patient_id)."""
    try:
        # Try UUID first
        try:
            uid = uuid.UUID(patient_id)
            patient = db.query(Patient).filter(Patient.id == uid).first()
        except ValueError:
            patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()

        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        return patient_to_dict(patient)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{patient_id}", dependencies=[Depends(permission_required("patients.update"))])
def update_patient(
    patient_id: str,
    payload: PatientUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update a patient record."""
    try:
        try:
            uid = uuid.UUID(patient_id)
            patient = db.query(Patient).filter(Patient.id == uid).first()
        except ValueError:
            patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()

        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        update_data = payload.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(patient, key, value)

        patient.updated_by = current_user.id
        patient.last_modified = datetime.utcnow()

        db.commit()
        db.refresh(patient)

        return {
            "message": "Patient updated successfully",
            "patient": patient_to_dict(patient),
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{patient_id}/followup", dependencies=[Depends(permission_required("patients.update"))])
def add_followup(
    patient_id: str,
    payload: FollowupUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add followup information to a patient."""
    try:
        try:
            uid = uuid.UUID(patient_id)
            patient = db.query(Patient).filter(Patient.id == uid).first()
        except ValueError:
            patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()

        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        update_data = payload.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(patient, key, value)

        patient.updated_by = current_user.id
        patient.last_modified = datetime.utcnow()

        db.commit()
        db.refresh(patient)

        return {
            "message": "Followup added successfully",
            "patient": patient_to_dict(patient),
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
