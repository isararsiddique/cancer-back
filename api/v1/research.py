from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta, date, timezone
from pydantic import BaseModel
import secrets
import csv
import io
import uuid

from core.deps import get_db, get_current_user
from db.models.research import ResearchRequest
from db.models.users import User

router = APIRouter(prefix="/research", tags=["research"])


# ============================================================================
# RESEARCHER SIGNUP
# ============================================================================

class ResearcherSignup(BaseModel):
    """Model for researcher registration"""
    email: str
    full_name: str
    password: str
    affiliation: str
    research_interests: Optional[str] = None


@router.post("/signup")
def researcher_signup(
    signup_data: ResearcherSignup,
    db: Session = Depends(get_db)
):
    """
    Allow researchers to sign up for data access.
    Creates a user account with researcher role and assigns to "researcher" tenant.
    """
    from core.security import get_password_hash
    from db.models.users import User
    from db.models.rbac import Role
    from db.models.core import Tenant
    
    try:
        # Check if email already exists
        existing = db.query(User).filter(User.email == signup_data.email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Get or create "researcher" tenant
        researcher_tenant = db.query(Tenant).filter(Tenant.name == "researcher").first()
        if not researcher_tenant:
            # Create default researcher tenant
            researcher_tenant = Tenant(
                name="researcher",
                meta={
                    "description": "Default tenant for researchers",
                    "auto_created": True
                }
            )
            db.add(researcher_tenant)
            db.flush()  # Flush to get the ID without committing
        
        # Get researcher role
        researcher_role = db.query(Role).filter(Role.slug == "researcher").first()
        if not researcher_role:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Researcher role not configured. Please contact administrator."
            )
        
        # Create user with tenant assignment
        user = User(
            email=signup_data.email,
            full_name=signup_data.full_name,
            hashed_password=get_password_hash(signup_data.password),
            tenant_id=researcher_tenant.id,  # Assign to researcher tenant
            is_active=True,
            is_email_verified=False,  # Require email verification in production
            meta={
                "affiliation": signup_data.affiliation,
                "research_interests": signup_data.research_interests,
                "user_type": "researcher"
            }
        )
        
        # Assign researcher role
        user.roles.append(researcher_role)
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return {
            "message": "Researcher account created successfully",
            "user_id": str(user.id),
            "email": user.email,
            "tenant_id": str(user.tenant_id),
            "tenant_name": researcher_tenant.name,
            "note": "You can now login and request research data"
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create researcher account: {str(e)}"
        )


# ============================================================================
# DATA STATISTICS & PREVIEW (For Researchers)
# ============================================================================

@router.get("/data/statistics")
def get_data_statistics(
    db: Session = Depends(get_db)
):
    """
    Get aggregated statistics about available anonymized data.
    Shows counts by cancer type, year, age - no individual patient data.
    Public access allowed - only returns aggregated counts, no sensitive data.
    """
    # Allow public access - this endpoint only returns aggregated statistics
    # No individual patient data is exposed
    
    stats = {}
    
    # Total count
    result = db.execute(text("SELECT COUNT(*) FROM registry.patients_anonymized"))
    stats["total_patients"] = result.scalar()
    
    # By Cancer Type (ICD-11)
    result = db.execute(text("""
        SELECT 
            LEFT(icd11_main_code, 3) as cancer_type,
            icd11_main_code,
            icd11_description,
            COUNT(*) as patient_count
        FROM registry.patients_anonymized
        WHERE icd11_main_code IS NOT NULL
        GROUP BY LEFT(icd11_main_code, 3), icd11_main_code, icd11_description
        ORDER BY patient_count DESC
    """))
    stats["by_cancer_type"] = [
        {
            "cancer_type_code": row[0],
            "icd11_main_code": row[1],
            "description": row[2],
            "patient_count": row[3]
        }
        for row in result.fetchall()
    ]
    
    # Year-wise distribution
    result = db.execute(text("""
        SELECT 
            diagnosis_year,
            COUNT(*) as patient_count
        FROM registry.patients_anonymized
        WHERE diagnosis_year IS NOT NULL
        GROUP BY diagnosis_year
        ORDER BY diagnosis_year DESC
    """))
    stats["by_year"] = [
        {
            "year": row[0],
            "patient_count": row[1]
        }
        for row in result.fetchall()
    ]
    
    # Age distribution
    result = db.execute(text("""
        SELECT 
            CASE 
                WHEN age_at_diagnosis < 20 THEN '<20'
                WHEN age_at_diagnosis < 30 THEN '20-29'
                WHEN age_at_diagnosis < 40 THEN '30-39'
                WHEN age_at_diagnosis < 50 THEN '40-49'
                WHEN age_at_diagnosis < 60 THEN '50-59'
                WHEN age_at_diagnosis < 70 THEN '60-69'
                WHEN age_at_diagnosis < 80 THEN '70-79'
                ELSE '80+'
            END as age_group,
            COUNT(*) as patient_count
        FROM registry.patients_anonymized
        WHERE age_at_diagnosis IS NOT NULL
        GROUP BY 
            CASE 
                WHEN age_at_diagnosis < 20 THEN '<20'
                WHEN age_at_diagnosis < 30 THEN '20-29'
                WHEN age_at_diagnosis < 40 THEN '30-39'
                WHEN age_at_diagnosis < 50 THEN '40-49'
                WHEN age_at_diagnosis < 60 THEN '50-59'
                WHEN age_at_diagnosis < 70 THEN '60-69'
                WHEN age_at_diagnosis < 80 THEN '70-79'
                ELSE '80+'
            END
        ORDER BY MIN(age_at_diagnosis)
    """))
    stats["by_age_group"] = [
        {
            "age_group": row[0],
            "patient_count": row[1]
        }
        for row in result.fetchall()
    ]
    
    # Gender distribution
    result = db.execute(text("""
        SELECT 
            gender,
            COUNT(*) as patient_count
        FROM registry.patients_anonymized
        WHERE gender IS NOT NULL
        GROUP BY gender
        ORDER BY patient_count DESC
    """))
    stats["by_gender"] = [
        {
            "gender": row[0],
            "patient_count": row[1]
        }
        for row in result.fetchall()
    ]
    
    # TNM Stage distribution
    result = db.execute(text("""
        SELECT 
            t_category,
            n_category,
            m_category,
            COUNT(*) as patient_count
        FROM registry.patients_anonymized
        WHERE t_category IS NOT NULL 
        AND n_category IS NOT NULL 
        AND m_category IS NOT NULL
        GROUP BY t_category, n_category, m_category
        ORDER BY patient_count DESC
        LIMIT 20
    """))
    stats["by_tnm_stage"] = [
        {
            "t_category": row[0],
            "n_category": row[1],
            "m_category": row[2],
            "patient_count": row[3]
        }
        for row in result.fetchall()
    ]
    
    # Treatment statistics
    result = db.execute(text("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN surgery_done THEN 1 ELSE 0 END) as surgery_count,
            SUM(CASE WHEN chemotherapy_done THEN 1 ELSE 0 END) as chemo_count,
            SUM(CASE WHEN radiotherapy_done THEN 1 ELSE 0 END) as radio_count,
            SUM(CASE WHEN hormonal_therapy THEN 1 ELSE 0 END) as hormonal_count,
            SUM(CASE WHEN immunotherapy THEN 1 ELSE 0 END) as immuno_count
        FROM registry.patients_anonymized
    """))
    treatment_row = result.fetchone()
    stats["treatment_statistics"] = {
        "total_patients": treatment_row[0],
        "surgery": treatment_row[1],
        "chemotherapy": treatment_row[2],
        "radiotherapy": treatment_row[3],
        "hormonal_therapy": treatment_row[4],
        "immunotherapy": treatment_row[5]
    }
    
    return {
        "summary": {
            "total_anonymized_patients": stats["total_patients"],
            "data_available": stats["total_patients"] > 0,
            "note": "These are aggregated statistics. Individual patient data requires approval."
        },
        "statistics": stats
    }


@router.get("/data/filter-options")
def get_filter_options(
    db: Session = Depends(get_db)
):
    """
    Get available filter options based on actual data in database.
    This endpoint is public (no auth required) as it only returns metadata about what data exists.
    """
    try:
        # Get unique cancer types with counts
        result = db.execute(text("""
            SELECT 
                icd11_main_code,
                icd11_description,
                COUNT(*) as patient_count
            FROM registry.patients_anonymized
            WHERE icd11_main_code IS NOT NULL
            GROUP BY icd11_main_code, icd11_description
            ORDER BY patient_count DESC, icd11_description ASC
        """))
        cancer_types = [
            {
                "icd11_code": row[0],
                "description": row[1] or "Unknown",
                "patient_count": row[2]
            }
            for row in result.fetchall()
        ]
        
        # Get available years with counts
        result = db.execute(text("""
            SELECT 
                diagnosis_year,
                COUNT(*) as patient_count
            FROM registry.patients_anonymized
            WHERE diagnosis_year IS NOT NULL
            GROUP BY diagnosis_year
            ORDER BY diagnosis_year DESC
        """))
        years = [
            {
                "year": row[0],
                "patient_count": row[1]
            }
            for row in result.fetchall()
        ]
        
        return {
            "cancer_types": cancer_types,
            "years": years,
            "total_patients": sum(ct["patient_count"] for ct in cancer_types),
            "note": "These are the available filter options based on current data in the database"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get filter options: {str(e)}"
        )


@router.get("/data/preview")
def preview_available_data(
    cancer_type: Optional[str] = Query(None, description="Filter by cancer type (ICD-11 code)"),
    year: Optional[int] = Query(None, description="Filter by diagnosis year"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Preview available data with filters.
    Shows counts only, not individual records.
    """
    # Check researcher access
    roles = [r.slug for r in current_user.roles]
    if 'researcher' not in roles and 'super_admin' not in roles and 'ummc_admin' not in roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Researcher access required"
        )
    
    conditions = []
    params = {}
    
    if cancer_type:
        conditions.append("icd11_main_code LIKE :cancer_type")
        params['cancer_type'] = f"{cancer_type}%"
    
    if year:
        conditions.append("diagnosis_year = :year")
        params['year'] = year
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    # Get count
    query = f"SELECT COUNT(*) FROM registry.patients_anonymized WHERE {where_clause}"
    result = db.execute(text(query), params)
    count = result.scalar()
    
    # Get breakdown by cancer type if not filtered
    breakdown = []
    if not cancer_type:
        query = f"""
            SELECT 
                icd11_main_code,
                icd11_description,
                COUNT(*) as count
            FROM registry.patients_anonymized
            WHERE {where_clause}
            GROUP BY icd11_main_code, icd11_description
            ORDER BY count DESC
        """
        result = db.execute(text(query), params)
        breakdown = [
            {
                "icd11_code": row[0],
                "description": row[1],
                "patient_count": row[2]
            }
            for row in result.fetchall()
        ]
    
    return {
        "filters_applied": {
            "cancer_type": cancer_type,
            "year": year
        },
        "matching_records": count,
        "breakdown_by_cancer_type": breakdown,
        "note": "This is a preview. To access actual data, create a research request."
    }


# ============================================================================
# RESEARCHER REQUEST PHASE
# ============================================================================

class ResearchRequestCreate(BaseModel):
    """Request model for creating research data requests"""
    researcher_name: str
    researcher_email: str
    researcher_affiliation: Optional[str] = None
    purpose_of_study: str
    manual_record_count: Optional[int] = None  # Allow manual override of estimated count
    
    # Filters
    icd11_main_code: Optional[str] = None
    icd11_description: Optional[str] = None  # Can contain keywords for filtering (e.g., "colon", "breast")
    diagnosis_year_from: Optional[int] = None
    diagnosis_year_to: Optional[int] = None
    age_from: Optional[int] = None
    age_to: Optional[int] = None
    gender: Optional[str] = None
    t_category: Optional[str] = None
    n_category: Optional[str] = None
    m_category: Optional[str] = None
    icd11_morphology_code: Optional[str] = None
    icd11_topography_code: Optional[str] = None
    surgery_done: Optional[bool] = None
    chemotherapy_done: Optional[bool] = None
    radiotherapy_done: Optional[bool] = None
    hormonal_therapy: Optional[bool] = None
    immunotherapy: Optional[bool] = None
    recurrence: Optional[bool] = None
    metastasis: Optional[bool] = None
    vital_status: Optional[str] = None
    treatment_intent: Optional[str] = None


def generate_request_id() -> str:
    """Generate unique request ID: REQ-UMMC-YYYYMMDD-HHMMSS-####"""
    now = datetime.now()
    random_suffix = secrets.token_hex(2).upper()  # 4 hex chars
    return f"REQ-UMMC-{now.strftime('%Y%m%d-%H%M%S')}-{random_suffix}"


def validate_filters(filters: Dict[str, Any], allow_all_data: bool = False) -> tuple:
    """Validate filters and check if dataset is too small"""
    # If ALL_DATA is selected, skip filter validation
    if allow_all_data and filters.get('icd11_main_code') == 'ALL_DATA':
        return True, None
    
    # At least one filter must be provided (unless ALL_DATA)
    if not filters:
        return False, "At least one filter must be provided (e.g., icd11_main_code, diagnosis_year_from, age_from, etc.)"
    
    # Check for keyword-based filtering (in icd11_description)
    has_keywords = bool(filters.get('icd11_description') and filters.get('icd11_description').strip())
    has_icd = bool(filters.get('icd11_main_code') and filters.get('icd11_main_code') != 'ALL_DATA')
    has_year = bool(filters.get('diagnosis_year_from') or filters.get('diagnosis_year_to'))
    has_age = bool(filters.get('age_from') or filters.get('age_to'))
    has_other = bool(filters.get('gender') or filters.get('t_category') or filters.get('surgery_done') is not None)
    
    # Require at least one meaningful filter (keywords OR ICD code OR year range OR age range OR other filters)
    # Exception: ALL_DATA or ALL_CANCER_TYPES don't need additional filters
    if filters.get('icd11_main_code') in ['ALL_DATA', 'ALL_CANCER_TYPES']:
        return True, None
    
    if not (has_keywords or has_icd or has_year or has_age or has_other):
        return False, "Please provide at least one meaningful filter: cancer type keywords, ICD-11 code, diagnosis year range, age range, gender, TNM staging, or treatment filters"
    
    return True, None


@router.post("/estimate-count")
def estimate_count(
    filters: ResearchRequestCreate,
    db: Session = Depends(get_db)
):
    """
    Estimate record count based on filters (for frontend preview).
    This endpoint allows researchers to see estimated counts before submitting.
    """
    try:
        # Check if ALL_DATA or ALL_CANCER_TYPES is selected
        is_all_data = filters.icd11_main_code == 'ALL_DATA'
        is_all_cancer_types = filters.icd11_main_code == 'ALL_CANCER_TYPES'
        
        # Convert request to filters dict
        filter_dict = {
            'icd11_main_code': filters.icd11_main_code,
            'icd11_description': filters.icd11_description,
            'diagnosis_year_from': filters.diagnosis_year_from,
            'diagnosis_year_to': filters.diagnosis_year_to,
            'age_from': filters.age_from,
            'age_to': filters.age_to,
            'gender': filters.gender,
            't_category': filters.t_category,
            'n_category': filters.n_category,
            'm_category': filters.m_category,
            'icd11_morphology_code': filters.icd11_morphology_code,
            'icd11_topography_code': filters.icd11_topography_code,
            'surgery_done': filters.surgery_done,
            'chemotherapy_done': filters.chemotherapy_done,
            'radiotherapy_done': filters.radiotherapy_done,
            'hormonal_therapy': filters.hormonal_therapy,
            'immunotherapy': filters.immunotherapy,
            'recurrence': filters.recurrence,
            'metastasis': filters.metastasis,
            'vital_status': filters.vital_status,
            'treatment_intent': filters.treatment_intent,
        }
        
        # Remove None values and empty strings, but keep ALL_DATA/ALL_CANCER_TYPES markers
        cleaned_filters = {}
        for k, v in filter_dict.items():
            if v is not None and v != "":
                # Keep ALL_DATA and ALL_CANCER_TYPES markers
                if k == 'icd11_main_code' and v in ['ALL_DATA', 'ALL_CANCER_TYPES']:
                    cleaned_filters[k] = v
                elif k != 'icd11_main_code' or v not in ['ALL_DATA', 'ALL_CANCER_TYPES']:
                    cleaned_filters[k] = v
        
        # For ALL_DATA, pass empty filters (will return all records)
        if is_all_data:
            count = estimate_record_count({}, db)
        else:
            # Estimate count with filters
            count = estimate_record_count(cleaned_filters, db)
        
        return {
            "estimated_count": count,
            "filters_applied": filter_dict
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to estimate count: {str(e)}"
        )


def estimate_record_count(filters: Dict[str, Any], db: Session) -> int:
    """Estimate how many records match the filters (from anonymized database)"""
    conditions = []
    params = {}
    
    # Handle ALL_DATA and ALL_CANCER_TYPES - return all records (no ICD filter)
    if filters.get('icd11_main_code') in ['ALL_DATA', 'ALL_CANCER_TYPES']:
        # For ALL_DATA or ALL_CANCER_TYPES, don't filter by ICD code
        # Just apply other filters if any
        pass
    else:
        # Handle keyword-based filtering (search in icd11_description)
        if filters.get('icd11_description') and not filters.get('icd11_main_code'):
            keywords = filters['icd11_description'].strip()
            if keywords:
                # Split keywords by comma or space and search for any match
                keyword_list = [k.strip().lower() for k in keywords.replace(',', ' ').split() if k.strip()]
                if keyword_list:
                    # Build OR conditions for each keyword
                    keyword_conditions = []
                    for i, keyword in enumerate(keyword_list):
                        keyword_conditions.append(f"LOWER(icd11_description) LIKE :keyword_{i}")
                        params[f'keyword_{i}'] = f"%{keyword}%"
                    if keyword_conditions:
                        conditions.append(f"({' OR '.join(keyword_conditions)})")
        
        # Handle ICD-11 code filtering (if provided)
        if filters.get('icd11_main_code'):
            icd11_code = filters['icd11_main_code']
            # Handle "ALL" option or comma-separated multiple codes
            if icd11_code == 'ALL':
                # Don't filter by ICD code - include all
                pass
            elif ',' in icd11_code:
                # Multiple codes - use IN clause with parameterized query
                codes = [c.strip() for c in icd11_code.split(',') if c.strip()]
                if codes:
                    # Build IN clause with placeholders
                    placeholders = ','.join([f':code_{i}' for i in range(len(codes))])
                    conditions.append(f"icd11_main_code IN ({placeholders})")
                    for i, code in enumerate(codes):
                        params[f'code_{i}'] = code
            else:
                # Single code
                conditions.append("icd11_main_code = :icd11_main_code")
                params['icd11_main_code'] = icd11_code
    
    if filters.get('diagnosis_year_from'):
        conditions.append("diagnosis_year >= :diagnosis_year_from")
        params['diagnosis_year_from'] = filters['diagnosis_year_from']
    
    if filters.get('diagnosis_year_to'):
        conditions.append("diagnosis_year <= :diagnosis_year_to")
        params['diagnosis_year_to'] = filters['diagnosis_year_to']
    
    if filters.get('age_from'):
        conditions.append("age_at_diagnosis >= :age_from")
        params['age_from'] = filters['age_from']
    
    if filters.get('age_to'):
        conditions.append("age_at_diagnosis <= :age_to")
        params['age_to'] = filters['age_to']
    
    if filters.get('gender'):
        conditions.append("gender = :gender")
        params['gender'] = filters['gender']
    
    if filters.get('t_category'):
        conditions.append("t_category = :t_category")
        params['t_category'] = filters['t_category']
    
    if filters.get('n_category'):
        conditions.append("n_category = :n_category")
        params['n_category'] = filters['n_category']
    
    if filters.get('m_category'):
        conditions.append("m_category = :m_category")
        params['m_category'] = filters['m_category']
    
    if filters.get('surgery_done') is not None:
        conditions.append("surgery_done = :surgery_done")
        params['surgery_done'] = filters['surgery_done']
    
    if filters.get('chemotherapy_done') is not None:
        conditions.append("chemotherapy_done = :chemotherapy_done")
        params['chemotherapy_done'] = filters['chemotherapy_done']
    
    if filters.get('radiotherapy_done') is not None:
        conditions.append("radiotherapy_done = :radiotherapy_done")
        params['radiotherapy_done'] = filters['radiotherapy_done']
    
    if filters.get('vital_status'):
        conditions.append("vital_status = :vital_status")
        params['vital_status'] = filters['vital_status']
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    query_sql = f"SELECT COUNT(*) FROM registry.patients_anonymized WHERE {where_clause}"
    result = db.execute(text(query_sql), params)
    return result.scalar() or 0


@router.post("/request/create")
def create_research_request(
    request_data: ResearchRequestCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new research data request.
    Researcher submits filters and study purpose.
    
    Required fields:
    - researcher_name: Name of the researcher
    - researcher_email: Email of the researcher
    - purpose_of_study: Description of the research purpose
    
    Filters (at least one required):
    - icd11_main_code: ICD-11 code for the cancer type
    - diagnosis_year_from: Start year for diagnosis
    - diagnosis_year_to: End year for diagnosis
    - age_from/age_to: Age range
    - gender: Patient gender
    - t_category, n_category, m_category: TNM staging
    - surgery_done, chemotherapy_done, etc.: Treatment filters
    """
    try:
        # Check if ALL_DATA is selected before processing filters
        is_all_data = request_data.icd11_main_code == 'ALL_DATA'
        is_all_cancer_types = request_data.icd11_main_code == 'ALL_CANCER_TYPES'
        
        # Handle special "ALL_DATA" and "ALL_CANCER_TYPES" requests
        if is_all_data:
            # Request all data - set all filters to None
            filters = {'icd11_main_code': 'ALL_DATA'}  # Keep the marker for validation
        elif is_all_cancer_types:
            # Request all cancer types - only set icd11_main_code to None to get all types
            filters = {
                'icd11_main_code': 'ALL_CANCER_TYPES',  # Keep the marker for validation
                'icd11_description': request_data.icd11_description,
            }
        else:
            # Normal filtered request
            filters = {
                'icd11_main_code': request_data.icd11_main_code,
                'icd11_description': request_data.icd11_description,
            }
        
        # Add other filters (only if not requesting all data)
        if not is_all_data:
            filters.update({
                'diagnosis_year_from': request_data.diagnosis_year_from,
                'diagnosis_year_to': request_data.diagnosis_year_to,
                'age_from': request_data.age_from,
                'age_to': request_data.age_to,
                'gender': request_data.gender,
                't_category': request_data.t_category,
                'n_category': request_data.n_category,
                'm_category': request_data.m_category,
                'icd11_morphology_code': request_data.icd11_morphology_code,
                'icd11_topography_code': request_data.icd11_topography_code,
                'surgery_done': request_data.surgery_done,
                'chemotherapy_done': request_data.chemotherapy_done,
                'radiotherapy_done': request_data.radiotherapy_done,
                'hormonal_therapy': request_data.hormonal_therapy,
                'immunotherapy': request_data.immunotherapy,
                'recurrence': request_data.recurrence,
                'metastasis': request_data.metastasis,
                'vital_status': request_data.vital_status,
                'treatment_intent': request_data.treatment_intent,
            })
        
        # Validate filters BEFORE removing None values (so we can check for ALL_DATA)
        is_valid, error = validate_filters(filters, allow_all_data=True)
        if not is_valid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)
        
        # Remove None values and empty strings AFTER validation
        # For ALL_DATA, keep the marker but remove it for actual query
        filtered = {}
        for k, v in filters.items():
            if v is not None and v != '':
                # For ALL_DATA, we'll handle it specially in the query function
                if k == 'icd11_main_code' and v in ['ALL_DATA', 'ALL_CANCER_TYPES']:
                    # Don't add to filtered - will be handled in query
                    continue
                filtered[k] = v
        filters = filtered
        
        # Use manual count if provided, otherwise estimate
        manual_count = getattr(request_data, 'manual_record_count', None)
        if manual_count is not None and manual_count >= 0:
            record_count = manual_count
        else:
            # Estimate record count
            # For ALL_DATA, pass empty filters (will return all records)
            if is_all_data:
                record_count = estimate_record_count({}, db)
            else:
                record_count = estimate_record_count(filters, db)
        
        # Privacy check: warn if < 5 records, but allow request creation
        # Admin will review and can approve/reject based on purpose of study
        min_records_warning = 5  # Warning threshold for production
        privacy_warning = None
        if record_count < min_records_warning:
            privacy_warning = f"Warning: Estimated dataset count ({record_count}) is below privacy threshold ({min_records_warning}). This request will be flagged for admin review."
        
        # Allow 0 records - admin can review and decide
        # Only block if explicitly configured to do so
        min_records_block = 0  # Set to 5+ for production to block small datasets
        if min_records_block > 0 and record_count < min_records_block:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Suppressed due to privacy compliance. Expected dataset count ({record_count}) is below minimum threshold ({min_records_block}). Please adjust filters or contact administrator."
            )
        
        # Generate request ID
        request_id = generate_request_id()
        
        # Create research request
        research_request = ResearchRequest(
            request_id=request_id,
            researcher_name=request_data.researcher_name,
            researcher_email=request_data.researcher_email,
            researcher_affiliation=request_data.researcher_affiliation,
            purpose_of_study=request_data.purpose_of_study,
            filters=filters,
            record_count=record_count,
            status='PENDING',
            created_by=current_user.id,
            tenant_id=current_user.tenant_id,
            organization_id=current_user.organization_id,
        )
        
        db.add(research_request)
        db.commit()
        db.refresh(research_request)
        
        response = {
            "request_id": request_id,
            "status": "PENDING",
            "estimated_record_count": record_count,
            "message": "Research request created. Awaiting UMMC Admin approval.",
            "note": "You will be notified once the request is reviewed."
        }
        
        # Add privacy warning if applicable
        if privacy_warning:
            response["privacy_warning"] = privacy_warning
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create research request: {str(e)}"
        )


# ============================================================================
# UMMC ADMIN REVIEW PHASE
# ============================================================================

def check_admin_role(current_user: User = Depends(get_current_user)):
    """Check if user has admin role"""
    roles = [r.slug for r in current_user.roles]
    if 'super_admin' not in roles and 'ummc_admin' not in roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Admin role required."
        )
    return current_user


@router.get("/admin/review")
def list_pending_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin_role)
):
    """List all research requests for admin review (pending, approved, rejected)"""
    # Get all requests grouped by status
    pending_requests = db.query(ResearchRequest).filter(
        ResearchRequest.status == 'PENDING'
    ).order_by(ResearchRequest.created_at.desc()).all()
    
    approved_requests = db.query(ResearchRequest).filter(
        ResearchRequest.status == 'APPROVED'
    ).order_by(ResearchRequest.approved_at.desc() if hasattr(ResearchRequest, 'approved_at') else ResearchRequest.created_at.desc()).all()
    
    rejected_requests = db.query(ResearchRequest).filter(
        ResearchRequest.status == 'REJECTED'
    ).order_by(ResearchRequest.approved_at.desc() if hasattr(ResearchRequest, 'approved_at') else ResearchRequest.created_at.desc()).all()
    
    def format_request(r):
        return {
            "id": str(r.id),
            "request_id": r.request_id,
            "researcher_name": r.researcher_name,
            "researcher_email": r.researcher_email,
            "researcher_affiliation": r.researcher_affiliation,
            "purpose_of_study": r.purpose_of_study,
            "estimated_record_count": r.record_count,
            "filters": r.filters,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "approved_at": r.approved_at.isoformat() if hasattr(r, 'approved_at') and r.approved_at else None,
            "rejection_reason": r.rejection_reason if hasattr(r, 'rejection_reason') else None,
        }
    
    return {
        "pending_requests": [format_request(r) for r in pending_requests],
        "approved_requests": [format_request(r) for r in approved_requests],
        "rejected_requests": [format_request(r) for r in rejected_requests],
        "total_pending": len(pending_requests),
        "total_approved": len(approved_requests),
        "total_rejected": len(rejected_requests),
    }


@router.get("/admin/review/{request_id}")
def get_request_details(
    request_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin_role)
):
    """Get detailed information about a specific research request"""
    request = db.query(ResearchRequest).filter(
        ResearchRequest.request_id == request_id
    ).first()
    
    if not request:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found")
    
    return {
        "request_id": request.request_id,
        "researcher_name": request.researcher_name,
        "researcher_email": request.researcher_email,
        "researcher_affiliation": request.researcher_affiliation,
        "purpose_of_study": request.purpose_of_study,
        "filters": request.filters,
        "estimated_record_count": request.record_count,
        "status": request.status,
        "rejection_reason": request.rejection_reason,
        "created_at": request.created_at,
        "approved_at": request.approved_at,
    }


# ============================================================================
# UMMC ADMIN APPROVAL PHASE
# ============================================================================

class ApprovalDecision(BaseModel):
    """Model for admin approval/rejection decision"""
    request_id: str
    decision: str  # "APPROVE" or "REJECT"
    rejection_reason: Optional[str] = None


def generate_download_token() -> str:
    """Generate secure 24-hour download token: UMMC-TOKEN-<32-char-random>"""
    random_part = secrets.token_hex(16)  # 32 hex characters
    return f"UMMC-TOKEN-{random_part}"


@router.post("/admin/approve")
def approve_or_reject_request(
    decision: ApprovalDecision,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin_role)
):
    """Approve or reject a research request"""
    request = db.query(ResearchRequest).filter(
        ResearchRequest.request_id == decision.request_id
    ).first()
    
    if not request:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found")
    
    if request.status != 'PENDING':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Request is already {request.status}"
        )
    
    if decision.decision.upper() == "REJECT":
        request.status = 'REJECTED'
        request.rejection_reason = decision.rejection_reason or "Rejected by admin"
        request.approved_by = current_user.id
        request.approved_at = datetime.now()
        
        db.commit()
        
        return {
            "request_id": request.request_id,
            "status": "REJECTED",
            "message": "Request has been rejected",
            "rejection_reason": request.rejection_reason
        }
    
    elif decision.decision.upper() == "APPROVE":
        # Generate download token (24-hour expiry)
        download_token = generate_download_token()
        token_expires_at = datetime.now() + timedelta(hours=24)
        
        request.status = 'APPROVED'
        request.approved_by = current_user.id
        request.approved_at = datetime.now()
        request.download_token = download_token
        request.token_expires_at = token_expires_at
        
        db.commit()
        
        return {
            "request_id": request.request_id,
            "status": "APPROVED",
            "message": "Request approved. Data extraction triggered.",
            "download_token": download_token,
            "download_link": f"https://api.ummc.my/research/download?token={download_token}",
            "token_expires_at": token_expires_at.isoformat(),
            "note": "Token is valid for 24 hours"
        }
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Decision must be 'APPROVE' or 'REJECT'"
        )


# ============================================================================
# FILTERED DATA EXTRACTION & DOWNLOAD
# ============================================================================

def extract_filtered_data(filters: Dict[str, Any], db: Session) -> List[Dict]:
    """Extract anonymized patient data using approved filters"""
    # Build dynamic query
    conditions = []
    params = {}
    
    # Handle keyword-based filtering (search in icd11_description)
    if filters.get('icd11_description') and not filters.get('icd11_main_code'):
        keywords = filters['icd11_description'].strip()
        if keywords:
            # Split keywords by comma or space and search for any match
            keyword_list = [k.strip().lower() for k in keywords.replace(',', ' ').split() if k.strip()]
            if keyword_list:
                # Build OR conditions for each keyword
                keyword_conditions = []
                for i, keyword in enumerate(keyword_list):
                    keyword_conditions.append(f"LOWER(icd11_description) LIKE :keyword_{i}")
                    params[f'keyword_{i}'] = f"%{keyword}%"
                if keyword_conditions:
                    conditions.append(f"({' OR '.join(keyword_conditions)})")
    
    # Handle ICD-11 code filtering (if provided)
    if filters.get('icd11_main_code'):
        icd11_code = filters['icd11_main_code']
        # Handle "ALL" option or comma-separated multiple codes
        if icd11_code == 'ALL':
            # Don't filter by ICD code - include all
            pass
        elif ',' in icd11_code:
            # Multiple codes - use IN clause with parameterized query
            codes = [c.strip() for c in icd11_code.split(',') if c.strip()]
            if codes:
                # Build IN clause with placeholders
                placeholders = ','.join([f':code_{i}' for i in range(len(codes))])
                conditions.append(f"icd11_main_code IN ({placeholders})")
                for i, code in enumerate(codes):
                    params[f'code_{i}'] = code
        else:
            # Single code
            conditions.append("icd11_main_code = :icd11_main_code")
            params['icd11_main_code'] = icd11_code
    
    if filters.get('diagnosis_year_from'):
        conditions.append("diagnosis_year >= :diagnosis_year_from")
        params['diagnosis_year_from'] = filters['diagnosis_year_from']
    
    if filters.get('diagnosis_year_to'):
        conditions.append("diagnosis_year <= :diagnosis_year_to")
        params['diagnosis_year_to'] = filters['diagnosis_year_to']
    
    if filters.get('age_from'):
        conditions.append("age_at_diagnosis >= :age_from")
        params['age_from'] = filters['age_from']
    
    if filters.get('age_to'):
        conditions.append("age_at_diagnosis <= :age_to")
        params['age_to'] = filters['age_to']
    
    if filters.get('gender'):
        conditions.append("gender = :gender")
        params['gender'] = filters['gender']
    
    if filters.get('t_category'):
        conditions.append("t_category = :t_category")
        params['t_category'] = filters['t_category']
    
    if filters.get('n_category'):
        conditions.append("n_category = :n_category")
        params['n_category'] = filters['n_category']
    
    if filters.get('m_category'):
        conditions.append("m_category = :m_category")
        params['m_category'] = filters['m_category']
    
    if filters.get('surgery_done') is not None:
        conditions.append("surgery_done = :surgery_done")
        params['surgery_done'] = filters['surgery_done']
    
    if filters.get('chemotherapy_done') is not None:
        conditions.append("chemotherapy_done = :chemotherapy_done")
        params['chemotherapy_done'] = filters['chemotherapy_done']
    
    if filters.get('vital_status'):
        conditions.append("vital_status = :vital_status")
        params['vital_status'] = filters['vital_status']
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    # Query anonymized database
    query_sql = f"""
        SELECT 
            research_id,
            gender,
            nationality,
            age_at_diagnosis,
            diagnosis_year,
            icd11_main_code,
            icd11_description,
            icd11_composite_expression,
            icd11_topography,
            icd11_morphology,
            icd11_behavior_code,
            icd11_stage_code,
            laterality,
            t_category,
            n_category,
            m_category,
            multiple_primary_flag,
            basis_of_diagnosis,
            primary_site_confirmed,
            surgery_done,
            chemotherapy_done,
            radiotherapy_done,
            hormonal_therapy,
            immunotherapy,
            treatment_intent,
            treatment_notes,
            vital_status,
            cause_of_death_icd11,
            recurrence,
            metastasis,
            survival_months,
            followup_notes,
            data_source,
            entry_mode,
            validation_status
        FROM registry.patients_anonymized
        WHERE {where_clause}
        ORDER BY diagnosis_year DESC, age_at_diagnosis DESC
    """
    
    result = db.execute(text(query_sql), params)
    rows = result.fetchall()
    
    # Get column names from result
    if rows:
        # Use the actual column names from the query
        columns = ['research_id', 'gender', 'nationality', 'age_at_diagnosis', 'diagnosis_year',
                   'icd11_main_code', 'icd11_description', 'icd11_composite_expression',
                   'icd11_topography', 'icd11_morphology', 'icd11_behavior_code', 'icd11_stage_code',
                   'laterality', 't_category', 'n_category', 'm_category', 'multiple_primary_flag',
                   'basis_of_diagnosis', 'primary_site_confirmed', 'surgery_done', 'chemotherapy_done',
                   'radiotherapy_done', 'hormonal_therapy', 'immunotherapy', 'treatment_intent',
                   'treatment_notes', 'vital_status', 'cause_of_death_icd11', 'recurrence',
                   'metastasis', 'survival_months', 'followup_notes', 'data_source',
                   'entry_mode', 'validation_status']
        
        return [dict(zip(columns, row)) for row in rows]
    return []


@router.get("/download")
def download_research_data(
    token: str = Query(..., description="Download token"),
    db: Session = Depends(get_db)
):
    """Download filtered research data using secure token"""
    # Validate token format
    if not token or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Download token is required"
        )
    
    # Clean and normalize token
    clean_token = token.strip()
    
    # Find request by token - try exact match first, then case-insensitive
    request = db.query(ResearchRequest).filter(
        ResearchRequest.download_token == clean_token
    ).first()
    
    # If not found, try case-insensitive search (in case of encoding issues)
    if not request:
        from sqlalchemy import func
        request = db.query(ResearchRequest).filter(
            func.lower(ResearchRequest.download_token) == func.lower(clean_token)
        ).first()
    
    if not request:
        # Debug: Check if any tokens exist and log sample
        sample_tokens = db.query(ResearchRequest.download_token).filter(
            ResearchRequest.download_token.isnot(None),
            ResearchRequest.status == 'APPROVED'
        ).limit(3).all()
        
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Download attempt with invalid token. Received: '{clean_token[:30]}...' (length: {len(clean_token)})")
        if sample_tokens:
            logger.warning(f"Sample tokens in DB: {[t[0][:30] + '...' if t[0] else 'None' for t in sample_tokens]}")
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid token: No request found with download token '{clean_token[:30]}...' (length: {len(clean_token)}). Please verify the token is correct and matches the one shown in 'View Details'."
        )
    
    if request.status != 'APPROVED':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Request status is {request.status}, not APPROVED. Cannot download data."
        )
    
    # Check token expiry (handle timezone-aware comparison)
    if request.token_expires_at:
        now = datetime.now(timezone.utc) if request.token_expires_at.tzinfo else datetime.now()
        if request.token_expires_at < now:
            request.status = 'EXPIRED'
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="Token has expired"
            )
    
    # Extract data using filters
    data = extract_filtered_data(request.filters, db)
    
    # Update extraction date
    request.extraction_date = datetime.now()
    request.record_count = len(data)
    db.commit()
    
    # Generate CSV - allow empty data (return empty CSV with headers)
    if not data:
        # Return empty CSV with standard headers instead of error
        # This allows researchers to download even when no data matches
        standard_columns = [
            'research_id', 'gender', 'nationality', 'age_at_diagnosis', 'diagnosis_year',
            'icd11_main_code', 'icd11_description', 'icd11_composite_expression',
            'icd11_topography', 'icd11_morphology', 'icd11_behavior_code', 'icd11_stage_code',
            'laterality', 't_category', 'n_category', 'm_category', 'multiple_primary_flag',
            'basis_of_diagnosis', 'primary_site_confirmed', 'surgery_done', 'chemotherapy_done',
            'radiotherapy_done', 'hormonal_therapy', 'immunotherapy', 'treatment_intent',
            'treatment_notes', 'vital_status', 'cause_of_death_icd11', 'recurrence',
            'metastasis', 'survival_months', 'followup_notes', 'data_source',
            'entry_mode', 'validation_status'
        ]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=standard_columns)
        writer.writeheader()
        # No data rows - just headers
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=ummc_research_data_{request.request_id}_empty.csv",
                "X-Data-Count": "0",
                "X-Message": "No data found matching the filters. Empty CSV with headers returned."
            }
        )
    
    # Convert UUIDs and other non-serializable types to strings
    csv_data = []
    for row in data:
        csv_row = {}
        for key, value in row.items():
            if value is None:
                csv_row[key] = ''
            elif isinstance(value, uuid.UUID):
                csv_row[key] = str(value)
            elif isinstance(value, (datetime, date)):
                csv_row[key] = value.isoformat()
            else:
                csv_row[key] = value
        csv_data.append(csv_row)
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=csv_data[0].keys())
    writer.writeheader()
    writer.writerows(csv_data)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=ummc_research_data_{request.request_id}.csv",
            "X-Data-Count": str(len(csv_data))
        }
    )


@router.get("/download-json")
def download_research_data_json(
    token: str = Query(..., description="Download token"),
    db: Session = Depends(get_db)
):
    """Download filtered research data as JSON for ML training"""
    # Validate token format
    if not token or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Download token is required"
        )
    
    # Clean and normalize token
    clean_token = token.strip()
    
    # Find request by token
    request = db.query(ResearchRequest).filter(
        ResearchRequest.download_token == clean_token
    ).first()
    
    if not request:
        from sqlalchemy import func
        request = db.query(ResearchRequest).filter(
            func.lower(ResearchRequest.download_token) == func.lower(clean_token)
        ).first()
    
    if not request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid download token"
        )
    
    if request.status != 'APPROVED':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Request status is {request.status}, not APPROVED"
        )
    
    # Check token expiry
    if request.token_expires_at:
        now = datetime.now(timezone.utc) if request.token_expires_at.tzinfo else datetime.now()
        if request.token_expires_at < now:
            request.status = 'EXPIRED'
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="Token has expired"
            )
    
    # Extract data using filters
    data = extract_filtered_data(request.filters, db)
    
    # Update extraction date
    request.extraction_date = datetime.now()
    request.record_count = len(data)
    db.commit()
    
    # Convert to JSON-serializable format
    json_data = []
    for row in data:
        json_row = {}
        for key, value in row.items():
            if value is None:
                json_row[key] = None
            elif isinstance(value, uuid.UUID):
                json_row[key] = str(value)
            elif isinstance(value, (datetime, date)):
                json_row[key] = value.isoformat()
            else:
                json_row[key] = value
        json_data.append(json_row)
    
    return json_data


@router.get("/requests/my")
def list_my_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all research requests created by the current researcher"""
    requests = db.query(ResearchRequest).filter(
        ResearchRequest.created_by == current_user.id
    ).order_by(ResearchRequest.created_at.desc()).all()
    
    def format_request(r):
        response = {
            "request_id": r.request_id,
            "researcher_name": r.researcher_name,
            "researcher_email": r.researcher_email,
            "purpose_of_study": r.purpose_of_study,
            "status": r.status,
            "estimated_record_count": r.record_count,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "filters": r.filters,
        }
        
        if r.status == 'APPROVED':
            response["download_token"] = r.download_token
            response["download_link"] = f"https://api.ummc.my/research/download?token={r.download_token}" if r.download_token else None
            response["token_expires_at"] = r.token_expires_at.isoformat() if r.token_expires_at else None
            response["approved_at"] = r.approved_at.isoformat() if hasattr(r, 'approved_at') and r.approved_at else None
        
        if r.status == 'REJECTED':
            response["rejection_reason"] = r.rejection_reason if hasattr(r, 'rejection_reason') else None
            response["rejected_at"] = r.approved_at.isoformat() if hasattr(r, 'approved_at') and r.approved_at else None
        
        return response
    
    return {
        "requests": [format_request(r) for r in requests],
        "total": len(requests),
        "pending": len([r for r in requests if r.status == 'PENDING']),
        "approved": len([r for r in requests if r.status == 'APPROVED']),
        "rejected": len([r for r in requests if r.status == 'REJECTED']),
    }


@router.get("/request/status/{request_id}")
def get_request_status(
    request_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get status of a research request (for researchers)"""
    request = db.query(ResearchRequest).filter(
        ResearchRequest.request_id == request_id,
        ResearchRequest.created_by == current_user.id
    ).first()
    
    if not request:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found")
    
    response = {
        "request_id": request.request_id,
        "status": request.status,
        "estimated_record_count": request.record_count,
        "created_at": request.created_at,
    }
    
    if request.status == 'APPROVED':
        response["download_token"] = request.download_token
        response["download_link"] = f"https://api.ummc.my/research/download?token={request.download_token}"
        response["token_expires_at"] = request.token_expires_at.isoformat() if request.token_expires_at else None
    
    if request.status == 'REJECTED':
        response["rejection_reason"] = request.rejection_reason
    
    return response



# ============================================================================
# SECURE NOTEBOOK DATA ACCESS
# ============================================================================

@router.get("/secure-data/{request_id}")
def get_secure_notebook_data(
    request_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Secure endpoint for JupyterLite notebook data access.
    Requires authentication and validates user access to the request.
    Returns data only if user is authorized.
    """
    # Find the research request
    request = db.query(ResearchRequest).filter(
        ResearchRequest.request_id == request_id
    ).first()
    
    if not request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Research request not found"
        )
    
    # Verify user owns this request
    if request.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this research request"
        )
    
    # Verify request is approved
    if request.status != 'APPROVED':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Request is not approved. Current status: {request.status}"
        )
    
    # Verify token hasn't expired
    if request.token_expires_at and request.token_expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access token has expired"
        )
    
    # Use the same filter extraction logic as download endpoint
    filters = request.filters or {}
    
    # Build dynamic query with filters
    conditions = []
    params = {}
    
    # Handle keyword-based filtering (search in icd11_description)
    if filters.get('icd11_description') and not filters.get('icd11_main_code'):
        keywords = filters['icd11_description'].strip()
        if keywords:
            keyword_list = [k.strip().lower() for k in keywords.replace(',', ' ').split() if k.strip()]
            if keyword_list:
                keyword_conditions = []
                for i, keyword in enumerate(keyword_list):
                    keyword_conditions.append(f"LOWER(icd11_description) LIKE :keyword_{i}")
                    params[f'keyword_{i}'] = f"%{keyword}%"
                if keyword_conditions:
                    conditions.append(f"({' OR '.join(keyword_conditions)})")
    
    # Handle ICD-11 code filtering
    if filters.get('icd11_main_code'):
        icd11_code = filters['icd11_main_code']
        if icd11_code == 'ALL' or icd11_code == 'ALL_DATA' or icd11_code == 'ALL_CANCER_TYPES':
            pass  # Don't filter by ICD code
        elif ',' in icd11_code:
            codes = [c.strip() for c in icd11_code.split(',') if c.strip()]
            if codes:
                placeholders = ','.join([f':code_{i}' for i in range(len(codes))])
                conditions.append(f"icd11_main_code IN ({placeholders})")
                for i, code in enumerate(codes):
                    params[f'code_{i}'] = code
        else:
            conditions.append("icd11_main_code = :icd11_main_code")
            params['icd11_main_code'] = icd11_code
    
    if filters.get('diagnosis_year_from'):
        conditions.append("diagnosis_year >= :diagnosis_year_from")
        params['diagnosis_year_from'] = filters['diagnosis_year_from']
    
    if filters.get('diagnosis_year_to'):
        conditions.append("diagnosis_year <= :diagnosis_year_to")
        params['diagnosis_year_to'] = filters['diagnosis_year_to']
    
    if filters.get('age_from'):
        conditions.append("age_at_diagnosis >= :age_from")
        params['age_from'] = filters['age_from']
    
    if filters.get('age_to'):
        conditions.append("age_at_diagnosis <= :age_to")
        params['age_to'] = filters['age_to']
    
    if filters.get('gender'):
        conditions.append("gender = :gender")
        params['gender'] = filters['gender']
    
    if filters.get('t_category'):
        conditions.append("t_category = :t_category")
        params['t_category'] = filters['t_category']
    
    if filters.get('n_category'):
        conditions.append("n_category = :n_category")
        params['n_category'] = filters['n_category']
    
    if filters.get('m_category'):
        conditions.append("m_category = :m_category")
        params['m_category'] = filters['m_category']
    
    if filters.get('surgery_done') is not None:
        conditions.append("surgery_done = :surgery_done")
        params['surgery_done'] = filters['surgery_done']
    
    if filters.get('chemotherapy_done') is not None:
        conditions.append("chemotherapy_done = :chemotherapy_done")
        params['chemotherapy_done'] = filters['chemotherapy_done']
    
    if filters.get('radiotherapy_done') is not None:
        conditions.append("radiotherapy_done = :radiotherapy_done")
        params['radiotherapy_done'] = filters['radiotherapy_done']
    
    if filters.get('vital_status'):
        conditions.append("vital_status = :vital_status")
        params['vital_status'] = filters['vital_status']
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    # Query anonymized database
    query_sql = f"""
        SELECT 
            research_id,
            gender,
            nationality,
            age_at_diagnosis,
            diagnosis_year,
            icd11_main_code,
            icd11_description,
            icd11_composite_expression,
            icd11_topography,
            icd11_morphology,
            icd11_behavior_code,
            icd11_stage_code,
            laterality,
            t_category,
            n_category,
            m_category,
            multiple_primary_flag,
            basis_of_diagnosis,
            primary_site_confirmed,
            surgery_done,
            chemotherapy_done,
            radiotherapy_done,
            hormonal_therapy,
            immunotherapy,
            treatment_intent,
            treatment_notes,
            vital_status,
            cause_of_death_icd11,
            recurrence,
            metastasis,
            survival_months,
            followup_notes,
            data_source,
            entry_mode,
            validation_status
        FROM registry.patients_anonymized
        WHERE {where_clause}
        ORDER BY diagnosis_year DESC, age_at_diagnosis DESC
    """
    
    # Execute query
    result = db.execute(text(query_sql), params)
    rows = result.fetchall()
    
    # Column names
    columns = ['research_id', 'gender', 'nationality', 'age_at_diagnosis', 'diagnosis_year',
               'icd11_main_code', 'icd11_description', 'icd11_composite_expression',
               'icd11_topography', 'icd11_morphology', 'icd11_behavior_code', 'icd11_stage_code',
               'laterality', 't_category', 'n_category', 'm_category', 'multiple_primary_flag',
               'basis_of_diagnosis', 'primary_site_confirmed', 'surgery_done', 'chemotherapy_done',
               'radiotherapy_done', 'hormonal_therapy', 'immunotherapy', 'treatment_intent',
               'treatment_notes', 'vital_status', 'cause_of_death_icd11', 'recurrence',
               'metastasis', 'survival_months', 'followup_notes', 'data_source',
               'entry_mode', 'validation_status']
    
    # Convert to JSON with transformations: null→0, false→0, true→1
    json_data = []
    for row in rows:
        json_row = {}
        for i, col in enumerate(columns):
            value = row[i]
            # Convert special types
            if isinstance(value, (datetime, date)):
                json_row[col] = value.isoformat()
            elif value is None:
                json_row[col] = 0  # Convert null to 0 for ML
            elif isinstance(value, bool):
                json_row[col] = 1 if value else 0  # Convert bool to 0/1
            else:
                json_row[col] = value
        json_data.append(json_row)
    
    return {
        "success": True,
        "request_id": request_id,
        "data": json_data,
        "rows": len(json_data),
        "columns": len(columns),
        "message": "Data accessed securely via authenticated session"
    }
