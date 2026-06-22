"""
Serverless Analytics API — Claim 3 implementation.

Enables SQL queries directly against anonymized data stored in columnar file format
(Parquet) using DuckDB as an embedded serverless query engine. No pre-provisioned
database infrastructure is required for research analytics.

This is the concrete technical implementation that satisfies:
- Claim 3: "anonymized data is queried using a serverless SQL execution engine"
- Claim 1(c): "data layer comprising... a serverless query engine"
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import logging

from core.deps import get_db, get_current_user
from core.rate_limit import limiter
from db.models.users import User
from services.anonymization_engine import export_anonymized_parquet, query_anonymized_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["serverless-analytics"])

RESEARCHER_ROLES = {"researcher", "super_admin", "ummc_admin"}


def _require_researcher(user: User):
    if not any(r.slug in RESEARCHER_ROLES for r in user.roles):
        raise HTTPException(status_code=403, detail="Research role required")


class AnalyticsQuery(BaseModel):
    sql: str = Field(..., min_length=5, max_length=2000,
                     description="SQL query against the anonymized registry view")
    limit: int = Field(1000, ge=1, le=10000)


@router.post("/export-store")
def export_store(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Admin: export the current anonymized dataset to Parquet (object storage).
    This creates the serverless-queryable data store.
    """
    if not any(r.slug in ("super_admin", "ummc_admin") for r in current_user.roles):
        raise HTTPException(status_code=403, detail="Admin required")
    try:
        path = export_anonymized_parquet(db)
        return {"status": "exported", "path": path, "note": "Anonymized store updated for serverless queries."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
@limiter.limit("30/minute")
def run_query(
    body: AnalyticsQuery,
    current_user: User = Depends(get_current_user),
    request: Request = None,
):
    """
    Execute a read-only SQL query against the anonymized Parquet data store
    using the embedded serverless query engine (DuckDB).

    No database server is involved — queries run on-demand against file-based
    columnar storage. This implements patent Claim 3.

    Example queries:
    - SELECT diagnosis_year, COUNT(*) as cases FROM registry GROUP BY diagnosis_year ORDER BY diagnosis_year
    - SELECT icd11_main_code, AVG(age_at_diagnosis) as avg_age FROM registry GROUP BY icd11_main_code
    - SELECT gender, vital_status, COUNT(*) FROM registry GROUP BY gender, vital_status
    """
    _require_researcher(current_user)

    # Security: block write operations
    forbidden = {"insert", "update", "delete", "drop", "alter", "create", "truncate", "grant", "revoke"}
    tokens = body.sql.lower().split()
    if any(t in forbidden for t in tokens):
        raise HTTPException(status_code=400, detail="Only SELECT queries are permitted")

    try:
        results = query_anonymized_store(body.sql, limit=body.limit)
        return {
            "query": body.sql,
            "rows": len(results),
            "data": results,
            "engine": "DuckDB (serverless, file-based)",
            "storage": "Parquet (columnar, anonymized)",
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Serverless query failed: {e}")
        raise HTTPException(status_code=400, detail=f"Query error: {e}")


@router.get("/schema")
def get_schema(current_user: User = Depends(get_current_user)):
    """Return the anonymized data schema available for serverless queries."""
    _require_researcher(current_user)
    return {
        "engine": "DuckDB (embedded serverless)",
        "storage_format": "Apache Parquet",
        "view_name": "registry",
        "columns": [
            {"name": "anon_hash", "type": "VARCHAR", "description": "One-way SHA-256 hash for de-duplication (not reversible to patient identity)"},
            {"name": "gender", "type": "VARCHAR"},
            {"name": "nationality", "type": "VARCHAR"},
            {"name": "age_at_diagnosis", "type": "INTEGER", "description": "Computed from DOB (DOB itself is stripped)"},
            {"name": "diagnosis_date", "type": "DATE"},
            {"name": "diagnosis_year", "type": "INTEGER"},
            {"name": "icd11_main_code", "type": "VARCHAR"},
            {"name": "icd11_description", "type": "VARCHAR"},
            {"name": "t_category", "type": "VARCHAR"},
            {"name": "n_category", "type": "VARCHAR"},
            {"name": "m_category", "type": "VARCHAR"},
            {"name": "surgery_done", "type": "BOOLEAN"},
            {"name": "chemotherapy_done", "type": "BOOLEAN"},
            {"name": "radiotherapy_done", "type": "BOOLEAN"},
            {"name": "vital_status", "type": "VARCHAR"},
            {"name": "survival_months", "type": "INTEGER"},
            {"name": "recurrence", "type": "BOOLEAN"},
            {"name": "metastasis", "type": "BOOLEAN"},
        ],
        "note": "All PII (name, DOB, address, identifiers) is permanently stripped. Only clinical and outcome variables are retained.",
    }
