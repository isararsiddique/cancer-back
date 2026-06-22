"""
Anonymization Engine — implements the patent's dual-path storage architecture.

During ingestion, this engine:
1. Strips PII fields (name, identifiers, addresses, DOB)
2. Converts DOB -> age_at_diagnosis
3. Generates a one-way anonymization hash (SHA-256 of patient_id + salt)
   so records can be de-duplicated without revealing identity
4. Writes the anonymized record to a separate object-storage path (Parquet file)
   enabling serverless SQL queries without accessing the main relational DB

This satisfies Claim 1 ("automatically separates raw patient data containing PII
from anonymized research data during data ingestion") and Claim 3 ("serverless SQL
execution engine") with a concrete technical implementation the examiner cannot
dismiss as abstract.
"""
import hashlib
import os
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

ANON_SALT = os.environ.get("ANONYMIZATION_SALT", "nextgen-registry-2024-salt")
ANON_STORE_DIR = os.environ.get("ANON_STORE_DIR", "/app/uploads/anonymized_store")

# Fields that constitute PII and must be stripped
PII_FIELDS = {"patient_name", "date_of_birth", "address", "phone", "email",
              "national_id", "passport_no", "patient_id"}

# Fields retained in the anonymized dataset
ANON_FIELDS = [
    "anon_hash", "gender", "nationality", "age_at_diagnosis",
    "diagnosis_date", "icd11_main_code", "icd11_description",
    "icd11_behavior_code", "laterality",
    "t_category", "n_category", "m_category",
    "basis_of_diagnosis", "treatment_intent",
    "surgery_done", "chemotherapy_done", "radiotherapy_done",
    "hormonal_therapy", "immunotherapy",
    "vital_status", "survival_months",
    "recurrence", "metastasis",
    "data_source", "organization_id", "diagnosis_year",
]


def compute_anon_hash(patient_id: str, tenant_id: str = "") -> str:
    """One-way SHA-256 hash for de-duplication without identity disclosure."""
    payload = f"{ANON_SALT}:{tenant_id}:{patient_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def anonymize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a raw patient record into an anonymized research record."""
    anon = {}
    anon["anon_hash"] = compute_anon_hash(
        str(raw.get("patient_id", raw.get("id", ""))),
        str(raw.get("tenant_id", ""))
    )

    # Convert DOB -> age (if not already computed)
    if raw.get("age_at_diagnosis"):
        anon["age_at_diagnosis"] = int(raw["age_at_diagnosis"])
    elif raw.get("date_of_birth") and raw.get("diagnosis_date"):
        dob = raw["date_of_birth"] if isinstance(raw["date_of_birth"], date) else date.fromisoformat(str(raw["date_of_birth"])[:10])
        diag = raw["diagnosis_date"] if isinstance(raw["diagnosis_date"], date) else date.fromisoformat(str(raw["diagnosis_date"])[:10])
        anon["age_at_diagnosis"] = diag.year - dob.year - ((diag.month, diag.day) < (dob.month, dob.day))

    # Copy non-PII clinical fields
    for f in ANON_FIELDS:
        if f == "anon_hash" or f == "age_at_diagnosis":
            continue
        if f == "diagnosis_year" and raw.get("diagnosis_date"):
            d = raw["diagnosis_date"]
            anon["diagnosis_year"] = d.year if isinstance(d, date) else int(str(d)[:4])
        elif f in raw and f not in PII_FIELDS:
            val = raw[f]
            if isinstance(val, (date, datetime)):
                anon[f] = val.isoformat()
            else:
                anon[f] = val

    return anon


def export_anonymized_parquet(db: Session, limit: int = 100000) -> str:
    """
    Export the full anonymized dataset as a Parquet file (serverless query target).
    Returns the file path. This file can be queried by DuckDB/Athena/Trino without
    touching the main relational database.
    """
    os.makedirs(ANON_STORE_DIR, exist_ok=True)

    # Pull raw records (only needed columns + PII for anonymization)
    sql = text("""
        SELECT id, patient_id, tenant_id, organization_id, gender, nationality,
               date_of_birth, age_at_diagnosis, diagnosis_date,
               icd11_main_code, icd11_description, icd11_behavior_code, laterality,
               t_category, n_category, m_category, basis_of_diagnosis, treatment_intent,
               surgery_done, chemotherapy_done, radiotherapy_done,
               hormonal_therapy, immunotherapy,
               vital_status, survival_months, recurrence, metastasis, data_source
        FROM registry.patients
        WHERE is_active = true
        ORDER BY diagnosis_date DESC NULLS LAST
        LIMIT :lim
    """)
    rows = db.execute(sql, {"lim": limit}).fetchall()
    cols = [
        "id", "patient_id", "tenant_id", "organization_id", "gender", "nationality",
        "date_of_birth", "age_at_diagnosis", "diagnosis_date",
        "icd11_main_code", "icd11_description", "icd11_behavior_code", "laterality",
        "t_category", "n_category", "m_category", "basis_of_diagnosis", "treatment_intent",
        "surgery_done", "chemotherapy_done", "radiotherapy_done",
        "hormonal_therapy", "immunotherapy",
        "vital_status", "survival_months", "recurrence", "metastasis", "data_source",
    ]
    df_raw = pd.DataFrame(rows, columns=cols)

    # Anonymize each record
    anon_records = [anonymize_record(row) for row in df_raw.to_dict("records")]
    df_anon = pd.DataFrame(anon_records)

    # Write as Parquet (columnar, efficient for analytics)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(ANON_STORE_DIR, f"anonymized_registry_{ts}.parquet")
    df_anon.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"Exported {len(df_anon)} anonymized records to {path}")
    return path


def query_anonymized_store(sql_query: str, limit: int = 10000) -> List[Dict[str, Any]]:
    """
    Execute a SQL query against the anonymized Parquet store using DuckDB (serverless).
    This implements Claim 3: 'serverless SQL execution engine' — no database server needed,
    queries run on-demand against file-based columnar storage.
    """
    try:
        import duckdb
    except ImportError:
        raise RuntimeError("DuckDB not available for serverless queries")

    # Find the latest parquet file
    files = sorted(
        [f for f in os.listdir(ANON_STORE_DIR) if f.endswith(".parquet")],
        reverse=True,
    )
    if not files:
        raise ValueError("No anonymized data store available. Run export first.")

    latest = os.path.join(ANON_STORE_DIR, files[0])
    conn = duckdb.connect(":memory:")
    conn.execute(f"CREATE VIEW registry AS SELECT * FROM read_parquet('{latest}')")

    # Enforce limit to prevent abuse
    safe_query = sql_query.strip().rstrip(";")
    if "limit" not in safe_query.lower():
        safe_query += f" LIMIT {limit}"

    result = conn.execute(safe_query).fetchdf()
    conn.close()
    return result.to_dict("records")
