"""
AutoML API for researchers.

Trains multiple models server-side on the registry data for a chosen target
and returns a ranked leaderboard plus the best model's metrics and feature
importance. No need for the client to upload a dataset.
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime
import os
import uuid
import logging

from core.deps import get_db, get_current_user
from core.rate_limit import limiter
from db.models.users import User
from db.models.safehaven import MLTrainingResult
from services import automl as automl_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml-automl", tags=["ml-automl"])


def _require_researcher(current_user: User):
    roles = [r.slug for r in current_user.roles]
    if not any(r in ("researcher", "super_admin", "ummc_admin") for r in roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="AutoML is only available to researchers",
        )


class AutoMLRequest(BaseModel):
    target_variable: str = Field(..., description="Target column to predict")
    features: Optional[List[str]] = Field(None, description="Feature columns (default: all safe columns)")
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    random_state: int = Field(42)
    row_limit: int = Field(automl_service.DEFAULT_ROWS, ge=200, le=automl_service.MAX_ROWS)
    cross_validate: bool = Field(True)
    tune: bool = Field(True, description="Hyperparameter-tune the winning model")
    save_model: bool = Field(False, description="Persist the best model for download")
    save_to_history: bool = Field(True, description="Record this run in the researcher's model history")


def _persist_history(db: Session, user: User, result: dict):
    """Best-effort: store an AutoML run in MLTrainingResult so it shows in history."""
    try:
        best = result.get("best_model", {})
        artifact = result.get("model_artifact") or {}
        model_id = f"automl-{result.get('target_variable')}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        rec = MLTrainingResult(
            model_id=model_id,
            algorithm=best.get("algorithm", "automl"),
            target_variable=result.get("target_variable", ""),
            features=result.get("features_used", []),
            hyperparameters=best.get("best_params") or {},
            metrics=best.get("metrics") or {},
            feature_importance=best.get("feature_importance") or [],
            confusion_matrix=best.get("confusion_matrix"),
            roc_curve=best.get("roc_curve"),
            training_status="completed",
            training_duration_seconds=int(result.get("total_seconds", 0)),
            n_samples=result.get("data_stats", {}).get("rows_used"),
            n_features=result.get("data_stats", {}).get("n_features"),
            n_train=result.get("data_stats", {}).get("n_train"),
            n_test=result.get("data_stats", {}).get("n_test"),
            model_artifact_path=artifact.get("path"),
            model_artifact_size=artifact.get("size_bytes"),
            model_artifact_hash=artifact.get("sha256"),
            tags=["automl", "tuned" if result.get("tuned") else "baseline"],
            notes=f"AutoML run on '{result.get('target_variable')}' ({result.get('task_type')})",
            created_by=user.id,
        )
        db.add(rec)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning(f"Could not persist AutoML history: {e}")


@router.get("/options")
def get_automl_options(current_user: User = Depends(get_current_user)):
    """Valid targets, features and limits for the AutoML UI."""
    _require_researcher(current_user)
    return automl_service.list_options()


@router.post("/run")
@limiter.limit("10/minute")
def run_automl(
    req: AutoMLRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    request: Request = None,
):
    """
    Run AutoML: train several scikit-learn models on registry data for the
    chosen target and return a ranked leaderboard with the best model.
    """
    _require_researcher(current_user)
    try:
        result = automl_service.run_automl(
            db=db,
            target_variable=req.target_variable,
            features=req.features,
            test_size=req.test_size,
            random_state=req.random_state,
            row_limit=req.row_limit,
            cross_validate=req.cross_validate,
            tune=req.tune,
            save_model=req.save_model,
        )
        if req.save_to_history:
            _persist_history(db, current_user, result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"AutoML run failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AutoML failed: {e}",
        )


@router.get("/model/{filename}")
def download_model(
    filename: str,
    current_user: User = Depends(get_current_user),
):
    """Download a saved AutoML model artifact (researcher only)."""
    _require_researcher(current_user)
    # Prevent path traversal: only allow a bare filename within the model dir
    safe = os.path.basename(filename)
    if safe != filename or not safe.endswith(".joblib"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid model filename")
    path = os.path.join(automl_service.MODEL_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model artifact not found")
    return FileResponse(path=path, filename=safe, media_type="application/octet-stream")
