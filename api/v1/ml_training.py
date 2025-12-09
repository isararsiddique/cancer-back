"""
ML Training & Model Management API
Backend endpoints for saving, loading, and managing ML training results
"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
from pydantic import BaseModel, Field
from pathlib import Path
import uuid

from core.deps import get_db, get_current_user
from db.models.safehaven import MLTrainingResult
from db.models.research import ResearchRequest
from db.models.users import User

router = APIRouter(prefix="/ml-training", tags=["ml-training"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class TrainingResultCreate(BaseModel):
    """Save a new ML training result"""
    model_id: str = Field(..., description="Client-generated unique model ID")
    research_request_id: Optional[str] = None
    project_id: Optional[str] = None
    
    # Model Configuration
    algorithm: str = Field(..., description="Algorithm used: xgboost, random_forest, neural_network")
    target_variable: str
    features: List[str]
    hyperparameters: Optional[Dict[str, Any]] = None
    
    # Training Configuration
    test_size: Optional[float] = None
    random_state: Optional[int] = None
    custom_pipeline: Optional[str] = None
    
    # Training Results
    metrics: Dict[str, Any] = Field(..., description="All training metrics")
    feature_importance: Optional[List[Dict[str, Any]]] = None
    predictions: Optional[List[Dict[str, Any]]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    roc_curve: Optional[List[Dict[str, float]]] = None
    cv_scores: Optional[Dict[str, Any]] = None
    
    # Training Metadata
    training_status: str = "completed"
    error_message: Optional[str] = None
    training_duration_seconds: Optional[int] = None
    
    # Resource Metrics
    resource_metrics: Optional[Dict[str, Any]] = Field(None, description="CPU and memory usage metrics")
    
    # Data Statistics
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    n_train: Optional[int] = None
    n_test: Optional[int] = None
    n_val: Optional[int] = None
    
    # Optional metadata
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


class TrainingResultUpdate(BaseModel):
    """Update training result metadata"""
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


class TrainingResultResponse(BaseModel):
    """Response model for training results"""
    id: str
    model_id: str
    research_request_id: Optional[str] = None
    project_id: Optional[str] = None
    algorithm: str
    target_variable: str
    features: List[str]
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any]
    feature_importance: Optional[List[Dict[str, Any]]] = None
    predictions: Optional[List[Dict[str, Any]]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    roc_curve: Optional[List[Dict[str, float]]] = None
    cv_scores: Optional[Dict[str, Any]] = None
    training_status: str
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    resource_metrics: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/save", response_model=TrainingResultResponse, status_code=status.HTTP_201_CREATED)
def save_training_result(
    result_data: TrainingResultCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Save ML training results to the database.
    Allows researchers to persist their model training results for later reference.
    """
    try:
        # Check if model_id already exists
        existing = db.query(MLTrainingResult).filter(
            MLTrainingResult.model_id == result_data.model_id
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model with ID '{result_data.model_id}' already exists"
            )
        
        # Validate research_request_id if provided
        research_request_id = None
        if result_data.research_request_id:
            try:
                research_request_id = uuid.UUID(result_data.research_request_id)
                # Verify request exists and belongs to user
                request = db.query(ResearchRequest).filter(
                    ResearchRequest.id == research_request_id,
                    ResearchRequest.created_by == current_user.id
                ).first()
                if not request:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Research request not found or access denied"
                    )
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid research_request_id format"
                )
        
        # Validate project_id if provided
        project_id = None
        if result_data.project_id:
            try:
                project_id = uuid.UUID(result_data.project_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid project_id format"
                )
        
        # Create training result record
        training_result = MLTrainingResult(
            model_id=result_data.model_id,
            research_request_id=research_request_id,
            project_id=project_id,
            algorithm=result_data.algorithm,
            target_variable=result_data.target_variable,
            features=result_data.features,
            hyperparameters=result_data.hyperparameters,
            test_size=result_data.test_size,
            random_state=result_data.random_state,
            custom_pipeline=result_data.custom_pipeline,
            metrics=result_data.metrics,
            feature_importance=result_data.feature_importance,
            predictions=result_data.predictions,
            confusion_matrix=result_data.confusion_matrix,
            roc_curve=result_data.roc_curve,
            cv_scores=result_data.cv_scores,
            training_status=result_data.training_status,
            error_message=result_data.error_message,
            training_duration_seconds=result_data.training_duration_seconds,
            n_samples=result_data.n_samples,
            n_features=result_data.n_features,
            n_train=result_data.n_train,
            n_test=result_data.n_test,
            n_val=result_data.n_val,
            resource_metrics=result_data.resource_metrics,
            tags=result_data.tags or [],
            notes=result_data.notes,
            created_by=current_user.id
        )
        
        db.add(training_result)
        db.commit()
        db.refresh(training_result)
        
        return TrainingResultResponse(
            id=str(training_result.id),
            model_id=training_result.model_id,
            research_request_id=str(training_result.research_request_id) if training_result.research_request_id else None,
            project_id=str(training_result.project_id) if training_result.project_id else None,
            algorithm=training_result.algorithm,
            target_variable=training_result.target_variable,
            features=training_result.features,
            hyperparameters=training_result.hyperparameters,
            metrics=training_result.metrics,
            feature_importance=training_result.feature_importance,
            predictions=training_result.predictions,
            confusion_matrix=training_result.confusion_matrix,
            roc_curve=training_result.roc_curve,
            cv_scores=training_result.cv_scores,
            training_status=training_result.training_status,
            n_samples=training_result.n_samples,
            n_features=training_result.n_features,
            resource_metrics=training_result.resource_metrics,
            tags=training_result.tags,
            notes=training_result.notes,
            created_at=training_result.created_at.isoformat() if training_result.created_at else "",
            updated_at=training_result.updated_at.isoformat() if training_result.updated_at else ""
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save training result: {str(e)}"
        )


@router.get("/list", response_model=List[TrainingResultResponse])
def list_training_results(
    research_request_id: Optional[str] = Query(None, description="Filter by research request ID"),
    algorithm: Optional[str] = Query(None, description="Filter by algorithm"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all training results for the current user.
    Results are ordered by creation date (newest first).
    """
    query = db.query(MLTrainingResult).filter(
        MLTrainingResult.created_by == current_user.id
    )
    
    # Apply filters
    if research_request_id:
        try:
            req_id = uuid.UUID(research_request_id)
            query = query.filter(MLTrainingResult.research_request_id == req_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid research_request_id format"
            )
    
    if algorithm:
        query = query.filter(MLTrainingResult.algorithm == algorithm)
    
    # Order by creation date (newest first)
    query = query.order_by(desc(MLTrainingResult.created_at))
    
    # Apply pagination
    results = query.offset(offset).limit(limit).all()
    
    return [
        TrainingResultResponse(
            id=str(r.id),
            model_id=r.model_id,
            research_request_id=str(r.research_request_id) if r.research_request_id else None,
            project_id=str(r.project_id) if r.project_id else None,
            algorithm=r.algorithm,
            target_variable=r.target_variable,
            features=r.features,
            hyperparameters=r.hyperparameters,
            metrics=r.metrics,
            feature_importance=r.feature_importance,
            predictions=r.predictions,
            confusion_matrix=r.confusion_matrix,
            roc_curve=r.roc_curve,
            cv_scores=r.cv_scores,
            training_status=r.training_status,
            n_samples=r.n_samples,
            n_features=r.n_features,
            tags=r.tags or [],
            notes=r.notes,
            created_at=r.created_at.isoformat() if r.created_at else "",
            updated_at=r.updated_at.isoformat() if r.updated_at else ""
        )
        for r in results
    ]


@router.get("/{model_id}", response_model=TrainingResultResponse)
def get_training_result(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific training result by model_id.
    """
    result = db.query(MLTrainingResult).filter(
        MLTrainingResult.model_id == model_id,
        MLTrainingResult.created_by == current_user.id
    ).first()
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training result not found"
        )
    
    return TrainingResultResponse(
        id=str(result.id),
        model_id=result.model_id,
        research_request_id=str(result.research_request_id) if result.research_request_id else None,
        project_id=str(result.project_id) if result.project_id else None,
        algorithm=result.algorithm,
        target_variable=result.target_variable,
        features=result.features,
        hyperparameters=result.hyperparameters,
        metrics=result.metrics,
        feature_importance=result.feature_importance,
        predictions=result.predictions,
        confusion_matrix=result.confusion_matrix,
        roc_curve=result.roc_curve,
        cv_scores=result.cv_scores,
        training_status=result.training_status,
        n_samples=result.n_samples,
        n_features=result.n_features,
        tags=result.tags or [],
        notes=result.notes,
        created_at=result.created_at.isoformat() if result.created_at else "",
        updated_at=result.updated_at.isoformat() if result.updated_at else ""
    )


@router.patch("/{model_id}", response_model=TrainingResultResponse)
def update_training_result(
    model_id: str,
    update_data: TrainingResultUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update training result metadata (tags, notes).
    """
    result = db.query(MLTrainingResult).filter(
        MLTrainingResult.model_id == model_id,
        MLTrainingResult.created_by == current_user.id
    ).first()
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training result not found"
        )
    
    if update_data.tags is not None:
        result.tags = update_data.tags
    if update_data.notes is not None:
        result.notes = update_data.notes
    
    db.commit()
    db.refresh(result)
    
    return TrainingResultResponse(
        id=str(result.id),
        model_id=result.model_id,
        research_request_id=str(result.research_request_id) if result.research_request_id else None,
        project_id=str(result.project_id) if result.project_id else None,
        algorithm=result.algorithm,
        target_variable=result.target_variable,
        features=result.features,
        hyperparameters=result.hyperparameters,
        metrics=result.metrics,
        feature_importance=result.feature_importance,
        predictions=result.predictions,
        confusion_matrix=result.confusion_matrix,
        roc_curve=result.roc_curve,
        cv_scores=result.cv_scores,
        training_status=result.training_status,
        n_samples=result.n_samples,
        n_features=result.n_features,
        tags=result.tags or [],
        notes=result.notes,
        created_at=result.created_at.isoformat() if result.created_at else "",
        updated_at=result.updated_at.isoformat() if result.updated_at else ""
    )


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_training_result(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a training result.
    """
    result = db.query(MLTrainingResult).filter(
        MLTrainingResult.model_id == model_id,
        MLTrainingResult.created_by == current_user.id
    ).first()
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training result not found"
        )
    
    db.delete(result)
    db.commit()
    
    return None


@router.get("/{model_id}/download")
def download_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Download the trained model artifact (pickle file).
    """
    from fastapi.responses import FileResponse
    from pathlib import Path
    
    result = db.query(MLTrainingResult).filter(
        MLTrainingResult.model_id == model_id,
        MLTrainingResult.created_by == current_user.id
    ).first()
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found or access denied"
        )
    
    if not result.model_artifact_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model artifact not available for this training result"
        )
    
    model_path = Path(result.model_artifact_path)
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model file not found on server"
        )
    
    # Use the original filename from model_artifact_path for download
    # This preserves the descriptive name (algorithm_target_timestamp.ext)
    original_filename = model_path.name
    
    return FileResponse(
        path=str(model_path),
        filename=original_filename,
        media_type="application/octet-stream"
    )


@router.get("/stats/summary")
def get_training_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get summary statistics for user's training results.
    """
    results = db.query(MLTrainingResult).filter(
        MLTrainingResult.created_by == current_user.id
    ).all()
    
    total_models = len(results)
    algorithms = {}
    status_counts = {}
    
    for r in results:
        # Count by algorithm
        algorithms[r.algorithm] = algorithms.get(r.algorithm, 0) + 1
        
        # Count by status
        status_counts[r.training_status] = status_counts.get(r.training_status, 0) + 1
    
    return {
        "total_models": total_models,
        "algorithms": algorithms,
        "status_counts": status_counts,
        "latest_training": results[0].created_at.isoformat() if results else None
    }

