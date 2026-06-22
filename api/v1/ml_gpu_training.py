"""
ML GPU Training API — Offline model training with local GPU acceleration.

Endpoints:
- POST /gpu-training/submit       Submit a training job to the GPU queue
- GET  /gpu-training/jobs         List training jobs
- GET  /gpu-training/jobs/{id}    Get job status and result
- POST /gpu-training/jobs/{id}/cancel  Cancel a queued job
- GET  /gpu-training/gpu/status   GPU hardware status and utilization
- GET  /gpu-training/gpu/devices  List all GPU devices
- GET  /gpu-training/queue/stats  Queue statistics
- GET  /gpu-training/algorithms   List supported algorithms
- GET  /gpu-training/models/{id}/download  Download trained model artifact
"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
import logging

from core.deps import get_current_user
from db.models.users import User
from services.gpu_manager import get_gpu_manager
from services.training_queue import get_training_queue
from services.offline_training import SUPPORTED_ALGORITHMS, ALL_ALGORITHMS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gpu-training", tags=["gpu-training"])


# ─── Pydantic Models ─────────────────────────────────────────────────────────


class GPUTrainingRequest(BaseModel):
    """Submit a GPU training job."""
    algorithm: str = Field(
        ...,
        description="Algorithm to use. Options: pytorch_neural_network, pytorch_deep, "
                    "tensorflow_neural_network, tensorflow_cnn, xgboost, xgboost_gpu, "
                    "random_forest, gradient_boosting, extra_trees, logistic_regression, svm, ridge"
    )
    target_variable: str = Field(..., description="Target column to predict")
    features: List[str] = Field(..., min_length=1, description="Feature columns")
    dataset: List[Dict[str, Any]] = Field(..., min_length=10, description="Training data as list of records")
    hyperparameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Model hyperparameters. Examples: epochs, learning_rate, hidden_layers, "
                    "batch_size, dropout, n_estimators, max_depth"
    )
    test_size: float = Field(0.2, ge=0.05, le=0.5, description="Test split ratio")
    random_state: int = Field(42, description="Random seed for reproducibility")
    priority: int = Field(5, ge=1, le=10, description="Job priority (1-10, higher = more urgent)")
    force_gpu: bool = Field(False, description="Require GPU (fail if not available)")
    force_cpu: bool = Field(False, description="Force CPU even if GPU available")


class GPUTrainingResponse(BaseModel):
    """Response after submitting a training job."""
    job_id: str
    status: str
    message: str
    queue_position: Optional[int] = None
    estimated_wait_seconds: Optional[int] = None


class JobStatusResponse(BaseModel):
    """Detailed job status response."""
    id: str
    user_id: str
    algorithm: str
    target_variable: str
    priority: int
    status: str
    force_gpu: bool
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress_pct: int
    dataset_size: int
    n_features: int
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    device_used: Optional[str] = None


# ─── Helper ──────────────────────────────────────────────────────────────────


def _require_researcher(current_user: User):
    """Only researchers, super_admin, or ummc_admin can use GPU training."""
    roles = [r.slug for r in current_user.roles]
    if not any(r in ("researcher", "super_admin", "ummc_admin") for r in roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="GPU training is only available to researchers and admins",
        )


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.post("/submit", response_model=GPUTrainingResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_training_job(
    req: GPUTrainingRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Submit a model training job to the GPU queue.

    The job is queued and executed asynchronously on the server's GPU (or CPU fallback).
    Poll GET /gpu-training/jobs/{job_id} for status updates.
    """
    _require_researcher(current_user)

    if req.algorithm not in ALL_ALGORITHMS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown algorithm '{req.algorithm}'. Supported: {ALL_ALGORITHMS}",
        )

    if req.force_gpu and req.force_cpu:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot set both force_gpu and force_cpu",
        )

    queue = get_training_queue()
    job = queue.submit_job(
        user_id=str(current_user.id),
        algorithm=req.algorithm,
        target_variable=req.target_variable,
        features=req.features,
        dataset=req.dataset,
        hyperparameters=req.hyperparameters,
        test_size=req.test_size,
        random_state=req.random_state,
        priority=req.priority,
        force_gpu=req.force_gpu,
        force_cpu=req.force_cpu,
    )

    # Calculate queue position
    stats = queue.get_queue_stats()
    queue_position = stats["queued"]
    est_wait = queue_position * 30  # rough estimate

    gpu_mgr = get_gpu_manager()
    device_info = "GPU" if gpu_mgr.has_gpu and not req.force_cpu else "CPU"

    return GPUTrainingResponse(
        job_id=job.id,
        status=job.status.value,
        message=f"Training job submitted. Will run on {device_info}. "
                f"Algorithm: {req.algorithm}, Dataset: {len(req.dataset)} rows.",
        queue_position=queue_position,
        estimated_wait_seconds=est_wait,
    )


@router.get("/jobs", response_model=List[JobStatusResponse])
def list_training_jobs(
    status_filter: Optional[str] = Query(None, description="Filter: queued, running, completed, failed"),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
):
    """List training jobs for the current user."""
    queue = get_training_queue()
    jobs = queue.list_jobs(user_id=str(current_user.id), status_filter=status_filter, limit=limit)
    return jobs


@router.get("/jobs/{job_id}")
def get_training_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Get detailed status and results for a training job."""
    queue = get_training_queue()
    job = queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.user_id != str(current_user.id):
        # Allow admins to view any job
        roles = [r.slug for r in current_user.roles]
        if "super_admin" not in roles:
            raise HTTPException(status_code=403, detail="Access denied")
    return job.to_dict()


@router.post("/jobs/{job_id}/cancel")
def cancel_training_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Cancel a queued training job. Running jobs cannot be cancelled."""
    queue = get_training_queue()
    success = queue.cancel_job(job_id, str(current_user.id))
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot cancel: job not found, not yours, or already running",
        )
    return {"message": "Job cancelled", "job_id": job_id}


@router.get("/gpu/status")
def get_gpu_status(current_user: User = Depends(get_current_user)):
    """
    Get GPU hardware status: available devices, memory, utilization, temperature.
    Returns system summary even if no GPU is present (reports CPU-only mode).
    """
    gpu_mgr = get_gpu_manager()
    summary = gpu_mgr.get_system_summary()
    summary["server_mode"] = "gpu" if gpu_mgr.has_gpu else "cpu_only"
    return summary


@router.get("/gpu/devices")
def get_gpu_devices(current_user: User = Depends(get_current_user)):
    """List all GPU devices with detailed specs."""
    gpu_mgr = get_gpu_manager()
    devices = gpu_mgr.get_all_devices()
    if not devices:
        return {
            "devices": [],
            "message": "No NVIDIA GPU detected. Training will use CPU.",
            "recommendation": "For GPU acceleration, deploy on a server with NVIDIA GPU and CUDA toolkit.",
        }
    return {"devices": [d.to_dict() for d in devices]}


@router.get("/queue/stats")
def get_queue_stats(current_user: User = Depends(get_current_user)):
    """Get training queue statistics."""
    queue = get_training_queue()
    stats = queue.get_queue_stats()
    gpu_mgr = get_gpu_manager()

    stats["system"] = {
        "server_mode": "gpu" if gpu_mgr.has_gpu else "cpu_only",
        "torch_cuda": gpu_mgr.torch_cuda_available,
        "tensorflow_gpu": gpu_mgr.tensorflow_gpu_available,
    }
    return stats


@router.get("/algorithms")
def list_algorithms(current_user: User = Depends(get_current_user)):
    """List all supported training algorithms grouped by framework."""
    gpu_mgr = get_gpu_manager()
    return {
        "algorithms": SUPPORTED_ALGORITHMS,
        "all_algorithms": ALL_ALGORITHMS,
        "gpu_available": gpu_mgr.has_gpu,
        "gpu_accelerated": ["pytorch_neural_network", "pytorch_deep",
                            "tensorflow_neural_network", "tensorflow_cnn",
                            "xgboost_gpu"],
        "cpu_only": ["random_forest", "gradient_boosting", "extra_trees",
                     "logistic_regression", "svm", "ridge"],
        "recommended": "pytorch_neural_network" if gpu_mgr.has_gpu else "xgboost",
        "hyperparameter_guide": {
            "pytorch_neural_network": {
                "epochs": "100 (default), training iterations",
                "hidden_layers": "[128, 64, 32] (default), list of layer sizes",
                "learning_rate": "0.001 (default)",
                "batch_size": "64 (default)",
                "dropout": "0.3 (default), regularization",
                "early_stop_patience": "15 (default)",
                "weight_decay": "0.0001 (default), L2 regularization",
            },
            "xgboost_gpu": {
                "n_estimators": "500 (default), number of trees",
                "max_depth": "8 (default), tree depth",
                "learning_rate": "0.05 (default)",
                "subsample": "0.8 (default)",
                "colsample_bytree": "0.8 (default)",
                "early_stopping_rounds": "20 (default)",
            },
            "tensorflow_neural_network": {
                "epochs": "100 (default)",
                "hidden_layers": "[128, 64, 32] (default)",
                "learning_rate": "0.001 (default)",
                "batch_size": "64 (default)",
                "dropout": "0.3 (default)",
            },
            "random_forest": {
                "n_estimators": "300 (default)",
                "max_depth": "None (default, unlimited)",
            },
        },
    }


@router.get("/models/{job_id}/download")
def download_trained_model(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Download the trained model artifact for a completed job."""
    _require_researcher(current_user)
    queue = get_training_queue()
    job = queue.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.user_id != str(current_user.id):
        roles = [r.slug for r in current_user.roles]
        if "super_admin" not in roles:
            raise HTTPException(status_code=403, detail="Access denied")

    if not job.result or not job.result.model_artifact_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No model artifact available (job may not be completed)",
        )

    artifact_path = Path(job.result.model_artifact_path)
    if not artifact_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model file not found on server",
        )

    return FileResponse(
        path=str(artifact_path),
        filename=artifact_path.name,
        media_type="application/octet-stream",
    )


@router.get("/models/{job_id}/metrics")
def get_model_metrics(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Get detailed metrics and training history for a completed job."""
    queue = get_training_queue()
    job = queue.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.user_id != str(current_user.id):
        roles = [r.slug for r in current_user.roles]
        if "super_admin" not in roles:
            raise HTTPException(status_code=403, detail="Access denied")

    if not job.result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not completed yet (status: {job.status.value})",
        )

    return {
        "job_id": job_id,
        "algorithm": job.result.algorithm,
        "framework": job.result.framework,
        "device_used": job.result.device_used,
        "metrics": job.result.metrics,
        "feature_importance": job.result.feature_importance,
        "training_history": job.result.training_history,
        "hyperparameters": job.result.hyperparameters,
        "duration_seconds": job.result.duration_seconds,
        "gpu_memory_used_mb": job.result.gpu_memory_used_mb,
        "n_samples": job.result.n_samples,
        "n_features": job.result.n_features,
        "n_train": job.result.n_train,
        "n_test": job.result.n_test,
        "model_artifact": {
            "path": job.result.model_artifact_path,
            "size_bytes": job.result.model_artifact_size,
            "sha256": job.result.model_artifact_hash,
        },
    }
