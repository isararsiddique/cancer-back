"""
Training Job Queue Service — Priority-based job scheduling for GPU training.

Inspired by CDCE compute-orchestration: manages a queue of training jobs,
allocates GPU resources, runs jobs in background threads, and tracks status.

Features:
- Priority queue (1-10, higher = more urgent)
- Concurrent job limit based on available GPUs
- Background execution with status polling
- Job lifecycle: queued -> running -> completed/failed
- Automatic retry on transient failures
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from services.gpu_manager import get_gpu_manager
from services.offline_training import run_offline_training, TrainingResult

logger = logging.getLogger(__name__)


# ─── Enums & Data Classes ────────────────────────────────────────────────────


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Represents a training job in the queue."""
    id: str
    user_id: str
    algorithm: str
    target_variable: str
    features: List[str]
    dataset: List[Dict[str, Any]]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    test_size: float = 0.2
    random_state: int = 42
    force_gpu: bool = False
    force_cpu: bool = False
    priority: int = 5  # 1-10, higher = more urgent
    status: JobStatus = JobStatus.QUEUED
    result: Optional[TrainingResult] = None
    error_message: Optional[str] = None
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress_pct: int = 0
    retry_count: int = 0
    max_retries: int = 1

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "algorithm": self.algorithm,
            "target_variable": self.target_variable,
            "features": self.features,
            "hyperparameters": self.hyperparameters,
            "test_size": self.test_size,
            "priority": self.priority,
            "status": self.status.value,
            "force_gpu": self.force_gpu,
            "force_cpu": self.force_cpu,
            "result": self.result.to_dict() if self.result else None,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress_pct": self.progress_pct,
            "retry_count": self.retry_count,
            "dataset_size": len(self.dataset),
            "n_features": len(self.features),
        }

    def to_summary(self) -> Dict[str, Any]:
        """Lightweight summary without dataset or full result."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "algorithm": self.algorithm,
            "target_variable": self.target_variable,
            "priority": self.priority,
            "status": self.status.value,
            "force_gpu": self.force_gpu,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress_pct": self.progress_pct,
            "dataset_size": len(self.dataset),
            "n_features": len(self.features),
            "metrics": self.result.metrics if self.result else None,
            "error_message": self.error_message,
            "duration_seconds": self.result.duration_seconds if self.result else None,
            "device_used": self.result.device_used if self.result else None,
        }


# ─── Training Queue Manager (Singleton) ──────────────────────────────────────


class TrainingQueue:
    """
    Manages a priority queue of training jobs with background execution.
    Thread-safe singleton that runs jobs on available GPU/CPU resources.
    """

    _instance: Optional["TrainingQueue"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TrainingQueue":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._jobs: OrderedDict[str, TrainingJob] = OrderedDict()
        self._queue_lock = threading.Lock()
        self._max_concurrent = 2  # max parallel jobs
        self._running_count = 0
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_concurrent,
            thread_name_prefix="gpu-train"
        )
        self._processing = False
        self._processor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Start the queue processor
        self._start_processor()

    def _start_processor(self):
        """Start background thread that processes queued jobs."""
        if self._processor_thread and self._processor_thread.is_alive():
            return
        self._stop_event.clear()
        self._processor_thread = threading.Thread(
            target=self._process_queue_loop, daemon=True, name="queue-processor"
        )
        self._processor_thread.start()
        logger.info("Training queue processor started")

    def _process_queue_loop(self):
        """Continuously check for queued jobs and dispatch them."""
        while not self._stop_event.is_set():
            try:
                self._dispatch_next()
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
            time.sleep(2)  # Check every 2 seconds

    def _dispatch_next(self):
        """Find the highest-priority queued job and run it if capacity allows."""
        with self._queue_lock:
            if self._running_count >= self._max_concurrent:
                return

            # Find highest priority queued job
            queued_jobs = [
                j for j in self._jobs.values() if j.status == JobStatus.QUEUED
            ]
            if not queued_jobs:
                return

            # Sort by priority (desc) then by created_at (asc)
            queued_jobs.sort(key=lambda j: (-j.priority, j.created_at))
            job = queued_jobs[0]

            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc).isoformat()
            job.progress_pct = 5
            self._running_count += 1

        # Submit to thread pool
        self._executor.submit(self._execute_job, job)

    def _execute_job(self, job: TrainingJob):
        """Execute a training job in a worker thread."""
        try:
            logger.info(f"[{job.id}] Executing training job: {job.algorithm}")
            job.progress_pct = 10

            result = run_offline_training(
                job_id=job.id,
                dataset=job.dataset,
                target_variable=job.target_variable,
                features=job.features,
                algorithm=job.algorithm,
                hyperparameters=job.hyperparameters,
                test_size=job.test_size,
                random_state=job.random_state,
                force_gpu=job.force_gpu,
                force_cpu=job.force_cpu,
            )

            if result.error:
                # Check for retry
                if job.retry_count < job.max_retries:
                    job.retry_count += 1
                    job.status = JobStatus.QUEUED
                    job.started_at = None
                    job.progress_pct = 0
                    logger.warning(f"[{job.id}] Retrying ({job.retry_count}/{job.max_retries}): {result.error}")
                else:
                    job.status = JobStatus.FAILED
                    job.error_message = result.error
                    job.completed_at = datetime.now(timezone.utc).isoformat()
                    job.progress_pct = 100
                    logger.error(f"[{job.id}] Training failed: {result.error}")
            else:
                job.status = JobStatus.COMPLETED
                job.result = result
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.progress_pct = 100
                logger.info(f"[{job.id}] Training completed in {result.duration_seconds}s")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.progress_pct = 100
            logger.error(f"[{job.id}] Job execution error: {e}", exc_info=True)
        finally:
            with self._queue_lock:
                self._running_count -= 1


    # ─── Public API ──────────────────────────────────────────────────────────

    def submit_job(
        self,
        user_id: str,
        algorithm: str,
        target_variable: str,
        features: List[str],
        dataset: List[Dict[str, Any]],
        hyperparameters: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        priority: int = 5,
        force_gpu: bool = False,
        force_cpu: bool = False,
    ) -> TrainingJob:
        """Submit a new training job to the queue."""
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            user_id=user_id,
            algorithm=algorithm,
            target_variable=target_variable,
            features=features,
            dataset=dataset,
            hyperparameters=hyperparameters or {},
            test_size=test_size,
            random_state=random_state,
            priority=max(1, min(10, priority)),
            force_gpu=force_gpu,
            force_cpu=force_cpu,
        )
        with self._queue_lock:
            self._jobs[job_id] = job
        logger.info(f"[{job_id}] Job submitted: {algorithm}, priority={priority}, "
                    f"samples={len(dataset)}")
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status summary."""
        job = self._jobs.get(job_id)
        if not job:
            return None
        return job.to_summary()

    def list_jobs(
        self,
        user_id: Optional[str] = None,
        status_filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List jobs with optional filtering."""
        jobs = list(self._jobs.values())
        if user_id:
            jobs = [j for j in jobs if j.user_id == user_id]
        if status_filter:
            jobs = [j for j in jobs if j.status.value == status_filter]
        # Sort by created_at descending (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_summary() for j in jobs[:limit]]

    def cancel_job(self, job_id: str, user_id: str) -> bool:
        """Cancel a queued job (cannot cancel running jobs)."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.user_id != user_id:
            return False
        if job.status != JobStatus.QUEUED:
            return False
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc).isoformat()
        return True

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        jobs = list(self._jobs.values())
        gpu_mgr = get_gpu_manager()
        return {
            "total_jobs": len(jobs),
            "queued": sum(1 for j in jobs if j.status == JobStatus.QUEUED),
            "running": sum(1 for j in jobs if j.status == JobStatus.RUNNING),
            "completed": sum(1 for j in jobs if j.status == JobStatus.COMPLETED),
            "failed": sum(1 for j in jobs if j.status == JobStatus.FAILED),
            "cancelled": sum(1 for j in jobs if j.status == JobStatus.CANCELLED),
            "max_concurrent": self._max_concurrent,
            "gpu_available": gpu_mgr.has_gpu,
            "gpu_count": gpu_mgr.gpu_count,
        }

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove completed/failed jobs older than max_age_hours."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        with self._queue_lock:
            to_remove = []
            for job_id, job in self._jobs.items():
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                    if job.completed_at:
                        completed = datetime.fromisoformat(job.completed_at.replace("Z", "+00:00"))
                        if completed < cutoff:
                            to_remove.append(job_id)
            for jid in to_remove:
                del self._jobs[jid]
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old training jobs")

    def shutdown(self):
        """Gracefully shut down the queue processor."""
        self._stop_event.set()
        self._executor.shutdown(wait=False)
        logger.info("Training queue shut down")


# ─── Module-level accessor ───────────────────────────────────────────────────


def get_training_queue() -> TrainingQueue:
    """Get or create the singleton training queue instance."""
    return TrainingQueue()
