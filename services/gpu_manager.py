"""
GPU Manager Service — Detect, monitor, and allocate local GPU resources.

Provides:
- NVIDIA GPU detection via pynvml (nvidia-ml-py3)
- Real-time utilization metrics (memory, compute, temperature)
- GPU allocation tracking for training jobs
- Fallback to CPU when no GPU is available
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── GPU Info Dataclass ──────────────────────────────────────────────────────


@dataclass
class GPUDevice:
    index: int
    name: str
    uuid: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_gpu: int  # percentage 0-100
    utilization_memory: int  # percentage 0-100
    temperature: int  # Celsius
    power_draw_watts: float
    power_limit_watts: float
    compute_capability: str
    driver_version: str
    cuda_version: str

    @property
    def memory_utilization_pct(self) -> float:
        if self.memory_total_mb == 0:
            return 0.0
        return round(self.memory_used_mb / self.memory_total_mb * 100, 1)

    @property
    def is_available(self) -> bool:
        """GPU is considered available if <80% memory used and <90% compute util."""
        return self.memory_utilization_pct < 80 and self.utilization_gpu < 90

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "uuid": self.uuid,
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
            "memory_free_mb": self.memory_free_mb,
            "memory_utilization_pct": self.memory_utilization_pct,
            "utilization_gpu_pct": self.utilization_gpu,
            "utilization_memory_pct": self.utilization_memory,
            "temperature_celsius": self.temperature,
            "power_draw_watts": self.power_draw_watts,
            "power_limit_watts": self.power_limit_watts,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "is_available": self.is_available,
        }


@dataclass
class GPUAllocation:
    gpu_index: int
    job_id: str
    allocated_at: str
    estimated_memory_mb: int = 0


# ─── GPU Manager Singleton ───────────────────────────────────────────────────


class GPUManager:
    """Singleton GPU resource manager."""

    _instance: Optional["GPUManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "GPUManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._nvml_available = False
        self._gpu_count = 0
        self._allocations: Dict[str, GPUAllocation] = {}  # job_id -> allocation
        self._torch_available = False
        self._tf_available = False
        self._init_backends()

    def _init_backends(self):
        """Initialize GPU backends (NVML, PyTorch, TensorFlow)."""
        # Try NVML for GPU monitoring
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_count = pynvml.nvmlDeviceGetCount()
            self._nvml_available = True
            logger.info(f"NVML initialized: {self._gpu_count} GPU(s) detected")
        except Exception as e:
            logger.info(f"NVML not available (no NVIDIA GPU or driver): {e}")
            self._nvml_available = False
            self._gpu_count = 0

        # Check PyTorch CUDA
        try:
            import torch
            self._torch_available = torch.cuda.is_available()
            if self._torch_available:
                logger.info(f"PyTorch CUDA available: {torch.cuda.device_count()} device(s)")
        except ImportError:
            self._torch_available = False

        # Check TensorFlow GPU
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            self._tf_available = len(gpus) > 0
            if self._tf_available:
                logger.info(f"TensorFlow GPU available: {len(gpus)} device(s)")
        except ImportError:
            self._tf_available = False

    @property
    def has_gpu(self) -> bool:
        return self._gpu_count > 0

    @property
    def gpu_count(self) -> int:
        return self._gpu_count

    @property
    def torch_cuda_available(self) -> bool:
        return self._torch_available

    @property
    def tensorflow_gpu_available(self) -> bool:
        return self._tf_available

    def get_device_info(self, index: int = 0) -> Optional[GPUDevice]:
        """Get detailed info for a specific GPU."""
        if not self._nvml_available or index >= self._gpu_count:
            return None
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            uuid_str = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid_str, bytes):
                uuid_str = uuid_str.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                power_draw = 0.0
            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except Exception:
                power_limit = 0.0

            # Compute capability
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_cap = f"{major}.{minor}"
            except Exception:
                compute_cap = "unknown"

            # Driver and CUDA version
            try:
                driver_ver = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver_ver, bytes):
                    driver_ver = driver_ver.decode("utf-8")
            except Exception:
                driver_ver = "unknown"
            try:
                cuda_ver = pynvml.nvmlSystemGetCudaDriverVersion_v2()
                cuda_major = cuda_ver // 1000
                cuda_minor = (cuda_ver % 1000) // 10
                cuda_ver_str = f"{cuda_major}.{cuda_minor}"
            except Exception:
                cuda_ver_str = "unknown"

            return GPUDevice(
                index=index,
                name=name,
                uuid=uuid_str,
                memory_total_mb=mem_info.total // (1024 * 1024),
                memory_used_mb=mem_info.used // (1024 * 1024),
                memory_free_mb=mem_info.free // (1024 * 1024),
                utilization_gpu=util.gpu,
                utilization_memory=util.memory,
                temperature=temp,
                power_draw_watts=power_draw,
                power_limit_watts=power_limit,
                compute_capability=compute_cap,
                driver_version=driver_ver,
                cuda_version=cuda_ver_str,
            )
        except Exception as e:
            logger.error(f"Failed to query GPU {index}: {e}")
            return None

    def get_all_devices(self) -> List[GPUDevice]:
        """Get info for all GPUs."""
        devices = []
        for i in range(self._gpu_count):
            dev = self.get_device_info(i)
            if dev:
                devices.append(dev)
        return devices

    def get_system_summary(self) -> Dict[str, Any]:
        """Get overall GPU system summary."""
        devices = self.get_all_devices()
        total_memory = sum(d.memory_total_mb for d in devices)
        used_memory = sum(d.memory_used_mb for d in devices)
        available_gpus = [d for d in devices if d.is_available]

        return {
            "gpu_count": self._gpu_count,
            "has_gpu": self.has_gpu,
            "torch_cuda_available": self._torch_available,
            "tensorflow_gpu_available": self._tf_available,
            "total_memory_mb": total_memory,
            "used_memory_mb": used_memory,
            "free_memory_mb": total_memory - used_memory,
            "available_gpu_count": len(available_gpus),
            "active_training_jobs": len(self._allocations),
            "devices": [d.to_dict() for d in devices],
            "allocations": [
                {
                    "job_id": a.job_id,
                    "gpu_index": a.gpu_index,
                    "allocated_at": a.allocated_at,
                    "estimated_memory_mb": a.estimated_memory_mb,
                }
                for a in self._allocations.values()
            ],
        }

    def allocate_gpu(self, job_id: str, estimated_memory_mb: int = 0) -> Optional[int]:
        """
        Allocate the best available GPU for a training job.
        Returns the GPU index or None if no GPU available.
        Allocation strategy: pick the GPU with the most free memory.
        """
        if not self.has_gpu:
            return None

        devices = self.get_all_devices()
        available = [d for d in devices if d.is_available]
        if not available:
            # All GPUs busy — check if any have enough free memory
            available = [d for d in devices if d.memory_free_mb > estimated_memory_mb]

        if not available:
            return None

        # Pick the one with the most free memory
        best = max(available, key=lambda d: d.memory_free_mb)
        allocation = GPUAllocation(
            gpu_index=best.index,
            job_id=job_id,
            allocated_at=datetime.now(timezone.utc).isoformat(),
            estimated_memory_mb=estimated_memory_mb,
        )
        self._allocations[job_id] = allocation
        logger.info(f"GPU {best.index} allocated to job {job_id} (free: {best.memory_free_mb}MB)")
        return best.index

    def release_gpu(self, job_id: str) -> bool:
        """Release GPU allocation for a completed/failed job."""
        if job_id in self._allocations:
            alloc = self._allocations.pop(job_id)
            logger.info(f"GPU {alloc.gpu_index} released from job {job_id}")
            return True
        return False

    def get_recommended_device(self, framework: str = "pytorch") -> str:
        """
        Get the recommended device string for a training framework.
        Returns 'cuda:0', 'cuda:1', etc. or 'cpu'.
        """
        if framework == "pytorch":
            if self._torch_available and self.has_gpu:
                # Find least loaded GPU
                devices = self.get_all_devices()
                available = [d for d in devices if d.is_available]
                if available:
                    best = max(available, key=lambda d: d.memory_free_mb)
                    return f"cuda:{best.index}"
            return "cpu"
        elif framework == "tensorflow":
            if self._tf_available and self.has_gpu:
                return "/GPU:0"
            return "/CPU:0"
        return "cpu"


# ─── Module-level accessor ───────────────────────────────────────────────────

def get_gpu_manager() -> GPUManager:
    """Get or create the singleton GPU manager instance."""
    return GPUManager()
