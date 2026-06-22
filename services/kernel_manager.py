"""
Researcher Kernel Manager Service

Manages dedicated Jupyter kernels for each researcher with 24-hour auto-cleanup.
Each researcher gets an isolated kernel environment that persists their session state.

Features:
- Dedicated kernel per researcher (isolated environment)
- 24-hour automatic cleanup
- Session state persistence (variables, imports)
- Resource limits per kernel
- Background cleanup scheduler
"""

import docker
import threading
import time
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import redis
import uuid

logger = logging.getLogger(__name__)

# Redis key prefixes
KERNEL_PREFIX = "kernel:"
KERNEL_LIST_KEY = "kernels:active"


@dataclass
class KernelInfo:
    """Information about a researcher's kernel"""
    kernel_id: str
    user_id: str
    user_email: str
    container_id: str
    created_at: str
    expires_at: str
    last_activity: str
    status: str  # 'running', 'stopped', 'expired'
    memory_limit: str
    cpu_limit: float
    execution_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KernelInfo':
        return cls(**data)
    
    def is_expired(self) -> bool:
        expires = datetime.fromisoformat(self.expires_at)
        return datetime.utcnow() > expires


class KernelManager:
    """
    Manages dedicated Jupyter kernels for researchers.
    
    Each researcher gets their own isolated Docker container that persists
    for 24 hours, maintaining session state between code executions.
    """
    
    def __init__(
        self,
        image_name: str = "ml-sandbox:latest",
        kernel_lifetime_hours: int = 24,
        memory_limit: str = "4g",
        cpu_limit: float = 2.0,
        redis_url: str = None,
        cleanup_interval_minutes: int = 15
    ):
        """
        Initialize the Kernel Manager.
        
        Args:
            image_name: Docker image for kernels
            kernel_lifetime_hours: How long kernels live (default 24 hours)
            memory_limit: Memory limit per kernel
            cpu_limit: CPU cores per kernel
            redis_url: Redis URL for kernel state storage
            cleanup_interval_minutes: How often to run cleanup
        """
        self.image_name = image_name
        self.kernel_lifetime_hours = kernel_lifetime_hours
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.cleanup_interval = cleanup_interval_minutes * 60
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized for kernel manager")
        except Exception as e:
            logger.error(f"Failed to initialize Docker: {e}")
            self.docker_client = None
        
        # Initialize Redis for kernel state
        redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        try:
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self.redis.ping()
            logger.info("Redis connected for kernel state storage")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory storage: {e}")
            self.redis = None
            self._memory_store: Dict[str, KernelInfo] = {}
        
        # Start background cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_scheduler()
    
    def _start_cleanup_scheduler(self):
        """Start background thread for automatic kernel cleanup"""
        def cleanup_loop():
            while not self._stop_cleanup.is_set():
                try:
                    self._cleanup_expired_kernels()
                except Exception as e:
                    logger.error(f"Error in cleanup loop: {e}")
                self._stop_cleanup.wait(self.cleanup_interval)
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.info(f"Kernel cleanup scheduler started (interval: {self.cleanup_interval}s)")
    
    def _get_kernel_key(self, user_id: str) -> str:
        """Get Redis key for a user's kernel"""
        return f"{KERNEL_PREFIX}{user_id}"
    
    def _save_kernel_info(self, kernel: KernelInfo):
        """Save kernel info to storage"""
        if self.redis:
            key = self._get_kernel_key(kernel.user_id)
            self.redis.set(key, json.dumps(kernel.to_dict()))
            self.redis.sadd(KERNEL_LIST_KEY, kernel.user_id)
            # Set expiry slightly after kernel expiry for cleanup
            ttl = self.kernel_lifetime_hours * 3600 + 3600
            self.redis.expire(key, ttl)
        else:
            self._memory_store[kernel.user_id] = kernel
    
    def _get_kernel_info(self, user_id: str) -> Optional[KernelInfo]:
        """Get kernel info from storage"""
        if self.redis:
            key = self._get_kernel_key(user_id)
            data = self.redis.get(key)
            if data:
                return KernelInfo.from_dict(json.loads(data))
        else:
            return self._memory_store.get(user_id)
        return None
    
    def _delete_kernel_info(self, user_id: str):
        """Delete kernel info from storage"""
        if self.redis:
            key = self._get_kernel_key(user_id)
            self.redis.delete(key)
            self.redis.srem(KERNEL_LIST_KEY, user_id)
        else:
            self._memory_store.pop(user_id, None)
    
    def _get_all_kernel_user_ids(self) -> List[str]:
        """Get all user IDs with active kernels"""
        if self.redis:
            return list(self.redis.smembers(KERNEL_LIST_KEY))
        else:
            return list(self._memory_store.keys())
    
    def get_or_create_kernel(
        self,
        user_id: str,
        user_email: str
    ) -> Dict[str, Any]:
        """
        Get existing kernel or create new one for researcher.
        
        Args:
            user_id: Unique user identifier
            user_email: User's email for logging
        
        Returns:
            Dictionary with kernel info and status
        """
        # Check for existing kernel
        existing = self._get_kernel_info(user_id)
        
        if existing and not existing.is_expired():
            # Verify container is still running
            try:
                container = self.docker_client.containers.get(existing.container_id)
                if container.status == 'running':
                    # Update last activity
                    existing.last_activity = datetime.utcnow().isoformat()
                    self._save_kernel_info(existing)
                    
                    return {
                        'success': True,
                        'kernel_id': existing.kernel_id,
                        'status': 'existing',
                        'created_at': existing.created_at,
                        'expires_at': existing.expires_at,
                        'message': 'Using existing kernel session'
                    }
            except docker.errors.NotFound:
                # Container gone, clean up and create new
                self._delete_kernel_info(user_id)
            except Exception as e:
                logger.error(f"Error checking container: {e}")
        
        # Create new kernel
        return self._create_kernel(user_id, user_email)
    
    def _create_kernel(self, user_id: str, user_email: str) -> Dict[str, Any]:
        """Create a new dedicated kernel for researcher"""
        if not self.docker_client:
            return {
                'success': False,
                'error': 'Docker not available'
            }
        
        kernel_id = f"kernel-{user_id[:8]}-{uuid.uuid4().hex[:8]}"
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=self.kernel_lifetime_hours)
        
        try:
            # Create persistent container for this researcher
            container = self.docker_client.containers.run(
                image=self.image_name,
                name=kernel_id,
                command=['tail', '-f', '/dev/null'],  # Keep container running
                detach=True,
                mem_limit=self.memory_limit,
                nano_cpus=int(self.cpu_limit * 1e9),
                network_mode='none',  # Network isolation
                user='1000:1000',
                security_opt=['no-new-privileges'],
                labels={
                    'kernel.user_id': user_id,
                    'kernel.user_email': user_email,
                    'kernel.expires_at': expires_at.isoformat(),
                    'kernel.type': 'researcher-kernel'
                },
                tmpfs={
                    '/tmp': 'size=1G,mode=1777',
                    '/home/sandbox': 'size=2G,mode=1777'
                },
                environment={
                    'KERNEL_ID': kernel_id,
                    'USER_ID': user_id,
                    'PYTHONUNBUFFERED': '1'
                }
            )
            
            # Save kernel info
            kernel_info = KernelInfo(
                kernel_id=kernel_id,
                user_id=user_id,
                user_email=user_email,
                container_id=container.id,
                created_at=now.isoformat(),
                expires_at=expires_at.isoformat(),
                last_activity=now.isoformat(),
                status='running',
                memory_limit=self.memory_limit,
                cpu_limit=self.cpu_limit,
                execution_count=0
            )
            self._save_kernel_info(kernel_info)
            
            logger.info(f"Created kernel {kernel_id} for user {user_email}")
            
            return {
                'success': True,
                'kernel_id': kernel_id,
                'status': 'created',
                'created_at': now.isoformat(),
                'expires_at': expires_at.isoformat(),
                'lifetime_hours': self.kernel_lifetime_hours,
                'message': f'New kernel created. Valid for {self.kernel_lifetime_hours} hours.'
            }
            
        except docker.errors.ImageNotFound:
            return {
                'success': False,
                'error': f"Docker image '{self.image_name}' not found"
            }
        except Exception as e:
            logger.error(f"Failed to create kernel: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def execute_in_kernel(
        self,
        user_id: str,
        code: str,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Execute code in researcher's dedicated kernel.
        
        Args:
            user_id: User's ID
            code: Python code to execute
            timeout: Execution timeout in seconds
        
        Returns:
            Execution result with stdout, stderr, etc.
        """
        kernel_info = self._get_kernel_info(user_id)
        
        if not kernel_info:
            return {
                'success': False,
                'error': 'No kernel found. Please create a kernel first.',
                'stdout': '',
                'stderr': '',
                'execution_time': 0
            }
        
        if kernel_info.is_expired():
            self._cleanup_kernel(user_id)
            return {
                'success': False,
                'error': 'Kernel has expired. Please create a new kernel.',
                'stdout': '',
                'stderr': '',
                'execution_time': 0
            }
        
        try:
            container = self.docker_client.containers.get(kernel_info.container_id)
            
            if container.status != 'running':
                container.start()
                time.sleep(0.5)
            
            start_time = time.time()
            
            # Execute code in the persistent container
            exec_result = container.exec_run(
                cmd=['python3', '-c', code],
                user='1000:1000',
                workdir='/home/sandbox',
                demux=True,
                environment={'PYTHONUNBUFFERED': '1'}
            )
            
            execution_time = time.time() - start_time
            
            stdout = exec_result.output[0].decode('utf-8') if exec_result.output[0] else ''
            stderr = exec_result.output[1].decode('utf-8') if exec_result.output[1] else ''
            
            # Update kernel info
            kernel_info.last_activity = datetime.utcnow().isoformat()
            kernel_info.execution_count += 1
            self._save_kernel_info(kernel_info)
            
            return {
                'success': exec_result.exit_code == 0,
                'stdout': stdout,
                'stderr': stderr,
                'execution_time': execution_time,
                'exit_code': exec_result.exit_code,
                'kernel_id': kernel_info.kernel_id,
                'execution_count': kernel_info.execution_count,
                'expires_at': kernel_info.expires_at
            }
            
        except docker.errors.NotFound:
            self._delete_kernel_info(user_id)
            return {
                'success': False,
                'error': 'Kernel container not found. Please create a new kernel.',
                'stdout': '',
                'stderr': '',
                'execution_time': 0
            }
        except Exception as e:
            logger.error(f"Error executing in kernel: {e}")
            return {
                'success': False,
                'error': str(e),
                'stdout': '',
                'stderr': '',
                'execution_time': 0
            }
    
    def _cleanup_kernel(self, user_id: str):
        """Clean up a specific kernel"""
        kernel_info = self._get_kernel_info(user_id)
        if not kernel_info:
            return
        
        try:
            container = self.docker_client.containers.get(kernel_info.container_id)
            container.stop(timeout=5)
            container.remove(force=True)
            logger.info(f"Cleaned up kernel {kernel_info.kernel_id} for user {kernel_info.user_email}")
        except docker.errors.NotFound:
            pass
        except Exception as e:
            logger.error(f"Error cleaning up kernel: {e}")
        
        self._delete_kernel_info(user_id)
    
    def _cleanup_expired_kernels(self):
        """Clean up all expired kernels (called by scheduler)"""
        user_ids = self._get_all_kernel_user_ids()
        cleaned = 0
        
        for user_id in user_ids:
            kernel_info = self._get_kernel_info(user_id)
            if kernel_info and kernel_info.is_expired():
                self._cleanup_kernel(user_id)
                cleaned += 1
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired kernels")
        
        # Also clean up orphaned containers
        self._cleanup_orphaned_containers()
    
    def _cleanup_orphaned_containers(self):
        """Clean up containers that exist but aren't tracked"""
        if not self.docker_client:
            return
        
        try:
            containers = self.docker_client.containers.list(
                all=True,
                filters={'label': 'kernel.type=researcher-kernel'}
            )
            
            tracked_container_ids = set()
            for user_id in self._get_all_kernel_user_ids():
                kernel_info = self._get_kernel_info(user_id)
                if kernel_info:
                    tracked_container_ids.add(kernel_info.container_id)
            
            for container in containers:
                if container.id not in tracked_container_ids:
                    # Check if expired based on label
                    expires_at = container.labels.get('kernel.expires_at')
                    if expires_at:
                        try:
                            if datetime.fromisoformat(expires_at) < datetime.utcnow():
                                container.stop(timeout=5)
                                container.remove(force=True)
                                logger.info(f"Cleaned up orphaned container {container.name}")
                        except:
                            pass
        except Exception as e:
            logger.error(f"Error cleaning orphaned containers: {e}")
    
    def get_kernel_status(self, user_id: str) -> Dict[str, Any]:
        """Get status of a researcher's kernel"""
        kernel_info = self._get_kernel_info(user_id)
        
        if not kernel_info:
            return {
                'has_kernel': False,
                'message': 'No active kernel'
            }
        
        # Check container status
        container_status = 'unknown'
        try:
            container = self.docker_client.containers.get(kernel_info.container_id)
            container_status = container.status
        except:
            container_status = 'not_found'
        
        is_expired = kernel_info.is_expired()
        
        return {
            'has_kernel': True,
            'kernel_id': kernel_info.kernel_id,
            'status': 'expired' if is_expired else kernel_info.status,
            'container_status': container_status,
            'created_at': kernel_info.created_at,
            'expires_at': kernel_info.expires_at,
            'last_activity': kernel_info.last_activity,
            'execution_count': kernel_info.execution_count,
            'is_expired': is_expired,
            'time_remaining': self._get_time_remaining(kernel_info.expires_at) if not is_expired else '0:00:00'
        }
    
    def _get_time_remaining(self, expires_at: str) -> str:
        """Get human-readable time remaining"""
        expires = datetime.fromisoformat(expires_at)
        remaining = expires - datetime.utcnow()
        if remaining.total_seconds() <= 0:
            return '0:00:00'
        hours, remainder = divmod(int(remaining.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    
    def terminate_kernel(self, user_id: str) -> Dict[str, Any]:
        """Manually terminate a researcher's kernel"""
        kernel_info = self._get_kernel_info(user_id)
        
        if not kernel_info:
            return {
                'success': False,
                'message': 'No kernel found'
            }
        
        self._cleanup_kernel(user_id)
        
        return {
            'success': True,
            'message': f'Kernel {kernel_info.kernel_id} terminated'
        }
    
    def list_all_kernels(self) -> List[Dict[str, Any]]:
        """List all active kernels (admin function)"""
        kernels = []
        for user_id in self._get_all_kernel_user_ids():
            status = self.get_kernel_status(user_id)
            if status.get('has_kernel'):
                kernels.append(status)
        return kernels
    
    def shutdown(self):
        """Shutdown the kernel manager"""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        logger.info("Kernel manager shutdown complete")


# Global kernel manager instance
_kernel_manager: Optional[KernelManager] = None


def get_kernel_manager() -> KernelManager:
    """Get or create the global kernel manager instance"""
    global _kernel_manager
    if _kernel_manager is None:
        _kernel_manager = KernelManager()
    return _kernel_manager
