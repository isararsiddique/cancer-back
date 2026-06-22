"""
Researcher Kernel API

API endpoints for managing dedicated Jupyter kernels for researchers.
Each researcher gets their own isolated kernel that persists for 24 hours.

Features:
- Create/get dedicated kernel per researcher
- Execute code in persistent kernel (maintains state)
- Check kernel status and time remaining
- Manual kernel termination
- Admin: list all active kernels
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import logging

from core.deps import get_db, get_current_user
from db.models.users import User
from services.kernel_manager import get_kernel_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kernel", tags=["kernel"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class KernelExecuteRequest(BaseModel):
    """Request model for code execution in kernel"""
    code: str = Field(..., description="Python code to execute", min_length=1)
    timeout: int = Field(60, description="Execution timeout in seconds", ge=1, le=300)
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": "import pandas as pd\ndf = pd.DataFrame({'a': [1,2,3]})\nprint(df)",
                "timeout": 60
            }
        }


class KernelExecuteResponse(BaseModel):
    """Response model for kernel execution"""
    success: bool
    stdout: str
    stderr: str
    execution_time: float
    exit_code: Optional[int] = None
    kernel_id: Optional[str] = None
    execution_count: Optional[int] = None
    expires_at: Optional[str] = None
    error: Optional[str] = None


class KernelStatusResponse(BaseModel):
    """Response model for kernel status"""
    has_kernel: bool
    kernel_id: Optional[str] = None
    status: Optional[str] = None
    container_status: Optional[str] = None
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    last_activity: Optional[str] = None
    execution_count: Optional[int] = None
    is_expired: Optional[bool] = None
    time_remaining: Optional[str] = None
    message: Optional[str] = None


class KernelCreateResponse(BaseModel):
    """Response model for kernel creation"""
    success: bool
    kernel_id: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    lifetime_hours: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_researcher_role(current_user: User) -> bool:
    """Check if user has researcher role"""
    user_roles = [r.slug for r in current_user.roles]
    return any(role in ['researcher', 'super_admin', 'ummc_admin'] for role in user_roles)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/create", response_model=KernelCreateResponse)
def create_or_get_kernel(
    current_user: User = Depends(get_current_user)
):
    """
    Create or get existing dedicated kernel for the researcher.
    
    Each researcher gets their own isolated kernel environment that:
    - Persists for 24 hours
    - Maintains session state (variables, imports)
    - Auto-cleans after expiration
    
    If a kernel already exists and is not expired, it will be reused.
    """
    if not check_researcher_role(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only researchers can create kernels"
        )
    
    try:
        manager = get_kernel_manager()
        result = manager.get_or_create_kernel(
            user_id=str(current_user.id),
            user_email=current_user.email
        )
        
        if not result.get('success'):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.get('error', 'Failed to create kernel')
            )
        
        return KernelCreateResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating kernel: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/execute", response_model=KernelExecuteResponse)
def execute_in_kernel(
    request: KernelExecuteRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Execute Python code in the researcher's dedicated kernel.
    
    The kernel maintains state between executions:
    - Variables persist across calls
    - Imported libraries remain available
    - Data loaded in previous executions is accessible
    
    This is different from the sandbox /ml/execute endpoint which
    creates a fresh container for each execution.
    """
    if not check_researcher_role(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only researchers can execute code in kernels"
        )
    
    if not request.code or not request.code.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Code cannot be empty"
        )
    
    try:
        manager = get_kernel_manager()
        
        # Auto-create kernel if not exists
        kernel_status = manager.get_kernel_status(str(current_user.id))
        if not kernel_status.get('has_kernel') or kernel_status.get('is_expired'):
            create_result = manager.get_or_create_kernel(
                user_id=str(current_user.id),
                user_email=current_user.email
            )
            if not create_result.get('success'):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=create_result.get('error', 'Failed to create kernel')
                )
        
        # Execute code
        result = manager.execute_in_kernel(
            user_id=str(current_user.id),
            code=request.code,
            timeout=request.timeout
        )
        
        return KernelExecuteResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing in kernel: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/status", response_model=KernelStatusResponse)
def get_kernel_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get status of the researcher's dedicated kernel.
    
    Returns:
    - Whether a kernel exists
    - Kernel ID and status
    - Creation and expiration times
    - Time remaining before auto-cleanup
    - Execution count
    """
    if not check_researcher_role(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only researchers can check kernel status"
        )
    
    try:
        manager = get_kernel_manager()
        status = manager.get_kernel_status(str(current_user.id))
        return KernelStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Error getting kernel status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/terminate")
def terminate_kernel(
    current_user: User = Depends(get_current_user)
):
    """
    Manually terminate the researcher's kernel.
    
    This immediately stops and removes the kernel container.
    A new kernel can be created afterwards.
    """
    if not check_researcher_role(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only researchers can terminate kernels"
        )
    
    try:
        manager = get_kernel_manager()
        result = manager.terminate_kernel(str(current_user.id))
        return result
        
    except Exception as e:
        logger.error(f"Error terminating kernel: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/admin/list")
def list_all_kernels(
    current_user: User = Depends(get_current_user)
):
    """
    List all active kernels (admin only).
    
    Returns information about all researcher kernels including:
    - Kernel IDs and user emails
    - Status and time remaining
    - Execution counts
    """
    user_roles = [r.slug for r in current_user.roles]
    if 'super_admin' not in user_roles and 'ummc_admin' not in user_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        manager = get_kernel_manager()
        kernels = manager.list_all_kernels()
        
        return {
            'total_kernels': len(kernels),
            'kernels': kernels
        }
        
    except Exception as e:
        logger.error(f"Error listing kernels: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/admin/cleanup")
def force_cleanup_expired(
    current_user: User = Depends(get_current_user)
):
    """
    Force cleanup of all expired kernels (admin only).
    
    This manually triggers the cleanup process that normally
    runs automatically every 15 minutes.
    """
    user_roles = [r.slug for r in current_user.roles]
    if 'super_admin' not in user_roles and 'ummc_admin' not in user_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        manager = get_kernel_manager()
        before_count = len(manager.list_all_kernels())
        manager._cleanup_expired_kernels()
        after_count = len(manager.list_all_kernels())
        
        return {
            'success': True,
            'kernels_before': before_count,
            'kernels_after': after_count,
            'cleaned': before_count - after_count
        }
        
    except Exception as e:
        logger.error(f"Error in cleanup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
