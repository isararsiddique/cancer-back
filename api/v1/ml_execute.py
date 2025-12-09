"""
ML Sandbox Code Execution API

This endpoint allows researchers to execute Python code in an isolated sandbox environment.
The code runs in a Docker container with resource limits and network isolation.

Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import logging

from core.deps import get_db, get_current_user, permission_required
from db.models.users import User
from services.ml_sandbox import MLSandboxService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml-execution"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class CodeExecutionRequest(BaseModel):
    """Request model for code execution"""
    code: str = Field(..., description="Python code to execute", min_length=1)
    dataset_token: Optional[str] = Field(None, description="Dataset access token (if accessing data)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": "import numpy as np\nprint('Hello from sandbox!')\nprint(np.array([1,2,3]))",
                "dataset_token": "optional-dataset-token"
            }
        }


class CodeExecutionResponse(BaseModel):
    """Response model for code execution"""
    success: bool = Field(..., description="Whether execution succeeded")
    stdout: str = Field(..., description="Standard output from code execution")
    stderr: str = Field(..., description="Standard error output")
    execution_time: float = Field(..., description="Execution time in seconds")
    timeout: bool = Field(..., description="Whether execution timed out")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    visualizations: Optional[list] = Field(None, description="Generated visualizations (base64 encoded)")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Structured metrics from execution")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "stdout": "Hello from sandbox!\n[1 2 3]",
                "stderr": "",
                "execution_time": 0.523,
                "timeout": False,
                "error": None,
                "visualizations": None,
                "metrics": None
            }
        }


class SandboxStatusResponse(BaseModel):
    """Response model for sandbox status"""
    available: bool = Field(..., description="Whether sandbox is available")
    image_name: str = Field(..., description="Docker image name")
    timeout_seconds: int = Field(..., description="Execution timeout in seconds")
    memory_limit: str = Field(..., description="Memory limit")
    cpu_limit: float = Field(..., description="CPU limit in cores")
    message: Optional[str] = Field(None, description="Status message")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/execute", response_model=CodeExecutionResponse)
def execute_code(
    request: CodeExecutionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Execute Python code in an isolated sandbox environment.
    
    The code runs in a Docker container with:
    - Resource limits (CPU, memory, timeout)
    - Network isolation
    - Pre-installed ML libraries (numpy, pandas, scikit-learn, matplotlib, plotly, seaborn)
    - Non-root user execution
    
    Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
    
    Authorization: Only users with researcher role can execute code (Requirement 3.2)
    """
    # Check if user has researcher role (Requirement 3.2)
    user_roles = [r.slug for r in current_user.roles]
    is_researcher = any(role in ['researcher', 'super_admin'] for role in user_roles)
    
    if not is_researcher:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="ML code execution is only available to researchers"
        )
    
    # Validate code input (Requirement 3.2)
    if not request.code or not request.code.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Code cannot be empty"
        )
    
    # Validate dataset token if provided (Requirement 3.2)
    if request.dataset_token:
        # TODO: Implement dataset token validation
        # For now, we'll accept any token
        logger.info(f"Dataset token provided: {request.dataset_token[:10]}...")
    
    try:
        # Initialize sandbox service
        sandbox = MLSandboxService(
            timeout_seconds=30,  # 30 second timeout (Requirement 3.2)
            memory_limit="2g",   # 2GB memory limit (Requirement 3.2)
            cpu_limit=2.0        # 2 CPU cores (Requirement 3.2)
        )
        
        # Check if Docker is available
        if not sandbox.check_docker_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML sandbox is not available. Docker service may be down or image not built."
            )
        
        # Execute code (Requirement 3.2)
        result = sandbox.execute_code(
            code=request.code,
            dataset_token=request.dataset_token
        )
        
        # Return results with visualizations and metrics (Requirements 3.3, 3.4, 3.5, 3.6)
        return CodeExecutionResponse(
            success=result['success'],
            stdout=result['stdout'],
            stderr=result['stderr'],
            execution_time=result['execution_time'],
            timeout=result['timeout'],
            error=result.get('error'),  # Clear error messages (Requirement 3.7)
            visualizations=None,  # TODO: Extract visualizations from output
            metrics=None  # TODO: Extract structured metrics from output
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute code: {str(e)}"
        )


@router.get("/status", response_model=SandboxStatusResponse)
def get_sandbox_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get the status of the ML sandbox environment.
    
    Returns information about Docker availability, resource limits, and configuration.
    """
    # Check if user has researcher role
    user_roles = [r.slug for r in current_user.roles]
    is_researcher = any(role in ['researcher', 'super_admin'] for role in user_roles)
    
    if not is_researcher:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="ML sandbox status is only available to researchers"
        )
    
    try:
        sandbox = MLSandboxService()
        available = sandbox.check_docker_available()
        
        message = "Sandbox is ready" if available else "Sandbox is not available. Docker may be down or image not built."
        
        return SandboxStatusResponse(
            available=available,
            image_name=sandbox.image_name,
            timeout_seconds=sandbox.timeout_seconds,
            memory_limit=sandbox.memory_limit,
            cpu_limit=sandbox.cpu_limit,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Error checking sandbox status: {str(e)}", exc_info=True)
        return SandboxStatusResponse(
            available=False,
            image_name="ml-sandbox:latest",
            timeout_seconds=30,
            memory_limit="2g",
            cpu_limit=2.0,
            message=f"Error: {str(e)}"
        )


@router.get("/libraries")
def get_available_libraries(
    current_user: User = Depends(get_current_user)
):
    """
    Get list of available libraries in the sandbox environment.
    
    Returns information about pre-installed ML libraries and their versions.
    
    Requirement: 3.8
    """
    # Check if user has researcher role
    user_roles = [r.slug for r in current_user.roles]
    is_researcher = any(role in ['researcher', 'super_admin'] for role in user_roles)
    
    if not is_researcher:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Library information is only available to researchers"
        )
    
    return {
        "python_version": "3.11",
        "libraries": [
            {"name": "numpy", "version": "1.26.4", "description": "Numerical computing"},
            {"name": "pandas", "version": "2.2.0", "description": "Data manipulation and analysis"},
            {"name": "scikit-learn", "version": "1.4.0", "description": "Machine learning"},
            {"name": "matplotlib", "version": "3.8.2", "description": "Data visualization"},
            {"name": "plotly", "version": "5.18.0", "description": "Interactive visualization"},
            {"name": "seaborn", "version": "0.13.2", "description": "Statistical visualization"},
            {"name": "scipy", "version": "1.12.0", "description": "Scientific computing"},
            {"name": "joblib", "version": "1.3.2", "description": "Model persistence"}
        ],
        "resource_limits": {
            "timeout_seconds": 30,
            "memory_limit": "2GB",
            "cpu_cores": 2.0,
            "network_access": False
        },
        "security": {
            "network_isolation": True,
            "non_root_user": True,
            "read_only_filesystem": True
        }
    }
