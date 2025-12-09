"""
ML Sandbox Execution Service

This service executes Python code in an isolated Docker container with resource limits.
It captures stdout, stderr, execution time, and handles errors.

Requirements: 3.2, 3.7
"""

import docker
import tempfile
import os
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MLSandboxService:
    """
    Service for executing Python code in an isolated Docker container.
    
    Features:
    - Executes code in isolated Docker container
    - Captures stdout and stderr
    - Enforces timeout limits
    - Enforces memory limits
    - Handles execution errors
    - Network isolation
    - Non-root user execution
    
    Requirements: 3.2, 3.7
    """
    
    def __init__(
        self,
        image_name: str = "ml-sandbox:latest",
        timeout_seconds: int = 30,
        memory_limit: str = "2g",
        cpu_limit: float = 2.0
    ):
        """
        Initialize the ML Sandbox Service.
        
        Args:
            image_name: Docker image name for the sandbox
            timeout_seconds: Maximum execution time in seconds (Requirement 3.2)
            memory_limit: Memory limit (e.g., "2g" for 2GB) (Requirement 3.2)
            cpu_limit: CPU limit in cores (e.g., 2.0 for 2 cores) (Requirement 3.2)
        """
        self.image_name = image_name
        self.timeout_seconds = timeout_seconds
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        
        try:
            self.docker_client = docker.from_env()
            logger.info(f"Docker client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {str(e)}")
            raise RuntimeError(f"Docker is not available: {str(e)}")
    
    def execute_code(
        self,
        code: str,
        dataset_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code in the isolated sandbox container.
        
        Args:
            code: Python code to execute
            dataset_token: Optional dataset access token
        
        Returns:
            Dictionary containing:
            - success: bool - Whether execution succeeded
            - stdout: str - Standard output
            - stderr: str - Standard error output
            - execution_time: float - Execution time in seconds
            - error: str - Error message if execution failed
            - timeout: bool - Whether execution timed out
        
        Requirements: 3.2, 3.7
        """
        start_time = time.time()
        
        # Create temporary file for the code
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        try:
            # Prepare volume mount
            volumes = {
                temp_file_path: {
                    'bind': '/sandbox/code.py',
                    'mode': 'ro'  # Read-only
                }
            }
            
            # Execute code in container with resource limits (Requirement 3.2)
            container = self.docker_client.containers.run(
                image=self.image_name,
                command=['python3', '/sandbox/code.py'],
                volumes=volumes,
                network_mode='none',  # Network isolation (Requirement 3.2)
                mem_limit=self.memory_limit,  # Memory limit (Requirement 3.2)
                nano_cpus=int(self.cpu_limit * 1e9),  # CPU limit (Requirement 3.2)
                user='1000:1000',  # Non-root user (Requirement 3.2)
                detach=True,
                remove=False,  # Don't auto-remove so we can get logs
                security_opt=['no-new-privileges'],
                read_only=True,
                tmpfs={
                    '/tmp': 'size=512M,mode=1777',
                    '/sandbox': 'size=1G,mode=1777'
                }
            )
            
            try:
                # Wait for container with timeout (Requirement 3.2)
                result = container.wait(timeout=self.timeout_seconds)
                execution_time = time.time() - start_time
                
                # Get stdout and stderr (Requirement 3.2, 3.7)
                stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
                stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
                
                # Check exit code
                exit_code = result.get('StatusCode', -1)
                success = exit_code == 0
                
                return {
                    'success': success,
                    'stdout': stdout,
                    'stderr': stderr,
                    'execution_time': execution_time,
                    'exit_code': exit_code,
                    'timeout': False,
                    'error': stderr if not success else None
                }
                
            except docker.errors.ContainerError as e:
                # Container exited with non-zero code
                execution_time = time.time() - start_time
                return {
                    'success': False,
                    'stdout': e.stdout.decode('utf-8') if e.stdout else '',
                    'stderr': e.stderr.decode('utf-8') if e.stderr else str(e),
                    'execution_time': execution_time,
                    'exit_code': e.exit_status,
                    'timeout': False,
                    'error': f"Container error: {str(e)}"
                }
                
            except Exception as e:
                # Timeout or other error
                execution_time = time.time() - start_time
                is_timeout = execution_time >= self.timeout_seconds
                
                # Try to stop the container
                try:
                    container.stop(timeout=1)
                except:
                    pass
                
                # Get logs if available
                try:
                    stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
                    stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
                except:
                    stdout = ''
                    stderr = ''
                
                error_msg = f"Execution timeout ({self.timeout_seconds}s)" if is_timeout else str(e)
                
                return {
                    'success': False,
                    'stdout': stdout,
                    'stderr': stderr,
                    'execution_time': execution_time,
                    'exit_code': -1,
                    'timeout': is_timeout,
                    'error': error_msg  # Clear error message (Requirement 3.7)
                }
                
            finally:
                # Clean up container
                try:
                    container.remove(force=True)
                except:
                    pass
                    
        except docker.errors.ImageNotFound:
            return {
                'success': False,
                'stdout': '',
                'stderr': '',
                'execution_time': 0,
                'exit_code': -1,
                'timeout': False,
                'error': f"Docker image '{self.image_name}' not found. Please build the image first."
            }
            
        except docker.errors.APIError as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': '',
                'execution_time': 0,
                'exit_code': -1,
                'timeout': False,
                'error': f"Docker API error: {str(e)}"
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in execute_code: {str(e)}", exc_info=True)
            return {
                'success': False,
                'stdout': '',
                'stderr': '',
                'execution_time': 0,
                'exit_code': -1,
                'timeout': False,
                'error': f"Unexpected error: {str(e)}"
            }
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    def check_docker_available(self) -> bool:
        """
        Check if Docker is available and the sandbox image exists.
        
        Returns:
            bool: True if Docker and image are available
        """
        try:
            self.docker_client.ping()
            self.docker_client.images.get(self.image_name)
            return True
        except:
            return False
    
    def get_image_info(self) -> Dict[str, Any]:
        """
        Get information about the sandbox Docker image.
        
        Returns:
            Dictionary with image information
        """
        try:
            image = self.docker_client.images.get(self.image_name)
            return {
                'id': image.id,
                'tags': image.tags,
                'created': image.attrs.get('Created'),
                'size': image.attrs.get('Size'),
            }
        except docker.errors.ImageNotFound:
            return {
                'error': f"Image '{self.image_name}' not found"
            }
        except Exception as e:
            return {
                'error': str(e)
            }
