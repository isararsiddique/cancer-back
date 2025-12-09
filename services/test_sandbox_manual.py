#!/usr/bin/env python3
"""
Manual test script for ML Sandbox Service

This script can be run manually to test the ML Sandbox Service when Docker is available.
It's not part of the automated test suite.

Usage:
    python3 app/services/test_sandbox_manual.py
"""

from ml_sandbox import MLSandboxService


def test_simple_execution():
    """Test simple code execution"""
    print("=" * 60)
    print("Test 1: Simple Code Execution")
    print("=" * 60)
    
    service = MLSandboxService()
    code = "print('Hello from ML Sandbox!')"
    
    result = service.execute_code(code)
    
    print(f"Success: {result['success']}")
    print(f"Stdout: {result['stdout']}")
    print(f"Stderr: {result['stderr']}")
    print(f"Execution time: {result['execution_time']:.3f}s")
    print()


def test_numpy_execution():
    """Test numpy code execution"""
    print("=" * 60)
    print("Test 2: NumPy Code Execution")
    print("=" * 60)
    
    service = MLSandboxService()
    code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f'Array: {arr}')
print(f'Sum: {arr.sum()}')
print(f'Mean: {arr.mean()}')
"""
    
    result = service.execute_code(code)
    
    print(f"Success: {result['success']}")
    print(f"Stdout:\n{result['stdout']}")
    print(f"Execution time: {result['execution_time']:.3f}s")
    print()


def test_ml_model():
    """Test ML model training"""
    print("=" * 60)
    print("Test 3: ML Model Training")
    print("=" * 60)
    
    service = MLSandboxService()
    code = """
from sklearn.linear_model import LinearRegression
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearRegression()
model.fit(X, y)

# Make prediction
prediction = model.predict([[6]])
print(f'Prediction for x=6: {prediction[0]:.1f}')
print(f'Coefficient: {model.coef_[0]:.1f}')
print(f'Intercept: {model.intercept_:.1f}')
"""
    
    result = service.execute_code(code)
    
    print(f"Success: {result['success']}")
    print(f"Stdout:\n{result['stdout']}")
    print(f"Execution time: {result['execution_time']:.3f}s")
    print()


def test_error_handling():
    """Test error handling"""
    print("=" * 60)
    print("Test 4: Error Handling")
    print("=" * 60)
    
    service = MLSandboxService()
    code = """
# This will cause an error
print(undefined_variable)
"""
    
    result = service.execute_code(code)
    
    print(f"Success: {result['success']}")
    print(f"Error: {result['error']}")
    print(f"Stderr:\n{result['stderr']}")
    print()


def test_timeout():
    """Test timeout enforcement"""
    print("=" * 60)
    print("Test 5: Timeout Enforcement")
    print("=" * 60)
    
    service = MLSandboxService(timeout_seconds=2)
    code = """
import time
print('Starting long operation...')
time.sleep(5)
print('This should not print')
"""
    
    result = service.execute_code(code)
    
    print(f"Success: {result['success']}")
    print(f"Timeout: {result['timeout']}")
    print(f"Error: {result['error']}")
    print(f"Execution time: {result['execution_time']:.3f}s")
    print()


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ML Sandbox Service Manual Tests")
    print("=" * 60)
    print()
    
    # Check if Docker is available
    try:
        service = MLSandboxService()
        if not service.check_docker_available():
            print("❌ Docker is not available or ml-sandbox image not found")
            print("Please build the Docker image first:")
            print("  cd docker/ml-sandbox")
            print("  ./build.sh")
            return
        
        print("✓ Docker is available")
        print("✓ ml-sandbox image found")
        print()
        
        # Run tests
        test_simple_execution()
        test_numpy_execution()
        test_ml_model()
        test_error_handling()
        test_timeout()
        
        print("=" * 60)
        print("✓ All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nMake sure Docker is running and the ml-sandbox image is built.")


if __name__ == "__main__":
    main()
