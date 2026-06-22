"""
ML Model Prediction API

Allows users to upload trained models (.pkl or .h5) and make predictions.
Supports both scikit-learn pipelines and Keras/TensorFlow models.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import uuid
import json
import logging
import tempfile
import os
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

from core.deps import get_db, get_current_user
from db.models.users import User

router = APIRouter(prefix="/ml-predict", tags=["ml-prediction"])

# Store uploaded models in memory (for session) or disk
UPLOADED_MODELS: Dict[str, Dict[str, Any]] = {}


class PredictionInput(BaseModel):
    """Input data for prediction"""
    model_id: str = Field(..., description="ID of the uploaded model")
    data: List[Dict[str, Any]] = Field(..., description="Data records to predict")


class PredictionResponse(BaseModel):
    """Response from prediction"""
    success: bool
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    model_info: Dict[str, Any]
    prediction_count: int


class ModelInfo(BaseModel):
    """Information about an uploaded model"""
    model_id: str
    filename: str
    model_type: str  # sklearn, keras, xgboost
    algorithm: Optional[str] = None
    features: Optional[List[str]] = None
    target: Optional[str] = None
    uploaded_at: str
    file_size: int


@router.post("/upload")
async def upload_model(
    file: UploadFile = File(..., description="Model file (.pkl for sklearn, .h5 for Keras)"),
    model_name: str = Form(None, description="Optional model name"),
    current_user: User = Depends(get_current_user)
):
    """
    Upload a trained model for predictions.
    
    Supports:
    - .pkl files (scikit-learn pipelines, XGBoost, etc.)
    - .h5 files (Keras/TensorFlow models)
    - .joblib files (scikit-learn models)
    """
    # Check user role
    user_roles = [r.slug for r in current_user.roles]
    if not any(role in ['researcher', 'super_admin', 'ummc_admin', 'doctor'] for role in user_roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only researchers and admins can upload models"
        )
    
    # Validate file extension
    filename = file.filename or "model"
    ext = Path(filename).suffix.lower()
    
    if ext not in ['.pkl', '.h5', '.joblib', '.keras']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ext}. Supported: .pkl, .h5, .joblib, .keras"
        )
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    # Limit file size (100MB)
    if file_size > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model file too large. Maximum size is 100MB."
        )
    
    # Generate model ID
    model_id = f"model_{uuid.uuid4().hex[:12]}"
    
    # Save to temp file and load model
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        model_info = {
            "model_id": model_id,
            "filename": filename,
            "model_name": model_name or filename,
            "file_path": tmp_path,
            "file_size": file_size,
            "uploaded_at": datetime.utcnow().isoformat(),
            "user_id": str(current_user.id),
            "user_email": current_user.email,
        }
        
        # Try to load and inspect the model
        if ext in ['.pkl', '.joblib']:
            import pickle
            import joblib
            
            try:
                if ext == '.joblib':
                    model = joblib.load(tmp_path)
                else:
                    with open(tmp_path, 'rb') as f:
                        model = pickle.load(f)
                
                model_info["model_type"] = "sklearn"
                
                # Extract model info from pipeline
                if hasattr(model, 'named_steps'):
                    # It's a Pipeline
                    clf_name = list(model.named_steps.keys())[-1]
                    clf = model.named_steps[clf_name]
                    model_info["algorithm"] = type(clf).__name__
                    
                    # Try to get feature names
                    if hasattr(model, 'feature_names_in_'):
                        model_info["features"] = list(model.feature_names_in_)
                    elif 'preprocess' in model.named_steps:
                        preprocessor = model.named_steps['preprocess']
                        if hasattr(preprocessor, 'feature_names_in_'):
                            model_info["features"] = list(preprocessor.feature_names_in_)
                else:
                    model_info["algorithm"] = type(model).__name__
                    if hasattr(model, 'feature_names_in_'):
                        model_info["features"] = list(model.feature_names_in_)
                
                model_info["model_object"] = model
                
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to load sklearn model: {str(e)}"
                )
        
        elif ext in ['.h5', '.keras']:
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(tmp_path)
                
                model_info["model_type"] = "keras"
                model_info["algorithm"] = "Neural Network (Keras)"
                
                # Get input shape
                if model.input_shape:
                    input_shape = model.input_shape
                    if isinstance(input_shape, tuple):
                        model_info["input_shape"] = input_shape
                        if len(input_shape) > 1 and input_shape[1]:
                            model_info["n_features"] = input_shape[1]
                
                model_info["model_object"] = model
                
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to load Keras model: {str(e)}"
                )
        
        # Store model in memory
        UPLOADED_MODELS[model_id] = model_info
        
        # Return info (without model object)
        response_info = {k: v for k, v in model_info.items() if k != "model_object"}
        
        return {
            "success": True,
            "model_id": model_id,
            "message": "Model uploaded successfully",
            "model_info": response_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process model: {str(e)}"
        )


@router.post("/predict", response_model=PredictionResponse)
async def make_prediction(
    request: PredictionInput,
    current_user: User = Depends(get_current_user)
):
    """
    Make predictions using an uploaded model.
    
    Send data as a list of records (dictionaries) with feature values.
    """
    model_id = request.model_id
    data = request.data
    
    if model_id not in UPLOADED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}. Please upload a model first."
        )
    
    model_info = UPLOADED_MODELS[model_id]
    model = model_info.get("model_object")
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model object not available"
        )
    
    if not data or len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No data provided for prediction"
        )
    
    try:
        import pandas as pd
        import numpy as np
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Handle missing values
        df = df.replace(['', None, 'None', 'null', 'NaN'], np.nan)
        
        predictions = []
        probabilities = None
        
        if model_info["model_type"] == "sklearn":
            # Get expected features if available
            expected_features = model_info.get("features")
            
            if expected_features:
                # Ensure all expected features are present
                for feat in expected_features:
                    if feat not in df.columns:
                        df[feat] = np.nan
                # Reorder columns to match training
                df = df[expected_features]
            
            # Fill missing values
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
                else:
                    df[col] = df[col].fillna('__MISSING__').astype(str)
            
            # Make predictions
            predictions = model.predict(df).tolist()
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(df)
                    probabilities = proba.tolist()
                except:
                    pass
        
        elif model_info["model_type"] == "keras":
            from sklearn.preprocessing import StandardScaler
            
            # For Keras, we need numeric data
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Keras model requires numeric features"
                )
            
            # Fill NaN and scale
            numeric_df = numeric_df.fillna(0)
            
            # Check if we need to reshape for CNN
            n_features = model_info.get("n_features")
            if n_features and numeric_df.shape[1] != n_features:
                # Try to match feature count
                if numeric_df.shape[1] < n_features:
                    # Pad with zeros
                    for i in range(n_features - numeric_df.shape[1]):
                        numeric_df[f'_pad_{i}'] = 0
                else:
                    # Take first n_features
                    numeric_df = numeric_df.iloc[:, :n_features]
            
            X = numeric_df.values
            
            # Reshape for CNN if needed (add channel dimension)
            input_shape = model_info.get("input_shape")
            if input_shape and len(input_shape) == 3:
                X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Predict
            raw_predictions = model.predict(X, verbose=0)
            
            # Convert predictions
            if raw_predictions.shape[1] == 1:
                # Binary classification
                probabilities = [[1 - p[0], p[0]] for p in raw_predictions]
                predictions = [(1 if p[0] > 0.5 else 0) for p in raw_predictions]
            else:
                # Multi-class
                probabilities = raw_predictions.tolist()
                predictions = np.argmax(raw_predictions, axis=1).tolist()
        
        # Prepare response info
        response_info = {
            "model_id": model_id,
            "model_type": model_info["model_type"],
            "algorithm": model_info.get("algorithm"),
            "filename": model_info["filename"]
        }
        
        return PredictionResponse(
            success=True,
            predictions=predictions,
            probabilities=probabilities,
            model_info=response_info,
            prediction_count=len(predictions)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/models")
async def list_uploaded_models(
    current_user: User = Depends(get_current_user)
):
    """List all models uploaded by the current user"""
    user_id = str(current_user.id)
    
    user_models = []
    for model_id, info in UPLOADED_MODELS.items():
        if info.get("user_id") == user_id:
            # Return info without model object
            model_data = {k: v for k, v in info.items() if k != "model_object"}
            user_models.append(model_data)
    
    return {
        "models": user_models,
        "total": len(user_models)
    }


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete an uploaded model"""
    if model_id not in UPLOADED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    model_info = UPLOADED_MODELS[model_id]
    
    # Check ownership
    if model_info.get("user_id") != str(current_user.id):
        user_roles = [r.slug for r in current_user.roles]
        if 'super_admin' not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only delete your own models"
            )
    
    # Clean up temp file
    try:
        file_path = model_info.get("file_path")
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except:
        pass
    
    del UPLOADED_MODELS[model_id]
    
    return {"success": True, "message": "Model deleted"}


@router.get("/models/{model_id}")
async def get_model_info(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get information about an uploaded model"""
    if model_id not in UPLOADED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    model_info = UPLOADED_MODELS[model_id]
    
    # Return info without model object
    return {k: v for k, v in model_info.items() if k != "model_object"}
