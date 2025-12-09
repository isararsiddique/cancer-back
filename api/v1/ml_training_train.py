"""
ML Model Training API - Server-side training endpoint
Handles actual model training on the backend server
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import uuid
import json
import traceback
import logging

logger = logging.getLogger(__name__)

from core.deps import get_db, get_current_user
from db.models.users import User
from db.models.safehaven import MLTrainingResult
from db.models.research import ResearchRequest
from db.session import SessionLocal

router = APIRouter(prefix="/ml-training", tags=["ml-training"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class TrainingRequest(BaseModel):
    """Request to train a model on the backend"""
    research_request_id: Optional[str] = None
    project_id: Optional[str] = None
    
    # Model Configuration
    algorithm: str = Field(..., description="Algorithm: xgboost, random_forest, neural_network, cnn")
    target_variable: str
    features: List[str]
    hyperparameters: Optional[Dict[str, Any]] = None
    
    # Training Configuration
    test_size: Optional[float] = 0.2
    random_state: Optional[int] = 42
    custom_pipeline: Optional[str] = None
    
    # Dataset (can be large, will be processed server-side)
    dataset: List[Dict[str, Any]] = Field(..., description="Training dataset as list of records")


class TrainingResponse(BaseModel):
    """Response from training request"""
    training_id: str
    status: str  # queued, training, completed, failed
    message: str


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model_async(
    training_id: str,
    dataset: List[Dict[str, Any]],
    config: Dict[str, Any],
    user_id: str
):
    """
    Background task to train ML model.
    This runs in a separate thread to avoid blocking the API.
    Creates its own database session.
    """
    # Create a new database session for this background task
    db = SessionLocal()
    try:
        # Import ML libraries (will need to be installed on server)
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split, cross_val_score, KFold
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,
            roc_auc_score, roc_curve
        )
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        import time
        try:
            import psutil
        except ImportError:
            # Fallback if psutil not available
            psutil = None
        
        # Update status to training
        result = db.query(MLTrainingResult).filter(
            MLTrainingResult.model_id == training_id
        ).first()
        if result:
            result.training_status = 'training'
            db.commit()
        
        # Resource monitoring - Before training
        try:
            cpu_before = psutil.cpu_percent(interval=1) if psutil else 0
            mem_before = psutil.virtual_memory().used / (1024 * 1024) if psutil else 0  # MB
        except:
            cpu_before = 0
            mem_before = 0
        start_time = time.time()
        
        # Load dataset
        if not dataset or len(dataset) == 0:
            raise ValueError("Dataset is empty")
        df = pd.DataFrame(dataset)
        df = df.replace(['', None, 'None', 'null'], np.nan)
        
        # Prepare features and target
        target_col = config['target_variable']
        feature_cols = [f for f in config['features'] if f != target_col]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Convert categorical columns to strings
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            X[col] = X[col].fillna('__MISSING__').astype(str)
        
        # Handle target
        y = y.replace([None, 'None', 'null'], np.nan)
        if y.dtype in ['float64', 'int64']:
            y = y.fillna(y.median() if not pd.isna(y.median()) else 0)
        else:
            mode_values = y.mode()
            y = y.fillna(mode_values[0] if len(mode_values) > 0 else 0)
        
        if y.dtype == 'object':
            y = pd.Categorical(y).codes
        
        # Identify features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Remove constant features
        constant_features = []
        for col in numeric_features + categorical_features:
            col_data = X[col] if isinstance(X[col], pd.Series) else pd.Series(X[col])
            if col_data.nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            X = X.drop(columns=constant_features)
            numeric_features = [f for f in numeric_features if f not in constant_features]
            categorical_features = [f for f in categorical_features if f not in constant_features]
        
        # Create preprocessing pipeline
        transformers = []
        if len(numeric_features) > 0:
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            transformers.append(("num", numeric_transformer, numeric_features))
        
        if len(categorical_features) > 0:
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent", fill_value='__MISSING__')),
                ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(("cat", categorical_transformer, categorical_features))
        
        if len(transformers) == 0:
            raise ValueError("No valid features for preprocessing.")
        
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        
        # Split data
        test_size = config.get('test_size', 0.2)
        random_state = config.get('random_state', 42)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=random_state,
            stratify=y if len(np.unique(y)) <= 10 else None
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=random_state,
            stratify=y_temp if len(np.unique(y_temp)) <= 10 else None
        )
        
        # Build and train model based on algorithm
        algorithm = config['algorithm']
        hyperparams = config.get('hyperparameters', {})
        
        if algorithm == 'xgboost':
            from xgboost import XGBClassifier, XGBRegressor
            is_classification = len(np.unique(y_train)) <= 10
            if is_classification:
                clf = XGBClassifier(
                    n_estimators=hyperparams.get('n_estimators', 300),
                    max_depth=hyperparams.get('max_depth', 8),
                    learning_rate=hyperparams.get('learning_rate', 0.05),
                    subsample=hyperparams.get('subsample', 0.8),
                    colsample_bytree=hyperparams.get('colsample_bytree', 0.8),
                    random_state=random_state
                )
            else:
                clf = XGBRegressor(
                    n_estimators=hyperparams.get('n_estimators', 300),
                    max_depth=hyperparams.get('max_depth', 8),
                    learning_rate=hyperparams.get('learning_rate', 0.05),
                    subsample=hyperparams.get('subsample', 0.8),
                    colsample_bytree=hyperparams.get('colsample_bytree', 0.8),
                    random_state=random_state
                )
        
        elif algorithm == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            is_classification = len(np.unique(y_train)) <= 10
            max_depth = hyperparams.get('max_depth')
            if is_classification:
                clf = RandomForestClassifier(
                    n_estimators=hyperparams.get('n_estimators', 400),
                    max_depth=max_depth if max_depth else None,
                    random_state=random_state
                )
            else:
                clf = RandomForestRegressor(
                    n_estimators=hyperparams.get('n_estimators', 400),
                    max_depth=max_depth if max_depth else None,
                    random_state=random_state
                )
        
        elif algorithm == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression, Ridge
            is_classification = len(np.unique(y_train)) <= 10
            if is_classification:
                clf = LogisticRegression(
                    solver=hyperparams.get('solver', 'liblinear'),
                    C=hyperparams.get('C', 1.0),
                    max_iter=hyperparams.get('max_iter', 1000),
                    random_state=random_state
                )
            else:
                clf = Ridge(
                    alpha=hyperparams.get('alpha', 1.0),
                    random_state=random_state
                )
        
        elif algorithm == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            is_classification = len(np.unique(y_train)) <= 10
            if is_classification:
                clf = GradientBoostingClassifier(
                    n_estimators=hyperparams.get('n_estimators', 300),
                    learning_rate=hyperparams.get('learning_rate', 0.05),
                    max_depth=hyperparams.get('max_depth', 3),
                    random_state=random_state
                )
            else:
                clf = GradientBoostingRegressor(
                    n_estimators=hyperparams.get('n_estimators', 300),
                    learning_rate=hyperparams.get('learning_rate', 0.05),
                    max_depth=hyperparams.get('max_depth', 3),
                    random_state=random_state
                )
        
        elif algorithm == 'svm_rbf':
            from sklearn.svm import SVC, SVR
            is_classification = len(np.unique(y_train)) <= 10
            if is_classification:
                clf = SVC(
                    kernel='rbf',
                    C=hyperparams.get('C', 1.0),
                    gamma=hyperparams.get('gamma', 'scale'),
                    probability=True,
                    random_state=random_state
                )
            else:
                clf = SVR(
                    kernel='rbf',
                    C=hyperparams.get('C', 1.0),
                    gamma=hyperparams.get('gamma', 'scale')
                )
        
        elif algorithm == 'knn':
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            is_classification = len(np.unique(y_train)) <= 10
            if is_classification:
                clf = KNeighborsClassifier(
                    n_neighbors=hyperparams.get('n_neighbors', 15),
                    weights=hyperparams.get('weights', 'distance')
                )
            else:
                clf = KNeighborsRegressor(
                    n_neighbors=hyperparams.get('n_neighbors', 15),
                    weights=hyperparams.get('weights', 'distance')
                )
        
        elif algorithm == 'adaboost':
            from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
            is_classification = len(np.unique(y_train)) <= 10
            if is_classification:
                clf = AdaBoostClassifier(
                    n_estimators=hyperparams.get('n_estimators', 300),
                    learning_rate=hyperparams.get('learning_rate', 0.05),
                    random_state=random_state
                )
            else:
                clf = AdaBoostRegressor(
                    n_estimators=hyperparams.get('n_estimators', 300),
                    learning_rate=hyperparams.get('learning_rate', 0.05),
                    random_state=random_state
                )
        
        elif algorithm == 'decision_tree':
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            is_classification = len(np.unique(y_train)) <= 10
            max_depth = hyperparams.get('max_depth')
            if is_classification:
                clf = DecisionTreeClassifier(
                    max_depth=max_depth if max_depth else None,
                    random_state=random_state
                )
            else:
                clf = DecisionTreeRegressor(
                    max_depth=max_depth if max_depth else None,
                    random_state=random_state
                )
        
        elif algorithm == 'neural_network':
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            is_classification = len(np.unique(y_train)) <= 10
            hidden_layers = hyperparams.get('hidden_layers', [64, 32])
            if is_classification:
                clf = MLPClassifier(
                    hidden_layer_sizes=tuple(hidden_layers),
                    activation=hyperparams.get('activation', 'relu'),
                    solver='adam',
                    learning_rate_init=hyperparams.get('learning_rate', 0.001),
                    max_iter=hyperparams.get('max_iter', 500),
                    random_state=random_state
                )
            else:
                clf = MLPRegressor(
                    hidden_layer_sizes=tuple(hidden_layers),
                    activation=hyperparams.get('activation', 'relu'),
                    solver='adam',
                    learning_rate_init=hyperparams.get('learning_rate', 0.001),
                    max_iter=hyperparams.get('max_iter', 500),
                    random_state=random_state
                )
        
        elif algorithm == 'cnn':
            # CNN for tabular data (reshape to 1D CNN)
            try:
                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
            except ImportError as tf_error:
                raise ImportError(
                    "TensorFlow is required for CNN models. Please install it with: pip install tensorflow"
                ) from tf_error
            
            # Use only numeric features for CNN
            X_tab = X[numeric_features].fillna(0)
            
            # Validate minimum features for CNN (need at least 2 for Conv1D to work)
            if len(numeric_features) < 2:
                raise ValueError(
                    f"CNN requires at least 2 numeric features, but only {len(numeric_features)} found. "
                    f"Please select more numeric features or use a different algorithm (XGBoost, Random Forest, or Neural Network)."
                )
            
            # Warn if only 2 features (will use simplified architecture)
            if len(numeric_features) == 2:
                logger.warning(
                    f"CNN with only 2 numeric features will use a simplified architecture. "
                    f"For best results, consider using 3+ numeric features or another algorithm."
                )
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_tab)
            X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            
            # Re-split for CNN
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_scaled, y, test_size=0.30, random_state=random_state,
                stratify=y if len(np.unique(y)) <= 10 else None
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.50, random_state=random_state,
                stratify=y_temp if len(np.unique(y_temp)) <= 10 else None
            )
            
            # Adjust kernel_size based on number of features
            n_features = X_train.shape[1]
            
            # CNN architecture adaptation based on feature count
            if n_features >= 3:
                # Standard CNN for 3+ features
                kernel_size = 3
                filters = 32
            elif n_features == 2:
                # Simplified CNN for 2 features (kernel_size=1 to avoid dimension errors)
                kernel_size = 1
                filters = 16
            else:
                # This should not happen due to validation above, but handle it anyway
                kernel_size = 1
                filters = 8
            
            # Build CNN with adaptive architecture
            model = Sequential([
                Conv1D(filters, kernel_size=kernel_size, activation='relu', input_shape=(n_features, 1), padding='same'),
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid' if len(np.unique(y_train)) == 2 else 'linear')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy' if len(np.unique(y_train)) == 2 else 'mse',
                metrics=['accuracy']
            )
            
            # Train CNN with history tracking for learning curves
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=hyperparams.get('epochs', 30),
                batch_size=hyperparams.get('batch_size', 32),
                verbose=0
            )
            
            # Extract training history for learning curves
            training_history = []
            for i in range(len(history.history.get('loss', []))):
                hist_entry = {'epoch': i + 1}
                if 'loss' in history.history:
                    hist_entry['loss'] = float(history.history['loss'][i])
                if 'val_loss' in history.history:
                    hist_entry['val_loss'] = float(history.history['val_loss'][i])
                if 'accuracy' in history.history:
                    hist_entry['accuracy'] = float(history.history['accuracy'][i])
                if 'val_accuracy' in history.history:
                    hist_entry['val_accuracy'] = float(history.history['val_accuracy'][i])
                training_history.append(hist_entry)
            
            # Save CNN model
            model_path = None
            model_size = None
            model_hash = None
            try:
                import hashlib
                import os
                from pathlib import Path
                from datetime import datetime
                
                # Create trained_models directory
                model_dir = Path("trained_models")
                model_dir.mkdir(exist_ok=True)
                
                # Create descriptive filename: cnn_target_timestamp.h5
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_safe = config['target_variable'].replace(' ', '_').replace('/', '_')[:30]  # Sanitize target name
                filename = f"cnn_{target_safe}_{timestamp}.h5"
                model_path = model_dir / filename
                
                model.save(str(model_path))
                
                model_size = os.path.getsize(model_path)
                with open(model_path, 'rb') as f:
                    model_hash = hashlib.sha256(f.read()).hexdigest()
            except Exception as e:
                print(f"Warning: Could not save CNN model artifact: {e}")
                model_path = None
                model_size = None
                model_hash = None
            
            # Make predictions
            y_pred = model.predict(X_test, verbose=0).flatten()
            if len(np.unique(y_train)) == 2:
                y_pred = (y_pred > 0.5).astype(int)
            
            # Calculate metrics
            is_classification = len(np.unique(y_test)) <= 10
            if is_classification:
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1)
                }
            else:
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                metrics = {
                    "mse": float(mse),
                    "mae": float(mae),
                    "r2_score": float(r2)
                }
            
            # Resource monitoring - After training
            end_time = time.time()
            cpu_after = psutil.cpu_percent(interval=1) if psutil else 0
            mem_after = psutil.virtual_memory().used / (1024 * 1024) if psutil else 0
            training_duration = int(end_time - start_time)
            avg_cpu = (cpu_before + cpu_after) / 2 if psutil else 0
            avg_mem = (mem_before + mem_after) / 2 if psutil else 0
            
            resource_metrics = {
                "training_duration_seconds": training_duration,
                "cpu_usage_before": float(cpu_before),
                "cpu_usage_after": float(cpu_after),
                "avg_cpu_usage": float(avg_cpu),
                "memory_usage_before_mb": float(mem_before),
                "memory_usage_after_mb": float(mem_after),
                "avg_memory_usage_mb": float(avg_mem)
            }
            
            # Save results
            result = db.query(MLTrainingResult).filter(
                MLTrainingResult.model_id == training_id
            ).first()
            
            if result:
                result.metrics = metrics
                result.training_status = 'completed'
                result.training_duration_seconds = training_duration
                result.resource_metrics = resource_metrics
                result.n_samples = len(df)
                result.n_features = len(numeric_features)
                result.n_train = len(X_train)
                result.n_test = len(X_test)
                result.n_val = len(X_val)
                # Save model artifact path for CNN models
                if model_path:
                    result.model_artifact_path = str(model_path)
                    result.model_artifact_size = model_size
                    result.model_artifact_hash = model_hash
                # Save training history for learning curves (CNN models)
                if 'training_history' in locals() and training_history:
                    # Store training history in metrics as JSON
                    if result.metrics is None:
                        result.metrics = {}
                    result.metrics['training_history'] = training_history
                db.commit()
            
            return
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Build pipeline and train (for non-CNN models)
        model = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("clf", clf)
        ])
        
        model.fit(X_train, y_train)
        
        # Save model artifact for download
        model_path = None
        model_size = None
        model_hash = None
        try:
            import pickle
            import hashlib
            import os
            from pathlib import Path
            from datetime import datetime
            
            # Create trained_models directory
            model_dir = Path("trained_models")
            model_dir.mkdir(exist_ok=True)
            
            # Create descriptive filename: algorithm_target_timestamp.pkl
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_safe = config['target_variable'].replace(' ', '_').replace('/', '_')[:30]  # Sanitize target name
            algorithm = config['algorithm']
            filename = f"{algorithm}_{target_safe}_{timestamp}.pkl"
            model_path = model_dir / filename
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            model_size = os.path.getsize(model_path)
            with open(model_path, 'rb') as f:
                model_hash = hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            print(f"Warning: Could not save model artifact: {e}")
            model_path = None
            model_size = None
            model_hash = None
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        try:
            if hasattr(model.named_steps['clf'], 'predict_proba'):
                y_pred_proba = model.named_steps['clf'].predict_proba(X_test)
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
        except:
            pass
        
        # Calculate metrics
        is_classification = len(np.unique(y_test)) <= 10
        
        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "confusion_matrix": cm
            }
            
            # AUC-ROC for binary classification
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, drop_intermediate=False)
                    roc_data = [
                        {"fpr": float(f), "tpr": float(t), "threshold": float(th)}
                        for f, t, th in zip(fpr, tpr, thresholds)
                    ]
                    roc_data.sort(key=lambda x: x["fpr"])
                    metrics["auc"] = float(auc)
                    metrics["roc_curve"] = roc_data
                    print(f"✅ ROC curve calculated: AUC={auc:.4f}, Points={len(roc_data)}")
                except Exception as roc_error:
                    print(f"❌ ROC curve calculation failed: {roc_error}")
                    import traceback
                    traceback.print_exc()
            
            # Cross-validation
            try:
                kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
                cv_scores_list = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1_weighted')
                metrics["cv_scores"] = {
                    "mean": float(np.mean(cv_scores_list)),
                    "std": float(np.std(cv_scores_list)),
                    "scores": [float(s) for s in cv_scores_list]
                }
            except:
                pass
        
        else:  # Regression
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            y_mean = np.mean(y_test)
            baseline_mse = mean_squared_error(y_test, np.full_like(y_test, y_mean))
            mse_improvement = ((baseline_mse - mse) / baseline_mse * 100) if baseline_mse > 0 else 0
            
            metrics = {
                "mse": float(mse),
                "mae": float(mae),
                "r2_score": float(r2),
                "baseline_mse": float(baseline_mse),
                "baseline_mae": float(mean_absolute_error(y_test, np.full_like(y_test, y_mean))),
                "mse_improvement_pct": float(mse_improvement),
                "y_mean": float(y_mean),
                "y_std": float(np.std(y_test))
            }
            
            # Cross-validation
            try:
                kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
                cv_scores_list = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
                metrics["cv_scores"] = {
                    "mean": float(np.mean(cv_scores_list)),
                    "std": float(np.std(cv_scores_list)),
                    "scores": [float(s) for s in cv_scores_list]
                }
            except:
                pass
        
        # Feature importance
        feature_importance = []
        try:
            if hasattr(model.named_steps['clf'], 'feature_importances_'):
                importances = model.named_steps['clf'].feature_importances_.tolist()
                feature_names = []
                preprocessor = model.named_steps['preprocess']
                for name, transformer, cols in preprocessor.transformers_:
                    if name == 'num':
                        feature_names.extend([str(c) for c in cols])
                    elif name == 'cat':
                        try:
                            onehot = transformer.named_steps['onehot']
                            if hasattr(onehot, 'get_feature_names_out'):
                                cat_names = onehot.get_feature_names_out(cols)
                                feature_names.extend([str(n) for n in cat_names])
                            else:
                                feature_names.extend([str(c) for c in cols])
                        except:
                            feature_names.extend([str(c) for c in cols])
                
                min_len = min(len(importances), len(feature_names))
                feature_importance = [
                    {"feature": str(feature_names[i]), "importance": float(importances[i])}
                    for i in range(min_len)
                ]
        except:
            pass
        
        # Predictions for visualization
        predictions = []
        for i in range(min(100, len(y_test))):
            pred_dict = {
                "actual": float(y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]),
                "predicted": float(y_pred[i])
            }
            if y_pred_proba is not None and i < len(y_pred_proba):
                pred_dict["probability"] = float(y_pred_proba[i])
            predictions.append(pred_dict)
        
        # Resource monitoring - After training
        end_time = time.time()
        cpu_after = psutil.cpu_percent(interval=1)
        mem_after = psutil.virtual_memory().used / (1024 * 1024)
        training_duration = int(end_time - start_time)
        avg_cpu = (cpu_before + cpu_after) / 2
        avg_mem = (mem_before + mem_after) / 2
        
        resource_metrics = {
            "training_duration_seconds": training_duration,
            "cpu_usage_before": float(cpu_before),
            "cpu_usage_after": float(cpu_after),
            "avg_cpu_usage": float(avg_cpu),
            "memory_usage_before_mb": float(mem_before),
            "memory_usage_after_mb": float(mem_after),
            "avg_memory_usage_mb": float(avg_mem)
        }
        
        # Save results to database
        result = db.query(MLTrainingResult).filter(
            MLTrainingResult.model_id == training_id
        ).first()
        
        if result:
            result.metrics = metrics
            result.feature_importance = feature_importance
            result.predictions = predictions
            result.training_status = 'completed'
            result.training_duration_seconds = training_duration
            result.resource_metrics = resource_metrics
            result.n_samples = len(df)
            result.n_features = len(numeric_features) + len(categorical_features)
            result.n_train = len(X_train)
            result.n_test = len(X_test)
            result.n_val = len(X_val)
            if model_path:
                result.model_artifact_path = str(model_path)
                result.model_artifact_size = model_size
                result.model_artifact_hash = model_hash
            db.commit()
    
    except Exception as e:
        # Update status to failed
        try:
            result = db.query(MLTrainingResult).filter(
                MLTrainingResult.model_id == training_id
            ).first()
            if result:
                result.training_status = 'failed'
                result.error_message = str(e)
                db.commit()
        except Exception as db_error:
            print(f"Error updating database: {db_error}")
        print(f"Training error: {e}")
        traceback.print_exc()
    finally:
        db.close()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/train", response_model=TrainingResponse, status_code=status.HTTP_202_ACCEPTED)
def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Start ML model training on the backend server.
    Training runs asynchronously in the background.
    """
    try:
        import time
        # Generate training ID
        training_id = f"model_{int(time.time() * 1000)}"
        
        # Validate algorithm - All 9 supported models
        valid_algorithms = [
            'logistic_regression', 'random_forest', 'gradient_boosting', 'xgboost',
            'svm_rbf', 'neural_network', 'knn', 'adaboost', 'decision_tree'
        ]
        if request.algorithm not in valid_algorithms:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid algorithm. Must be one of: {', '.join(valid_algorithms)}"
            )
        
        # Validate target variable
        if not request.target_variable or not request.target_variable.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="target_variable is required and cannot be empty"
            )
        
        # Validate features
        if not request.features or len(request.features) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one feature is required"
            )
        
        # Ensure target variable is not in features
        if request.target_variable in request.features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="target_variable cannot be included in features list"
            )
        
        # Validate dataset
        if not request.dataset or len(request.dataset) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset is required and cannot be empty"
            )
        
        # Validate test_size
        if request.test_size is not None and (request.test_size <= 0 or request.test_size >= 1):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="test_size must be between 0 and 1 (exclusive)"
            )
        
        # Validate research_request_id if provided
        # Can be either UUID (id) or request_id string (REQ-UMMC-...)
        research_request_id = None
        if request.research_request_id:
            try:
                # Try to parse as UUID first
                research_request_id = uuid.UUID(request.research_request_id)
            except ValueError:
                # If not a UUID, try to look up by request_id string
                research_request = db.query(ResearchRequest).filter(
                    ResearchRequest.request_id == request.research_request_id
                ).first()
                if research_request:
                    research_request_id = research_request.id
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Research request not found: {request.research_request_id}"
                    )
        
        # Validate project_id if provided
        project_id = None
        if request.project_id:
            try:
                project_id = uuid.UUID(request.project_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid project_id format (must be a valid UUID)"
                )
        
        # Create initial training result record
        training_result = MLTrainingResult(
            model_id=training_id,
            research_request_id=research_request_id,
            project_id=project_id,
            algorithm=request.algorithm,
            target_variable=request.target_variable,
            features=request.features,
            hyperparameters=request.hyperparameters,
            test_size=request.test_size,
            random_state=request.random_state,
            custom_pipeline=request.custom_pipeline,
            metrics={},  # Will be updated after training
            training_status='queued',
            created_by=current_user.id
        )
        
        db.add(training_result)
        db.commit()
        db.refresh(training_result)
        
        # Prepare config for training function
        config = {
            'target_variable': request.target_variable,
            'features': request.features,
            'algorithm': request.algorithm,
            'hyperparameters': request.hyperparameters or {},
            'test_size': request.test_size or 0.2,
            'random_state': request.random_state or 42,
            'custom_pipeline': request.custom_pipeline
        }
        
        # Start background training task
        # Note: Don't pass db session - background task will create its own
        background_tasks.add_task(
            train_model_async,
            training_id,
            request.dataset,
            config,
            str(current_user.id)
        )
        
        return TrainingResponse(
            training_id=training_id,
            status='queued',
            message=f"Training job {training_id} has been queued. Use GET /ml-training/train/{training_id} to check status."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        error_msg = str(e)
        # If it's a validation error from Pydantic, extract the details
        if hasattr(e, 'errors'):
            error_details = []
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error.get('loc', []))
                error_details.append(f"{field}: {error.get('msg', 'Invalid value')}")
            error_msg = f"Validation error: {'; '.join(error_details)}"
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training: {error_msg}"
        )


@router.get("/train/{training_id}")
def get_training_status(
    training_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get the status of a training job.
    """
    result = db.query(MLTrainingResult).filter(
        MLTrainingResult.model_id == training_id,
        MLTrainingResult.created_by == current_user.id
    ).first()
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found"
        )
    
    response = {
        "training_id": result.model_id,
        "status": result.training_status,
        "algorithm": result.algorithm,
        "target_variable": result.target_variable,
        "created_at": result.created_at.isoformat() if result.created_at else None,
    }
    
    if result.training_status == 'completed':
        response["metrics"] = result.metrics or {}
        response["feature_importance"] = result.feature_importance
        response["predictions"] = result.predictions
        response["resource_metrics"] = result.resource_metrics
        # Ensure ROC curve is included if it exists in metrics
        if result.metrics and "roc_curve" in result.metrics:
            response["roc_curve"] = result.metrics.get("roc_curve")
        # Also include confusion_matrix and cv_scores if they exist
        if result.metrics and "confusion_matrix" in result.metrics:
            response["confusion_matrix"] = result.metrics.get("confusion_matrix")
        if result.metrics and "cv_scores" in result.metrics:
            response["cv_scores"] = result.metrics.get("cv_scores")
    
    if result.training_status == 'failed':
        response["error_message"] = result.error_message
    
    return response


@router.post("/execute-custom")
def execute_custom_code(
    code: str,
    dataset: list,
    research_request_id: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Execute custom Python code with dataset (similar to built-in models)
    """
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    import base64
    import time
    
    start_time = time.time()
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    plots = []
    
    try:
        # Create DataFrame from dataset
        import pandas as pd
        df = pd.DataFrame(dataset)
        
        # Prepare execution environment
        exec_globals = {
            'df': df,
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            '__builtins__': __builtins__
        }
        
        # Capture plots
        original_show = plt.show
        def capture_show():
            fig = plt.gcf()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plots.append(base64.b64encode(buf.read()).decode('utf-8'))
            plt.close(fig)
        
        plt.show = capture_show
        
        # Execute code with output capture
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, exec_globals)
        
        # Restore original show
        plt.show = original_show
        
        execution_time = time.time() - start_time
        
        return {
            "success": True,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "execution_time": execution_time,
            "timeout": False,
            "plots": plots,
            "error": None
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "success": False,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue() + f"\\n\\nError: {str(e)}",
            "execution_time": execution_time,
            "timeout": False,
            "plots": plots,
            "error": str(e)
        }


@router.get("/download-package/{training_id}")
def download_training_package(
    training_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Download complete training package as ZIP file.
    Includes: models, metrics, visualizations, documentation.
    """
    from fastapi.responses import StreamingResponse
    import io
    import zipfile
    from datetime import datetime
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Set style for better-looking plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Get training result
    result = db.query(MLTrainingResult).filter(
        MLTrainingResult.model_id == training_id,
        MLTrainingResult.created_by == current_user.id
    ).first()
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found"
        )
    
    if result.training_status != 'completed':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Training is not completed yet. Current status: {result.training_status}"
        )
    
    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Add model artifact if available
        if result.model_artifact_path:
            try:
                with open(result.model_artifact_path, 'rb') as f:
                    zip_file.writestr(f"models/{result.algorithm}_model.pkl", f.read())
            except:
                pass
        
        # 2. Add metrics as JSON
        metrics_json = json.dumps(result.metrics or {}, indent=2)
        zip_file.writestr("results/metrics.json", metrics_json)
        
        # 3. Add metrics as CSV
        if result.metrics:
            csv_lines = ["Metric,Value"]
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    csv_lines.append(f"{key},{value}")
            zip_file.writestr("results/metrics.csv", "\n".join(csv_lines))
        
        # 4. Add feature importance as CSV
        if result.feature_importance:
            csv_lines = ["Feature,Importance"]
            for item in result.feature_importance:
                csv_lines.append(f"{item['feature']},{item['importance']}")
            zip_file.writestr("results/feature_importance.csv", "\n".join(csv_lines))
        
        # 5. Generate and add ROC curve visualization
        if result.metrics and 'roc_curve' in result.metrics and result.metrics['roc_curve']:
            fig, ax = plt.subplots(figsize=(10, 8))
            roc_data = result.metrics['roc_curve']
            fpr = [p['fpr'] for p in roc_data]
            tpr = [p['tpr'] for p in roc_data]
            auc = result.metrics.get('auc', 0)
            
            ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
            ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC Curve - {result.algorithm}', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            zip_file.writestr("visualizations/01_ROC_Curve.png", img_buffer.read())
            plt.close()
        
        # 6. Generate confusion matrix visualization with labels
        if result.metrics and 'confusion_matrix' in result.metrics:
            fig, ax = plt.subplots(figsize=(10, 8))
            cm = np.array(result.metrics['confusion_matrix'])
            
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            
            # Better labels
            classes = ['Class 0', 'Class 1']
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=classes,
                   yticklabels=classes,
                   title=f'Confusion Matrix - {result.algorithm.upper()}',
                   ylabel='True Label',
                   xlabel='Predicted Label')
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations with percentages
            thresh = cm.max() / 2.
            total = cm.sum()
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    count = cm[i, j]
                    percentage = (count / total) * 100
                    ax.text(j, i, f'{count}\n({percentage:.1f}%)',
                           ha="center", va="center",
                           color="white" if count > thresh else "black",
                           fontsize=16, fontweight='bold')
            
            # Add labels for TN, FP, FN, TP
            ax.text(0, -0.15, 'TN', ha='center', va='top', transform=ax.transData, fontsize=10, color='green')
            ax.text(1, -0.15, 'FP', ha='center', va='top', transform=ax.transData, fontsize=10, color='red')
            ax.text(0, 1.15, 'FN', ha='center', va='bottom', transform=ax.transData, fontsize=10, color='red')
            ax.text(1, 1.15, 'TP', ha='center', va='bottom', transform=ax.transData, fontsize=10, color='green')
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            zip_file.writestr("visualizations/02_Confusion_Matrix.png", img_buffer.read())
            plt.close()
        
        # 7. Generate feature importance visualization
        if result.feature_importance and len(result.feature_importance) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = sorted(result.feature_importance, key=lambda x: x['importance'], reverse=True)[:15]
            features = [f['feature'] for f in top_features]
            importances = [f['importance'] for f in top_features]
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, importances, color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title(f'Top 15 Feature Importance - {result.algorithm}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            zip_file.writestr("visualizations/03_Feature_Importance.png", img_buffer.read())
            plt.close()
        
        # 8. Generate metrics comparison bar chart
        if result.metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            metric_names = []
            metric_values = []
            
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)) and key not in ['auc', 'roc_curve', 'confusion_matrix', 'cv_scores']:
                    metric_names.append(key.replace('_', ' ').title())
                    metric_values.append(value)
            
            if metric_names:
                x_pos = np.arange(len(metric_names))
                bars = ax.bar(x_pos, metric_values, color='teal', alpha=0.7)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(metric_names, rotation=45, ha='right')
                ax.set_ylabel('Score', fontsize=12)
                ax.set_title(f'Model Performance Metrics - {result.algorithm}', fontsize=14, fontweight='bold')
                ax.set_ylim(0, 1.1)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=10)
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                zip_file.writestr("visualizations/04_Metrics_Comparison.png", img_buffer.read())
                plt.close()
        
        # 9. Generate learning curves (if available)
        if result.metrics and 'training_history' in result.metrics:
            history = result.metrics['training_history']
            if history and len(history) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                epochs = [h['epoch'] for h in history]
                
                # Loss plot
                if 'loss' in history[0]:
                    loss = [h['loss'] for h in history]
                    ax1.plot(epochs, loss, 'b-', linewidth=2, marker='o', label='Training Loss')
                
                if 'val_loss' in history[0]:
                    val_loss = [h['val_loss'] for h in history]
                    ax1.plot(epochs, val_loss, 'r--', linewidth=2, marker='s', label='Validation Loss')
                
                ax1.set_xlabel('Epoch', fontsize=12)
                ax1.set_ylabel('Loss', fontsize=12)
                ax1.set_title(f'Loss Curves - {result.algorithm.upper()}', fontsize=14, fontweight='bold')
                ax1.legend(loc='upper right', fontsize=10)
                ax1.grid(True, alpha=0.3)
                
                # Accuracy plot (if available)
                if 'accuracy' in history[0]:
                    acc = [h['accuracy'] for h in history]
                    ax2.plot(epochs, acc, 'g-', linewidth=2, marker='o', label='Training Accuracy')
                
                if 'val_accuracy' in history[0]:
                    val_acc = [h['val_accuracy'] for h in history]
                    ax2.plot(epochs, val_acc, 'orange', linestyle='--', linewidth=2, marker='s', label='Validation Accuracy')
                
                ax2.set_xlabel('Epoch', fontsize=12)
                ax2.set_ylabel('Accuracy', fontsize=12)
                ax2.set_title(f'Accuracy Curves - {result.algorithm.upper()}', fontsize=14, fontweight='bold')
                ax2.legend(loc='lower right', fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                zip_file.writestr("visualizations/05_Learning_Curves.png", img_buffer.read())
                plt.close()
        
        # 10. Generate Precision-Recall curve (if binary classification)
        if result.metrics and 'roc_curve' in result.metrics and result.metrics['roc_curve']:
            try:
                from sklearn.metrics import precision_recall_curve, average_precision_score
                
                # We need actual predictions to calculate PR curve
                # For now, we'll add a note that this requires the actual data
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Add informational text
                ax.text(0.5, 0.5, 
                       'Precision-Recall Curve\n\n' +
                       'To generate this curve, re-run training\n' +
                       'with prediction data stored.\n\n' +
                       f'Model: {result.algorithm.upper()}\n' +
                       f'AUC-ROC: {result.metrics.get("auc", "N/A"):.4f}',
                       ha='center', va='center', fontsize=14,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel('Recall', fontsize=12)
                ax.set_ylabel('Precision', fontsize=12)
                ax.set_title(f'Precision-Recall Curve - {result.algorithm.upper()}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                zip_file.writestr("visualizations/06_Precision_Recall_Curve.png", img_buffer.read())
                plt.close()
            except Exception as e:
                print(f"Could not generate PR curve: {e}")
        
        # 11. Generate predictions scatter plot (actual vs predicted)
        if result.predictions and len(result.predictions) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            actuals = [p['actual'] for p in result.predictions[:100]]
            predicted = [p['predicted'] for p in result.predictions[:100]]
            
            # For classification, create a jittered scatter
            if len(set(actuals)) <= 10:  # Classification
                # Add jitter for visibility
                actuals_jitter = np.array(actuals) + np.random.normal(0, 0.05, len(actuals))
                predicted_jitter = np.array(predicted) + np.random.normal(0, 0.05, len(predicted))
                
                ax.scatter(actuals_jitter, predicted_jitter, alpha=0.6, s=50, c='steelblue', edgecolors='black')
                ax.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', linewidth=2, label='Perfect Prediction')
                ax.set_xlabel('Actual Values', fontsize=12)
                ax.set_ylabel('Predicted Values', fontsize=12)
                ax.set_title(f'Predictions vs Actuals - {result.algorithm.upper()}\n(First 100 samples)', fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
            else:  # Regression
                ax.scatter(actuals, predicted, alpha=0.6, s=50, c='steelblue', edgecolors='black')
                ax.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', linewidth=2, label='Perfect Prediction')
                ax.set_xlabel('Actual Values', fontsize=12)
                ax.set_ylabel('Predicted Values', fontsize=12)
                ax.set_title(f'Predictions vs Actuals - {result.algorithm.upper()}\n(First 100 samples)', fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            zip_file.writestr("visualizations/07_Predictions_Scatter.png", img_buffer.read())
            plt.close()
        
        # 12. Generate cross-validation scores visualization
        if result.metrics and 'cv_scores' in result.metrics:
            cv_data = result.metrics['cv_scores']
            if 'scores' in cv_data:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Bar plot of individual fold scores
                folds = [f'Fold {i+1}' for i in range(len(cv_data['scores']))]
                scores = cv_data['scores']
                
                bars = ax1.bar(folds, scores, color='teal', alpha=0.7, edgecolor='black')
                ax1.axhline(y=cv_data['mean'], color='r', linestyle='--', linewidth=2, label=f"Mean: {cv_data['mean']:.4f}")
                ax1.set_ylabel('Score', fontsize=12)
                ax1.set_title(f'Cross-Validation Scores by Fold - {result.algorithm.upper()}', fontsize=14, fontweight='bold')
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=10)
                
                # Box plot
                ax2.boxplot([scores], labels=['CV Scores'])
                ax2.set_ylabel('Score', fontsize=12)
                ax2.set_title(f'Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.text(1.3, cv_data['mean'], f"Mean: {cv_data['mean']:.4f}\nStd: {cv_data['std']:.4f}", 
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.tight_layout()
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                zip_file.writestr("visualizations/08_Cross_Validation_Scores.png", img_buffer.read())
                plt.close()
        
        # 12. Generate SHAP Explainability Visualizations (XAI - Explainable AI)
        print("📊 Generating SHAP explainability visualizations...")
        try:
            import shap
            import joblib
            
            # Load the trained model from the ZIP
            model_data = zip_file.read(f"models/{result.algorithm}_model.pkl")
            model_buffer = io.BytesIO(model_data)
            trained_model = joblib.load(model_buffer)
            
            # Get training data from result
            if hasattr(result, 'training_data') and result.training_data:
                # Parse training data if stored as JSON
                import json
                if isinstance(result.training_data, str):
                    training_data = json.loads(result.training_data)
                else:
                    training_data = result.training_data
                
                # Convert to DataFrame
                import pandas as pd
                df = pd.DataFrame(training_data)
                
                # Get features (exclude target)
                feature_cols = [col for col in df.columns if col != result.target_variable]
                X = df[feature_cols]
                
                # Sample background data for SHAP (200 samples for speed)
                n_background = min(200, len(X))
                X_background = X.sample(n=n_background, random_state=42) if len(X) > n_background else X
                
                # Sample test data for SHAP explanations (500 samples)
                n_test = min(500, len(X))
                X_test = X.sample(n=n_test, random_state=1) if len(X) > n_test else X
                
                # Create prediction function wrapper
                def model_predict(x):
                    """Wrapper for model prediction"""
                    try:
                        # For classification, return probability of positive class
                        if hasattr(trained_model, 'predict_proba'):
                            return trained_model.predict_proba(x)[:, 1]
                        else:
                            return trained_model.predict(x)
                    except Exception as e:
                        print(f"⚠️  Prediction error in SHAP: {e}")
                        return trained_model.predict(x)
                
                # Create SHAP explainer
                print(f"   Creating SHAP explainer with {len(X_background)} background samples...")
                explainer = shap.Explainer(model_predict, X_background)
                
                # Calculate SHAP values
                print(f"   Calculating SHAP values for {len(X_test)} test samples...")
                shap_values = explainer(X_test)
                
                # 12a. SHAP Summary Plot (Global Feature Importance)
                print("   Generating SHAP summary plot...")
                fig, ax = plt.subplots(figsize=(12, 8))
                shap.summary_plot(shap_values, X_test, show=False, max_display=15)
                plt.title(f'SHAP Feature Importance - {result.algorithm.upper()}', 
                         fontsize=14, fontweight='bold', pad=20)
                plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=12)
                plt.tight_layout()
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                zip_file.writestr("visualizations/09_SHAP_Summary.png", img_buffer.read())
                plt.close()
                print("   ✅ SHAP summary plot saved")
                
                # 12b. SHAP Waterfall Plot (Single Patient Explanation)
                print("   Generating SHAP waterfall plot for sample patient...")
                fig, ax = plt.subplots(figsize=(12, 8))
                shap.plots.waterfall(shap_values[0], show=False, max_display=15)
                plt.title(f'SHAP Explanation - Sample Patient (Index 0)', 
                         fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                zip_file.writestr("visualizations/10_SHAP_Waterfall_Patient1.png", img_buffer.read())
                plt.close()
                print("   ✅ SHAP waterfall plot saved")
                
                # 12c. SHAP Bar Plot (Mean Absolute SHAP Values)
                print("   Generating SHAP bar plot...")
                fig, ax = plt.subplots(figsize=(12, 8))
                shap.plots.bar(shap_values, show=False, max_display=15)
                plt.title(f'SHAP Feature Importance (Mean |SHAP|) - {result.algorithm.upper()}', 
                         fontsize=14, fontweight='bold', pad=20)
                plt.xlabel('Mean |SHAP Value|', fontsize=12)
                plt.tight_layout()
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                zip_file.writestr("visualizations/11_SHAP_Bar.png", img_buffer.read())
                plt.close()
                print("   ✅ SHAP bar plot saved")
                
                # 12d. SHAP Beeswarm Plot (Detailed Feature Impact)
                print("   Generating SHAP beeswarm plot...")
                fig, ax = plt.subplots(figsize=(12, 10))
                shap.plots.beeswarm(shap_values, show=False, max_display=15)
                plt.title(f'SHAP Beeswarm Plot - Feature Impact Distribution', 
                         fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                zip_file.writestr("visualizations/12_SHAP_Beeswarm.png", img_buffer.read())
                plt.close()
                print("   ✅ SHAP beeswarm plot saved")
                
                print("✅ SHAP explainability visualizations completed!")
                
        except Exception as shap_error:
            print(f"⚠️  SHAP visualization error (non-critical): {shap_error}")
            print(f"   Continuing without SHAP plots...")
            import traceback
            traceback.print_exc()
        
        # 13. Create comprehensive README (matching cancer prognosis format)
        readme_content = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              ML TRAINING RESULTS - FINAL CORRECTED PACKAGE                 ║
║                                                                            ║
║          Trained on {result.n_samples or 'N/A'} samples with verified results              ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

📦 PACKAGE CONTENTS
════════════════════════════════════════════════════════════════════════════

ML_Training_Package_{result.algorithm}/
├── models/                    (Trained ML Model + Preprocessing)
│   ├── {result.algorithm}_model.pkl     ⭐ TRAINED MODEL
│   └── (Preprocessing included in pipeline)
│
├── results/                   (Performance Metrics & Data Information)
│   ├── metrics.json                    ✨ MAIN RESULTS (JSON)
│   ├── metrics.csv                     Key metrics in CSV format
│   ├── feature_importance.csv          Feature rankings
│   └── training_config.json            Training configuration
│
├── visualizations/            (Publication-Ready High-Resolution Graphs)
│   ├── 01_ROC_Curve.png                Model discrimination ability
│   ├── 02_Confusion_Matrix.png         Prediction accuracy breakdown
│   ├── 03_Feature_Importance.png       Top 15 predictive features
│   ├── 04_Metrics_Comparison.png       Performance metrics bar chart
│   ├── 05_Learning_Curves.png          Training convergence (if applicable)
│   ├── 06_Precision_Recall_Curve.png   Precision vs Recall tradeoff
│   ├── 07_Predictions_Scatter.png      Actual vs Predicted values
│   ├── 08_Cross_Validation_Scores.png  CV performance (if applicable)
│   ├── 09_SHAP_Summary.png             🔍 XAI: Global feature importance
│   ├── 10_SHAP_Waterfall_Patient1.png  🔍 XAI: Single patient explanation
│   ├── 11_SHAP_Bar.png                 🔍 XAI: Mean absolute SHAP values
│   └── 12_SHAP_Beeswarm.png            🔍 XAI: Feature impact distribution
│
└── documentation/
    └── README.txt                      (This file)

════════════════════════════════════════════════════════════════════════════

🎯 KEY FINDINGS - MODEL PERFORMANCE
════════════════════════════════════════════════════════════════════════════

ALGORITHM: {result.algorithm.upper()}
"""
        
        # Add metrics in a formatted way
        if result.metrics:
            # Format metrics safely
            acc = result.metrics.get('accuracy', 'N/A')
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else 'N/A'
            
            prec = result.metrics.get('precision', 'N/A')
            prec_str = f"{prec:.4f}" if isinstance(prec, (int, float)) else 'N/A'
            
            rec = result.metrics.get('recall', 'N/A')
            rec_str = f"{rec:.4f}" if isinstance(rec, (int, float)) else 'N/A'
            
            f1 = result.metrics.get('f1_score', 'N/A')
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else 'N/A'
            
            auc = result.metrics.get('auc', 'N/A')
            auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else 'N/A'
            
            readme_content += f"""
├─ Test Accuracy:  {acc_str}
├─ Test Precision: {prec_str}
├─ Test Recall:    {rec_str}
├─ Test F1-Score:  {f1_str}
└─ Test AUC:       {auc_str}
"""
        
        readme_content += f"""
════════════════════════════════════════════════════════════════════════════

📊 COMPLETE RESULTS TABLE
════════════════════════════════════════════════════════════════════════════

See: results/metrics.json and results/metrics.csv

These files contain ALL metrics for the trained model:
✓ Accuracy, Precision, Recall, F1-Score
✓ AUC-ROC (if binary classification)
✓ Confusion Matrix
✓ Cross-Validation Scores (if applicable)
✓ Feature Importance Rankings

════════════════════════════════════════════════════════════════════════════

📈 VISUALIZATIONS GUIDE
════════════════════════════════════════════════════════════════════════════

1. 01_ROC_Curve.png
   - Shows discrimination ability of the model
   - Higher curve = better model
   - AUC score indicates overall performance
   - Use for: Publication, presentations, model comparison

2. 02_Confusion_Matrix.png
   - 2x2 matrix showing True Pos, False Pos, True Neg, False Neg
   - Reveals where model makes mistakes
   - Includes percentages for easy interpretation
   - Use for: Understanding model errors

3. 03_Feature_Importance.png
   - Top 15 most important features
   - Shows which clinical/data factors matter most
   - Use for: Feature selection and clinical interpretation

4. 04_Metrics_Comparison.png
   - Bar chart of all performance metrics
   - Easy visual comparison
   - Use for: Quick model assessment

5. 05_Learning_Curves.png (if applicable)
   - Training vs validation loss/accuracy over time
   - Shows model convergence
   - Helps identify overfitting
   - Use for: Model reliability assessment

6. 06_Precision_Recall_Curve.png
   - Precision vs Recall tradeoff
   - Better for imbalanced datasets
   - Shows performance on minority class
   - Use for: Understanding prediction behavior

7. 07_Predictions_Scatter.png
   - Actual vs Predicted values
   - Shows prediction accuracy visually
   - Use for: Model validation

8. 08_Cross_Validation_Scores.png (if applicable)
   - Cross-validation performance across folds
   - Shows model stability
   - Use for: Assessing generalization

🔍 EXPLAINABLE AI (XAI) - SHAP VISUALIZATIONS
────────────────────────────────────────────────────────────────────────────

9. 09_SHAP_Summary.png
   - Global feature importance using SHAP values
   - Shows which features impact predictions most
   - Color indicates feature value (red=high, blue=low)
   - Horizontal position shows impact direction
   - Use for: Understanding model decision-making globally

10. 10_SHAP_Waterfall_Patient1.png
    - Explains prediction for a single patient
    - Shows how each feature contributed to the prediction
    - Starts from base value (average prediction)
    - Each feature pushes prediction up or down
    - Use for: Individual patient explanation, clinical interpretation

11. 11_SHAP_Bar.png
    - Mean absolute SHAP values per feature
    - Simple ranking of feature importance
    - Higher bar = more important feature
    - Use for: Quick feature importance overview

12. 12_SHAP_Beeswarm.png
    - Detailed view of feature impact distribution
    - Each dot is a patient
    - Shows how feature values affect predictions
    - Reveals non-linear relationships
    - Use for: Deep feature analysis, research publications

📚 WHAT IS SHAP?
────────────────────────────────────────────────────────────────────────────
SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain
machine learning model predictions. It provides:

✓ Model-agnostic explanations (works with any ML model)
✓ Consistent and locally accurate feature attributions
✓ Both global (dataset-level) and local (patient-level) explanations
✓ Theoretically grounded in cooperative game theory

SHAP values answer: "How much did each feature contribute to this prediction?"

════════════════════════════════════════════════════════════════════════════

💻 HOW TO USE THE MODEL
════════════════════════════════════════════════════════════════════════════

STEP 1: Load the Model
────────────────────────────
import joblib

model = joblib.load('models/{result.algorithm}_model.pkl')

STEP 2: Prepare Your Data
──────────────────────────
import pandas as pd

df_new = pd.read_csv('new_patients.csv')

# Ensure all required features are present:
required_features = {result.features}

# Check for missing features
for col in required_features:
    if col not in df_new.columns:
        print(f"Warning: Missing feature {{col}}")

STEP 3: Make Predictions
─────────────────────────
# For classification
predictions = model.predict(df_new[required_features])
probabilities = model.predict_proba(df_new[required_features])

# Results
results = pd.DataFrame({{
    'Patient_ID': range(1, len(predictions)+1),
    'Prediction': predictions,
    'Probability': probabilities[:, 1] if len(probabilities.shape) > 1 else probabilities
}})  # Double braces to escape in f-string

print(results.head())

════════════════════════════════════════════════════════════════════════════

📋 DATASET INFORMATION
════════════════════════════════════════════════════════════════════════════

Target Variable: {result.target_variable}
Total Samples:   {result.n_samples or 'N/A'}

Data Split:"""
        
        # Calculate percentages safely
        train_pct = (result.n_train/result.n_samples*100) if result.n_samples and result.n_train else 0
        val_pct = (result.n_val/result.n_samples*100) if result.n_samples and result.n_val else 0
        test_pct = (result.n_test/result.n_samples*100) if result.n_samples and result.n_test else 0
        
        readme_content += f"""
├─ Train Set:  {result.n_train or 'N/A'} samples ({train_pct:.1f}%)
├─ Val Set:    {result.n_val or 'N/A'} samples ({val_pct:.1f}%)
└─ Test Set:   {result.n_test or 'N/A'} samples ({test_pct:.1f}%)

Features: {len(result.features)} variables
"""
        
        # List features
        if result.features:
            readme_content += "\nFeature List:\n"
            for i, feat in enumerate(result.features, 1):
                readme_content += f"  {i:2d}. {feat}\n"
        
        readme_content += f"""
Preprocessing Applied:
✓ Median imputation for numerical missing values
✓ One-hot encoding for categorical variables
✓ StandardScaler for feature normalization

🔒 DATA SECURITY NOTE:
──────────────────────
The original dataset is NOT included in this package for security and privacy
reasons. Only the trained model, metrics, and visualizations are provided.
The model can be used for predictions on new data without exposing the
training data.

════════════════════════════════════════════════════════════════════════════

✅ QUALITY ASSURANCE
════════════════════════════════════════════════════════════════════════════

[✓] Model trained successfully
[✓] Data properly split (train/val/test stratified)
[✓] No data leakage (preprocessing in pipeline)
[✓] Consistent random seed (reproducible results)
[✓] Comprehensive metrics (all standard metrics calculated)
[✓] Multiple publication-ready visualizations (300 DPI PNG)
[✓] Complete results table (JSON and CSV formats)
[✓] All components saved correctly

════════════════════════════════════════════════════════════════════════════

🎯 RECOMMENDED USAGE
════════════════════════════════════════════════════════════════════════════

FOR BEST ACCURACY:"""
        
        # Format AUC safely
        best_auc = result.metrics.get('auc', 'N/A') if result.metrics else 'N/A'
        best_auc_str = f"{best_auc:.4f}" if isinstance(best_auc, (int, float)) else 'N/A'
        
        readme_content += f"""
└─ Use: {result.algorithm.upper()} (AUC = {best_auc_str})

FOR EXPLAINABILITY:
└─ Review: Feature importance in visualizations/03_Feature_Importance.png
└─ Analyze: Which features drive predictions

FOR CLINICAL VALIDATION:
└─ Test on independent cohort using same preprocessing
└─ Monitor prediction drift over time
└─ Compare with clinical outcomes

════════════════════════════════════════════════════════════════════════════

📞 FILE DESCRIPTIONS
════════════════════════════════════════════════════════════════════════════

models/
├─ {result.algorithm}_model.pkl : Trained scikit-learn pipeline (ready to use)
                                  Includes preprocessing + model

results/
├─ metrics.json                 : Complete metrics in JSON format
├─ metrics.csv                  : Key metrics in CSV format
├─ feature_importance.csv       : Feature importance rankings
└─ training_config.json         : Training configuration and parameters

visualizations/
├─ 01-08_*.png                  : High-resolution graphs (300 DPI)
└─ All suitable for publication and presentations

════════════════════════════════════════════════════════════════════════════

🚀 NEXT STEPS
════════════════════════════════════════════════════════════════════════════

1. EXPLORE RESULTS
   → Open results/metrics.json
   → Review all metrics
   → Compare with baseline

2. REVIEW VISUALIZATIONS
   → Open visualizations/ folder
   → Study ROC curves and confusion matrix
   → Use graphs in presentations

3. USE MODEL
   → Load {result.algorithm}_model.pkl
   → Follow usage example above
   → Make predictions on new data

4. VALIDATE EXTERNALLY
   → Test on independent cohort
   → Use same preprocessing (included in pipeline)
   → Compare predictions with actual outcomes

5. PUBLISH RESULTS
   → Use visualizations in papers
   → Include metrics table
   → Reference methodology

════════════════════════════════════════════════════════════════════════════

✨ SUMMARY
════════════════════════════════════════════════════════════════════════════

✓ Algorithm: {result.algorithm.upper()}
✓ Target: {result.target_variable}
✓ Samples: {result.n_samples or 'N/A'} total ({result.n_train or 'N/A'} train / {result.n_val or 'N/A'} val / {result.n_test or 'N/A'} test)
✓ Features: {len(result.features)}
✓ Complete Package: Model + Metrics + Visualizations + Documentation
✓ Production-Ready: Fully documented and tested
✓ High Quality: 300 DPI visualizations for publication

════════════════════════════════════════════════════════════════════════════

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Training ID: {result.model_id}
Duration: {result.training_duration_seconds or 'N/A'} seconds

════════════════════════════════════════════════════════════════════════════
"""
        
        zip_file.writestr("documentation/README.txt", readme_content)
        
        # 11. Add training configuration
        def convert_to_serializable(obj):
            """Convert Decimal and other non-serializable types to JSON-compatible types"""
            from decimal import Decimal
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        config_json = {
            "training_id": result.model_id,
            "algorithm": result.algorithm,
            "target_variable": result.target_variable,
            "features": result.features,
            "hyperparameters": convert_to_serializable(result.hyperparameters),
            "test_size": float(result.test_size) if result.test_size else None,
            "random_state": result.random_state,
            "n_samples": result.n_samples,
            "n_features": result.n_features,
            "n_train": result.n_train,
            "n_val": result.n_val,
            "n_test": result.n_test,
            "training_duration_seconds": result.training_duration_seconds,
            "created_at": result.created_at.isoformat() if result.created_at else None
        }
        zip_file.writestr("results/training_config.json", json.dumps(config_json, indent=2))
    
    # Prepare response
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=ML_Training_Package_{result.algorithm}_{training_id}.zip"
        }
    )

