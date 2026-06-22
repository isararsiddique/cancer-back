"""
Offline Training Service — GPU-accelerated model training for the cancer registry.

Inspired by CDCE ml-engine: supports PyTorch, TensorFlow, scikit-learn, XGBoost.
Trains models on the server's local GPU, tracks artifacts, versioning, and metrics.
Falls back to CPU gracefully when no GPU is available.

Supported algorithms:
- PyTorch: Deep neural networks (MLP, CNN for tabular), with CUDA acceleration
- TensorFlow/Keras: Sequential/functional models with GPU memory growth
- XGBoost: GPU-accelerated gradient boosting (tree_method='gpu_hist')
- scikit-learn: CPU-based (RandomForest, GradientBoosting, SVM, etc.)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from services.gpu_manager import get_gpu_manager

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("ML_MODEL_DIR", "trained_models"))
MODEL_DIR.mkdir(exist_ok=True)


# ─── Training Result Dataclass ───────────────────────────────────────────────


class TrainingResult:
    """Encapsulates the output of a training run."""

    def __init__(
        self,
        job_id: str,
        algorithm: str,
        framework: str,
        device_used: str,
        metrics: Dict[str, Any],
        feature_importance: Optional[List[Dict[str, Any]]] = None,
        training_history: Optional[List[Dict[str, Any]]] = None,
        model_artifact_path: Optional[str] = None,
        model_artifact_size: Optional[int] = None,
        model_artifact_hash: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        duration_seconds: float = 0,
        n_samples: int = 0,
        n_features: int = 0,
        n_train: int = 0,
        n_test: int = 0,
        gpu_memory_used_mb: float = 0,
        error: Optional[str] = None,
    ):
        self.job_id = job_id
        self.algorithm = algorithm
        self.framework = framework
        self.device_used = device_used
        self.metrics = metrics
        self.feature_importance = feature_importance or []
        self.training_history = training_history or []
        self.model_artifact_path = model_artifact_path
        self.model_artifact_size = model_artifact_size
        self.model_artifact_hash = model_artifact_hash
        self.hyperparameters = hyperparameters or {}
        self.duration_seconds = duration_seconds
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_train = n_train
        self.n_test = n_test
        self.gpu_memory_used_mb = gpu_memory_used_mb
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "algorithm": self.algorithm,
            "framework": self.framework,
            "device_used": self.device_used,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "training_history": self.training_history,
            "model_artifact_path": self.model_artifact_path,
            "model_artifact_size": self.model_artifact_size,
            "model_artifact_hash": self.model_artifact_hash,
            "hyperparameters": self.hyperparameters,
            "duration_seconds": self.duration_seconds,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "error": self.error,
        }


# ─── Data Preprocessing ─────────────────────────────────────────────────────


def _preprocess_data(
    dataset: List[Dict[str, Any]],
    target_variable: str,
    features: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Any, Any, Any, Any, Any, Any, List[str], List[str], bool]:
    """
    Preprocess dataset into train/val/test splits with proper encoding.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test,
             numeric_features, categorical_features, is_classification
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    df = pd.DataFrame(dataset)
    df = df.replace(["", None, "None", "null"], np.nan)

    feature_cols = [f for f in features if f != target_variable and f in df.columns]
    if not feature_cols:
        raise ValueError("No valid features found in dataset")

    X = df[feature_cols].copy()
    y = df[target_variable].copy()

    # Handle target
    y = y.replace([None, "None", "null"], np.nan)
    n_unique = y.dropna().nunique()
    is_classification = n_unique <= 12 or not pd.api.types.is_numeric_dtype(y)

    if is_classification:
        le = LabelEncoder()
        y = y.fillna("__MISSING__").astype(str)
        y = pd.Series(le.fit_transform(y))
    else:
        y = pd.to_numeric(y, errors="coerce")
        y = y.fillna(y.median() if not pd.isna(y.median()) else 0)

    # Identify feature types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in categorical_features:
        X[col] = X[col].fillna("__MISSING__").astype(str)

    # Remove constant columns
    for col in numeric_features + categorical_features:
        if X[col].nunique() <= 1:
            X = X.drop(columns=[col])
            if col in numeric_features:
                numeric_features.remove(col)
            if col in categorical_features:
                categorical_features.remove(col)

    # Split: 70/15/15
    stratify = y if (is_classification and y.value_counts().min() >= 2) else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=stratify
    )
    stratify_temp = y_temp if (is_classification and y_temp.value_counts().min() >= 2) else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=stratify_temp
    )

    return (X_train, X_val, X_test, y_train, y_val, y_test,
            numeric_features, categorical_features, is_classification)


# ─── PyTorch GPU Training ────────────────────────────────────────────────────


def _train_pytorch(
    X_train, X_val, X_test, y_train, y_val, y_test,
    numeric_features: List[str],
    categorical_features: List[str],
    is_classification: bool,
    hyperparams: Dict[str, Any],
    device_str: str,
    job_id: str,
) -> TrainingResult:
    """Train a PyTorch model on GPU/CPU."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
    )

    start_time = time.time()
    device = torch.device(device_str)
    logger.info(f"[{job_id}] PyTorch training on device: {device}")

    # Preprocess features into numeric tensors
    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_features))

    if not transformers:
        raise ValueError("No valid features for training")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_train_np = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_np = preprocessor.transform(X_val).astype(np.float32)
    X_test_np = preprocessor.transform(X_test).astype(np.float32)

    n_input = X_train_np.shape[1]
    n_classes = len(np.unique(y_train)) if is_classification else 1

    # Convert to tensors
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)

    if is_classification:
        y_train_t = torch.tensor(y_train.values, dtype=torch.long)
        y_val_t = torch.tensor(y_val.values, dtype=torch.long)
        y_test_t = torch.tensor(y_test.values, dtype=torch.long)
    else:
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
        y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # DataLoaders
    batch_size = hyperparams.get("batch_size", 64)
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Build model architecture
    hidden_layers = hyperparams.get("hidden_layers", [128, 64, 32])
    dropout = hyperparams.get("dropout", 0.3)

    layers = []
    prev_size = n_input
    for h in hidden_layers:
        layers.append(nn.Linear(prev_size, h))
        layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        prev_size = h

    if is_classification:
        layers.append(nn.Linear(prev_size, n_classes))
    else:
        layers.append(nn.Linear(prev_size, 1))

    model = nn.Sequential(*layers).to(device)

    # Loss and optimizer
    lr = hyperparams.get("learning_rate", 0.001)
    weight_decay = hyperparams.get("weight_decay", 1e-4)
    if is_classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    epochs = hyperparams.get("epochs", 100)
    early_stop_patience = hyperparams.get("early_stop_patience", 15)
    best_val_loss = float("inf")
    patience_counter = 0
    training_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * len(xb)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"[{job_id}] Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Evaluate on test set
    with torch.no_grad():
        X_test_d = X_test_t.to(device)
        outputs = model(X_test_d)
        if is_classification:
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.cpu().numpy()
        else:
            y_pred = outputs.cpu().numpy().flatten()

    y_true = y_test.values
    if is_classification:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    else:
        metrics = {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2_score": float(r2_score(y_true, y_pred)),
        }

    # GPU memory tracking
    gpu_mem_used = 0.0
    if "cuda" in device_str:
        gpu_mem_used = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats(device)

    # Save model artifact
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pytorch_gpu_{job_id[:8]}_{timestamp}.pt"
    artifact_path = MODEL_DIR / filename
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": {
            "n_input": n_input,
            "hidden_layers": hidden_layers,
            "dropout": dropout,
            "n_classes": n_classes,
            "is_classification": is_classification,
        },
        "preprocessor": preprocessor,
        "hyperparameters": hyperparams,
        "metrics": metrics,
    }, str(artifact_path))

    artifact_size = os.path.getsize(artifact_path)
    with open(artifact_path, "rb") as f:
        artifact_hash = hashlib.sha256(f.read()).hexdigest()

    duration = time.time() - start_time

    return TrainingResult(
        job_id=job_id,
        algorithm="pytorch_neural_network",
        framework="pytorch",
        device_used=device_str,
        metrics=metrics,
        training_history=training_history,
        model_artifact_path=str(artifact_path),
        model_artifact_size=artifact_size,
        model_artifact_hash=artifact_hash,
        hyperparameters=hyperparams,
        duration_seconds=round(duration, 2),
        n_samples=len(X_train) + len(X_val) + len(X_test),
        n_features=n_input,
        n_train=len(X_train),
        n_test=len(X_test),
        gpu_memory_used_mb=round(gpu_mem_used, 2),
    )


# ─── XGBoost GPU Training ────────────────────────────────────────────────────


def _train_xgboost_gpu(
    X_train, X_val, X_test, y_train, y_val, y_test,
    numeric_features: List[str],
    categorical_features: List[str],
    is_classification: bool,
    hyperparams: Dict[str, Any],
    use_gpu: bool,
    job_id: str,
) -> TrainingResult:
    """Train XGBoost with GPU acceleration (tree_method='gpu_hist')."""
    from xgboost import XGBClassifier, XGBRegressor
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
    )

    start_time = time.time()
    device_used = "gpu" if use_gpu else "cpu"
    logger.info(f"[{job_id}] XGBoost training on: {device_used}")

    # Preprocessing pipeline
    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_train_np = preprocessor.fit_transform(X_train)
    X_val_np = preprocessor.transform(X_val)
    X_test_np = preprocessor.transform(X_test)

    # XGBoost parameters
    xgb_params = {
        "n_estimators": hyperparams.get("n_estimators", 500),
        "max_depth": hyperparams.get("max_depth", 8),
        "learning_rate": hyperparams.get("learning_rate", 0.05),
        "subsample": hyperparams.get("subsample", 0.8),
        "colsample_bytree": hyperparams.get("colsample_bytree", 0.8),
        "reg_alpha": hyperparams.get("reg_alpha", 0.1),
        "reg_lambda": hyperparams.get("reg_lambda", 1.0),
        "random_state": hyperparams.get("random_state", 42),
        "n_jobs": -1,
        "early_stopping_rounds": hyperparams.get("early_stopping_rounds", 20),
    }

    # GPU acceleration
    if use_gpu:
        xgb_params["tree_method"] = "gpu_hist"
        xgb_params["gpu_id"] = 0
    else:
        xgb_params["tree_method"] = "hist"

    if is_classification:
        model = XGBClassifier(**xgb_params)
    else:
        model = XGBRegressor(**xgb_params)

    # Train with eval set for early stopping
    model.fit(
        X_train_np, y_train,
        eval_set=[(X_val_np, y_val)],
        verbose=False,
    )

    # Predictions
    y_pred = model.predict(X_test_np)
    y_true = y_test.values

    if is_classification:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else 0,
        }
    else:
        metrics = {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2_score": float(r2_score(y_true, y_pred)),
            "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else 0,
        }

    # Feature importance
    importances = model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    feature_importance = sorted(
        [{"feature": str(n), "importance": float(v)} for n, v in zip(feature_names, importances)],
        key=lambda x: x["importance"], reverse=True
    )[:20]

    # Save model artifact
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"xgboost_gpu_{job_id[:8]}_{timestamp}.pkl"
    artifact_path = MODEL_DIR / filename

    with open(artifact_path, "wb") as f:
        pickle.dump({"model": model, "preprocessor": preprocessor}, f)

    artifact_size = os.path.getsize(artifact_path)
    with open(artifact_path, "rb") as f:
        artifact_hash = hashlib.sha256(f.read()).hexdigest()

    duration = time.time() - start_time

    return TrainingResult(
        job_id=job_id,
        algorithm="xgboost_gpu" if use_gpu else "xgboost",
        framework="xgboost",
        device_used=device_used,
        metrics=metrics,
        feature_importance=feature_importance,
        model_artifact_path=str(artifact_path),
        model_artifact_size=artifact_size,
        model_artifact_hash=artifact_hash,
        hyperparameters=xgb_params,
        duration_seconds=round(duration, 2),
        n_samples=len(X_train) + len(X_val) + len(X_test),
        n_features=X_train_np.shape[1],
        n_train=len(X_train),
        n_test=len(X_test),
    )


# ─── TensorFlow/Keras GPU Training ──────────────────────────────────────────


def _train_tensorflow_gpu(
    X_train, X_val, X_test, y_train, y_val, y_test,
    numeric_features: List[str],
    categorical_features: List[str],
    is_classification: bool,
    hyperparams: Dict[str, Any],
    job_id: str,
) -> TrainingResult:
    """Train a TensorFlow/Keras model with GPU memory growth."""
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
    )

    start_time = time.time()

    # Enable memory growth to avoid grabbing all GPU memory
    gpus = tf.config.list_physical_devices("GPU")
    device_used = "cpu"
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        device_used = f"gpu:{gpus[0].name}"
        logger.info(f"[{job_id}] TensorFlow using GPU: {gpus[0].name}")
    else:
        logger.info(f"[{job_id}] TensorFlow using CPU (no GPU found)")

    # Preprocess
    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_train_np = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_np = preprocessor.transform(X_val).astype(np.float32)
    X_test_np = preprocessor.transform(X_test).astype(np.float32)

    n_input = X_train_np.shape[1]
    n_classes = len(np.unique(y_train)) if is_classification else 1

    # Build Keras model
    hidden_layers = hyperparams.get("hidden_layers", [128, 64, 32])
    dropout = hyperparams.get("dropout", 0.3)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(n_input,)))
    for h in hidden_layers:
        model.add(tf.keras.layers.Dense(h, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(dropout))

    if is_classification:
        if n_classes == 2:
            model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
            metrics_list = ["accuracy"]
        else:
            model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"
            metrics_list = ["accuracy"]
    else:
        model.add(tf.keras.layers.Dense(1))
        loss = "mse"
        metrics_list = ["mae"]

    lr = hyperparams.get("learning_rate", 0.001)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=metrics_list)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    ]

    epochs = hyperparams.get("epochs", 100)
    batch_size = hyperparams.get("batch_size", 64)

    history = model.fit(
        X_train_np, y_train.values,
        validation_data=(X_val_np, y_val.values),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    # Training history
    training_history = []
    for i in range(len(history.history.get("loss", []))):
        entry = {"epoch": i + 1, "train_loss": float(history.history["loss"][i])}
        if "val_loss" in history.history:
            entry["val_loss"] = float(history.history["val_loss"][i])
        if "accuracy" in history.history:
            entry["accuracy"] = float(history.history["accuracy"][i])
        if "val_accuracy" in history.history:
            entry["val_accuracy"] = float(history.history["val_accuracy"][i])
        training_history.append(entry)

    # Evaluate
    y_pred_raw = model.predict(X_test_np, verbose=0)
    if is_classification:
        if n_classes == 2:
            y_pred = (y_pred_raw.flatten() > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_pred_raw, axis=1)
        y_true = y_test.values
        eval_metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    else:
        y_pred = y_pred_raw.flatten()
        y_true = y_test.values
        eval_metrics = {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2_score": float(r2_score(y_true, y_pred)),
        }

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tensorflow_gpu_{job_id[:8]}_{timestamp}.h5"
    artifact_path = MODEL_DIR / filename
    model.save(str(artifact_path))

    artifact_size = os.path.getsize(artifact_path)
    with open(artifact_path, "rb") as f:
        artifact_hash = hashlib.sha256(f.read()).hexdigest()

    duration = time.time() - start_time

    return TrainingResult(
        job_id=job_id,
        algorithm="tensorflow_neural_network",
        framework="tensorflow",
        device_used=device_used,
        metrics=eval_metrics,
        training_history=training_history,
        model_artifact_path=str(artifact_path),
        model_artifact_size=artifact_size,
        model_artifact_hash=artifact_hash,
        hyperparameters=hyperparams,
        duration_seconds=round(duration, 2),
        n_samples=len(X_train) + len(X_val) + len(X_test),
        n_features=n_input,
        n_train=len(X_train),
        n_test=len(X_test),
    )


# ─── Sklearn Training (CPU, no GPU needed) ───────────────────────────────────


def _train_sklearn(
    X_train, X_val, X_test, y_train, y_val, y_test,
    numeric_features: List[str],
    categorical_features: List[str],
    is_classification: bool,
    algorithm: str,
    hyperparams: Dict[str, Any],
    job_id: str,
) -> TrainingResult:
    """Train a scikit-learn model (CPU-based)."""
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
    )
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        ExtraTreesClassifier, ExtraTreesRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.svm import SVC, SVR

    start_time = time.time()
    random_state = hyperparams.get("random_state", 42)

    # Preprocessing
    transformers = []
    if numeric_features:
        transformers.append(("num", SkPipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), numeric_features))
    if categorical_features:
        transformers.append(("cat", SkPipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Select estimator
    estimator_map_clf = {
        "random_forest": RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 300),
            max_depth=hyperparams.get("max_depth"),
            random_state=random_state, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=hyperparams.get("n_estimators", 200),
            learning_rate=hyperparams.get("learning_rate", 0.05),
            max_depth=hyperparams.get("max_depth", 3),
            random_state=random_state
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=hyperparams.get("n_estimators", 300),
            random_state=random_state, n_jobs=-1
        ),
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=random_state
        ),
        "svm": SVC(
            kernel="rbf", probability=True, random_state=random_state
        ),
    }
    estimator_map_reg = {
        "random_forest": RandomForestRegressor(
            n_estimators=hyperparams.get("n_estimators", 300),
            max_depth=hyperparams.get("max_depth"),
            random_state=random_state, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=hyperparams.get("n_estimators", 200),
            learning_rate=hyperparams.get("learning_rate", 0.05),
            max_depth=hyperparams.get("max_depth", 3),
            random_state=random_state
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=hyperparams.get("n_estimators", 300),
            random_state=random_state, n_jobs=-1
        ),
        "ridge": Ridge(alpha=hyperparams.get("alpha", 1.0)),
        "svm": SVR(kernel="rbf"),
    }

    if is_classification:
        est = estimator_map_clf.get(algorithm)
    else:
        est = estimator_map_reg.get(algorithm)

    if est is None:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    pipe = SkPipeline([("prep", preprocessor), ("clf", est)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_true = y_test.values

    if is_classification:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    else:
        metrics = {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2_score": float(r2_score(y_true, y_pred)),
        }

    # Feature importance
    feature_importance = []
    try:
        feature_names = preprocessor.get_feature_names_out()
        if hasattr(est, "feature_importances_"):
            importances = est.feature_importances_
            feature_importance = sorted(
                [{"feature": str(n), "importance": float(v)} for n, v in zip(feature_names, importances)],
                key=lambda x: x["importance"], reverse=True
            )[:20]
    except Exception:
        pass

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{algorithm}_{job_id[:8]}_{timestamp}.pkl"
    artifact_path = MODEL_DIR / filename

    with open(artifact_path, "wb") as f:
        pickle.dump(pipe, f)

    artifact_size = os.path.getsize(artifact_path)
    with open(artifact_path, "rb") as f:
        artifact_hash = hashlib.sha256(f.read()).hexdigest()

    duration = time.time() - start_time

    return TrainingResult(
        job_id=job_id,
        algorithm=algorithm,
        framework="scikit-learn",
        device_used="cpu",
        metrics=metrics,
        feature_importance=feature_importance,
        model_artifact_path=str(artifact_path),
        model_artifact_size=artifact_size,
        model_artifact_hash=artifact_hash,
        hyperparameters=hyperparams,
        duration_seconds=round(duration, 2),
        n_samples=len(X_train) + len(X_val) + len(X_test),
        n_features=X_train.shape[1],
        n_train=len(X_train),
        n_test=len(X_test),
    )


# ─── Main Training Dispatcher ────────────────────────────────────────────────


# Supported algorithms grouped by framework
SUPPORTED_ALGORITHMS = {
    "pytorch": ["pytorch_neural_network", "pytorch_deep"],
    "tensorflow": ["tensorflow_neural_network", "tensorflow_cnn"],
    "xgboost": ["xgboost", "xgboost_gpu"],
    "sklearn": ["random_forest", "gradient_boosting", "extra_trees",
                "logistic_regression", "svm", "ridge"],
}

ALL_ALGORITHMS = []
for algos in SUPPORTED_ALGORITHMS.values():
    ALL_ALGORITHMS.extend(algos)


def run_offline_training(
    job_id: str,
    dataset: List[Dict[str, Any]],
    target_variable: str,
    features: List[str],
    algorithm: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    force_gpu: bool = False,
    force_cpu: bool = False,
) -> TrainingResult:
    """
    Main entry point: run offline model training with automatic GPU detection.

    Args:
        job_id: Unique job identifier
        dataset: Training data as list of records
        target_variable: Column to predict
        features: Feature columns
        algorithm: Algorithm name (from SUPPORTED_ALGORITHMS)
        hyperparameters: Optional model hyperparameters
        test_size: Test split ratio
        random_state: Random seed
        force_gpu: Require GPU (fail if not available)
        force_cpu: Force CPU even if GPU available

    Returns:
        TrainingResult with metrics, artifact path, and metadata
    """
    hyperparams = hyperparameters or {}
    hyperparams["random_state"] = random_state
    gpu_mgr = get_gpu_manager()

    logger.info(f"[{job_id}] Starting offline training: algorithm={algorithm}, "
                f"samples={len(dataset)}, features={len(features)}, "
                f"gpu_available={gpu_mgr.has_gpu}")

    if force_gpu and not gpu_mgr.has_gpu:
        return TrainingResult(
            job_id=job_id, algorithm=algorithm, framework="none",
            device_used="none", metrics={},
            error="GPU training requested but no GPU available on this server",
        )

    # Preprocess data
    try:
        (X_train, X_val, X_test, y_train, y_val, y_test,
         numeric_features, categorical_features, is_classification) = _preprocess_data(
            dataset, target_variable, features, test_size, random_state
        )
    except Exception as e:
        return TrainingResult(
            job_id=job_id, algorithm=algorithm, framework="none",
            device_used="none", metrics={},
            error=f"Data preprocessing failed: {str(e)}",
        )

    # Allocate GPU
    gpu_index = None
    if not force_cpu and gpu_mgr.has_gpu:
        gpu_index = gpu_mgr.allocate_gpu(job_id, estimated_memory_mb=2048)

    try:
        # Route to appropriate trainer
        if algorithm in SUPPORTED_ALGORITHMS["pytorch"]:
            device_str = f"cuda:{gpu_index}" if gpu_index is not None else "cpu"
            if force_cpu:
                device_str = "cpu"
            result = _train_pytorch(
                X_train, X_val, X_test, y_train, y_val, y_test,
                numeric_features, categorical_features, is_classification,
                hyperparams, device_str, job_id,
            )

        elif algorithm in SUPPORTED_ALGORITHMS["xgboost"]:
            use_gpu = gpu_index is not None and not force_cpu
            result = _train_xgboost_gpu(
                X_train, X_val, X_test, y_train, y_val, y_test,
                numeric_features, categorical_features, is_classification,
                hyperparams, use_gpu, job_id,
            )

        elif algorithm in SUPPORTED_ALGORITHMS["tensorflow"]:
            result = _train_tensorflow_gpu(
                X_train, X_val, X_test, y_train, y_val, y_test,
                numeric_features, categorical_features, is_classification,
                hyperparams, job_id,
            )

        elif algorithm in SUPPORTED_ALGORITHMS["sklearn"]:
            result = _train_sklearn(
                X_train, X_val, X_test, y_train, y_val, y_test,
                numeric_features, categorical_features, is_classification,
                algorithm, hyperparams, job_id,
            )

        else:
            result = TrainingResult(
                job_id=job_id, algorithm=algorithm, framework="none",
                device_used="none", metrics={},
                error=f"Unknown algorithm: {algorithm}. Supported: {ALL_ALGORITHMS}",
            )

    except Exception as e:
        logger.error(f"[{job_id}] Training failed: {e}", exc_info=True)
        result = TrainingResult(
            job_id=job_id, algorithm=algorithm, framework="none",
            device_used="gpu" if gpu_index is not None else "cpu",
            metrics={},
            error=f"Training failed: {str(e)}\n{traceback.format_exc()[:500]}",
        )
    finally:
        # Release GPU allocation
        if gpu_index is not None:
            gpu_mgr.release_gpu(job_id)

    return result
