"""
AutoML service for the cancer registry.

Pulls anonymized registry data directly from the database (server-side), trains a
zoo of scikit-learn models, optionally tunes the best one with randomized search,
and returns a ranked leaderboard with explanations (permutation importance),
ROC/PR data, and an optional saved model artifact.

Design inspired by open-source AutoML tools (FLAML, mljar-supervised, PyCaret):
leaderboard + tuning + explanations + persistence. Pure scikit-learn so it runs
in the slim backend image (no xgboost/lightgbm required).
"""
from __future__ import annotations

import os
import time
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "age_at_diagnosis", "gender", "nationality", "icd11_main_code",
    "t_category", "n_category", "m_category", "laterality",
    "basis_of_diagnosis", "treatment_intent",
    "surgery_done", "chemotherapy_done", "radiotherapy_done",
    "hormonal_therapy", "immunotherapy", "survival_months",
]

ALLOWED_TARGETS = [
    "vital_status", "recurrence", "metastasis",
    "treatment_intent", "icd11_main_code", "survival_months",
]

MAX_ROWS = 20000
DEFAULT_ROWS = 6000
MODEL_DIR = os.environ.get("AUTOML_MODEL_DIR", "trained_models")


def _build_estimators(is_classification: bool, random_state: int) -> Dict[str, Any]:
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        ExtraTreesClassifier, ExtraTreesRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        AdaBoostClassifier, AdaBoostRegressor,
        HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    if is_classification:
        return {
            "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
            "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1),
            "extra_trees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1),
            "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=300, random_state=random_state),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=200, random_state=random_state),
            "adaboost": AdaBoostClassifier(n_estimators=200, random_state=random_state),
            "decision_tree": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=random_state),
            "knn": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        }
    return {
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=300, random_state=random_state),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=200, random_state=random_state),
        "adaboost": AdaBoostRegressor(n_estimators=200, random_state=random_state),
        "decision_tree": DecisionTreeRegressor(max_depth=12, random_state=random_state),
        "knn": KNeighborsRegressor(n_neighbors=15, weights="distance"),
    }


def _param_distributions(name: str) -> Dict[str, list]:
    """Small randomized-search spaces keyed by the pipeline step 'clf__<param>'."""
    spaces = {
        "random_forest": {"clf__n_estimators": [200, 400, 600], "clf__max_depth": [None, 10, 20, 40], "clf__min_samples_leaf": [1, 2, 4]},
        "extra_trees": {"clf__n_estimators": [200, 400, 600], "clf__max_depth": [None, 10, 20, 40], "clf__min_samples_leaf": [1, 2, 4]},
        "hist_gradient_boosting": {"clf__max_iter": [200, 400, 600], "clf__learning_rate": [0.03, 0.05, 0.1], "clf__max_leaf_nodes": [15, 31, 63]},
        "gradient_boosting": {"clf__n_estimators": [150, 300, 450], "clf__learning_rate": [0.03, 0.05, 0.1], "clf__max_depth": [2, 3, 4]},
        "adaboost": {"clf__n_estimators": [100, 200, 400], "clf__learning_rate": [0.05, 0.1, 0.5]},
        "decision_tree": {"clf__max_depth": [6, 10, 16, None], "clf__min_samples_leaf": [1, 2, 5, 10]},
        "knn": {"clf__n_neighbors": [5, 11, 15, 25], "clf__weights": ["uniform", "distance"]},
        "logistic_regression": {"clf__C": [0.1, 0.5, 1.0, 2.0]},
        "ridge": {"clf__alpha": [0.1, 1.0, 5.0, 10.0]},
    }
    return spaces.get(name, {})


def _load_dataframe(db: Session, columns: List[str], limit: int) -> pd.DataFrame:
    cols_sql = ", ".join(f'"{c}"' for c in columns)
    sql = text(
        f"SELECT {cols_sql} FROM registry.patients "
        f"WHERE is_active = true ORDER BY entry_timestamp DESC NULLS LAST LIMIT :lim"
    )
    rows = db.execute(sql, {"lim": limit}).fetchall()
    return pd.DataFrame(rows, columns=columns)


def run_automl(
    db: Session,
    target_variable: str,
    features: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    row_limit: int = DEFAULT_ROWS,
    cross_validate: bool = True,
    tune: bool = True,
    save_model: bool = False,
) -> Dict[str, Any]:
    from sklearn.model_selection import (
        train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold, KFold,
    )
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,
    )

    started = time.time()
    if target_variable not in ALLOWED_TARGETS:
        raise ValueError(f"target_variable must be one of {ALLOWED_TARGETS}")
    row_limit = max(200, min(int(row_limit), MAX_ROWS))

    feat = ([f for f in features if f in FEATURE_COLUMNS and f != target_variable]
            if features else [f for f in FEATURE_COLUMNS if f != target_variable])
    if not feat:
        raise ValueError("No valid features selected for AutoML")

    needed = list(dict.fromkeys(feat + [target_variable]))
    df = _load_dataframe(db, needed, row_limit)
    if df.empty:
        raise ValueError("No active patient rows available to train on")

    df = df.replace(["", "None", "null", None], np.nan).dropna(subset=[target_variable])
    if df.empty or df[target_variable].nunique() < 2:
        raise ValueError(f"Target '{target_variable}' has fewer than 2 distinct values after cleaning")

    y_raw = df[target_variable]
    X = df[feat].copy()
    n_unique = y_raw.nunique()
    is_numeric_target = pd.api.types.is_numeric_dtype(y_raw) or pd.api.types.is_bool_dtype(y_raw)
    is_classification = (not is_numeric_target) or (n_unique <= 12)

    class_labels: Optional[List[str]] = None
    if is_classification:
        y_cat = pd.Categorical(y_raw.astype(str))
        y = np.asarray(y_cat.codes)
        class_labels = [str(c) for c in y_cat.categories.tolist()]
    else:
        y = pd.to_numeric(y_raw, errors="coerce").astype(float).to_numpy()
        mask = ~np.isnan(y)
        X, y = X.loc[mask], y[mask]

    numeric_features = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    for c in categorical_features:
        X[c] = X[c].astype("object").where(X[c].notna(), "__MISSING__").astype(str)

    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    binary = is_classification and len(np.unique(y)) == 2
    stratify = y if (is_classification and pd.Series(y).value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    estimators = _build_estimators(is_classification, random_state)
    primary_metric = "f1_weighted" if is_classification else "r2"
    scoring = "f1_weighted" if is_classification else "r2"

    leaderboard: List[Dict[str, Any]] = []
    fitted: Dict[str, Any] = {}

    def _score_model(pipe, name) -> Dict[str, Any]:
        y_pred = pipe.predict(X_test)
        entry: Dict[str, Any] = {"algorithm": name}
        if is_classification:
            m = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            }
            try:
                if hasattr(pipe.named_steps["clf"], "predict_proba"):
                    proba = pipe.predict_proba(X_test)
                    if binary:
                        m["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
                    else:
                        m["roc_auc"] = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
            except Exception:
                pass
            entry["metrics"] = m
            entry["score"] = m["f1_weighted"]
        else:
            m = {
                "r2": float(r2_score(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            }
            entry["metrics"] = m
            entry["score"] = m["r2"]
        return entry

    for name, est in estimators.items():
        try:
            t0 = time.time()
            pipe = Pipeline([("prep", preprocessor), ("clf", est)])
            pipe.fit(X_train, y_train)
            entry = _score_model(pipe, name)
            entry["train_seconds"] = round(time.time() - t0, 2)
            if cross_validate and len(X_train) <= 8000:
                try:
                    cv = (StratifiedKFold(3, shuffle=True, random_state=random_state)
                          if is_classification else KFold(3, shuffle=True, random_state=random_state))
                    cvs = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                    entry["cv_mean"] = float(np.mean(cvs)); entry["cv_std"] = float(np.std(cvs))
                except Exception:
                    pass
            leaderboard.append(entry)
            fitted[name] = pipe
        except Exception as e:
            logger.warning(f"AutoML model '{name}' failed: {e}")
            leaderboard.append({"algorithm": name, "error": str(e), "score": float("-inf")})

    leaderboard.sort(key=lambda d: d.get("score", float("-inf")), reverse=True)
    if not leaderboard or leaderboard[0].get("score", float("-inf")) == float("-inf"):
        raise RuntimeError("All AutoML models failed to train")

    best_name = leaderboard[0]["algorithm"]
    best_pipe = fitted.get(best_name)
    tuned = False
    best_params: Dict[str, Any] = {}

    # Hyperparameter tuning on the winning model (randomized search)
    if tune and best_pipe is not None:
        space = _param_distributions(best_name)
        if space and len(X_train) <= 8000:
            try:
                cv = (StratifiedKFold(3, shuffle=True, random_state=random_state)
                      if is_classification else KFold(3, shuffle=True, random_state=random_state))
                search = RandomizedSearchCV(best_pipe, space, n_iter=8, cv=cv, scoring=scoring,
                                            random_state=random_state, n_jobs=-1, refit=True)
                search.fit(X_train, y_train)
                tuned_entry = _score_model(search.best_estimator_, best_name)
                if tuned_entry["score"] >= leaderboard[0].get("score", float("-inf")):
                    best_pipe = search.best_estimator_
                    fitted[best_name] = best_pipe
                    leaderboard[0]["metrics"] = tuned_entry["metrics"]
                    leaderboard[0]["score"] = tuned_entry["score"]
                    leaderboard[0]["tuned"] = True
                    tuned = True
                    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
            except Exception as e:
                logger.warning(f"AutoML tuning failed for {best_name}: {e}")

    # Permutation importance (model-agnostic explanation), on a capped sample
    feature_importance: List[Dict[str, Any]] = []
    try:
        sample_n = min(1000, len(X_test))
        Xs = X_test.iloc[:sample_n]
        ys = y_test[:sample_n]
        pi = permutation_importance(best_pipe, Xs, ys, n_repeats=5, random_state=random_state,
                                    scoring=scoring, n_jobs=-1)
        pairs = sorted(zip(feat, pi.importances_mean), key=lambda x: x[1], reverse=True)
        feature_importance = [{"feature": n, "importance": float(v)} for n, v in pairs if v > 0][:20]
        if not feature_importance:
            feature_importance = [{"feature": n, "importance": float(v)} for n, v in pairs][:20]
    except Exception as e:
        logger.warning(f"Permutation importance failed: {e}")

    confusion: Optional[List[List[int]]] = None
    roc_points: Optional[List[Dict[str, float]]] = None
    if is_classification and best_pipe is not None:
        try:
            confusion = confusion_matrix(y_test, best_pipe.predict(X_test)).tolist()
        except Exception:
            pass
        if binary:
            try:
                proba = best_pipe.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, proba)
                step = max(1, len(fpr) // 50)
                roc_points = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr[::step], tpr[::step])]
            except Exception:
                pass

    # Optional model persistence
    artifact: Optional[Dict[str, Any]] = None
    if save_model and best_pipe is not None:
        try:
            import joblib
            os.makedirs(MODEL_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_target = target_variable.replace("/", "_")
            fname = f"automl_{best_name}_{safe_target}_{ts}.joblib"
            fpath = os.path.join(MODEL_DIR, fname)
            joblib.dump(best_pipe, fpath)
            size = os.path.getsize(fpath)
            with open(fpath, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            artifact = {"filename": fname, "path": fpath, "size_bytes": size, "sha256": digest}
        except Exception as e:
            logger.warning(f"Model persistence failed: {e}")

    return {
        "task_type": "classification" if is_classification else "regression",
        "target_variable": target_variable,
        "primary_metric": primary_metric,
        "features_used": feat,
        "class_labels": class_labels,
        "tuned": tuned,
        "data_stats": {
            "rows_used": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": int(len(feat)),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
        "leaderboard": leaderboard,
        "best_model": {
            "algorithm": best_name,
            "score": leaderboard[0].get("score"),
            "metrics": leaderboard[0].get("metrics"),
            "tuned": tuned,
            "best_params": best_params,
            "confusion_matrix": confusion,
            "roc_curve": roc_points,
            "feature_importance": feature_importance,
        },
        "model_artifact": artifact,
        "total_seconds": round(time.time() - started, 2),
    }


def list_options() -> Dict[str, Any]:
    return {
        "targets": ALLOWED_TARGETS,
        "features": FEATURE_COLUMNS,
        "max_rows": MAX_ROWS,
        "default_rows": DEFAULT_ROWS,
        "models": [
            "logistic_regression/ridge", "random_forest", "extra_trees",
            "hist_gradient_boosting", "gradient_boosting", "adaboost",
            "decision_tree", "knn",
        ],
        "capabilities": [
            "model leaderboard", "hyperparameter tuning (randomized search)",
            "cross-validation", "permutation-importance explanations",
            "ROC curve (binary)", "class-imbalance handling", "model download",
        ],
        "note": "AutoML trains and tunes multiple scikit-learn models server-side on registry data.",
    }
