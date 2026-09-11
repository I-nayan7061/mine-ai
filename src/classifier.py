"""Supervised Strata Risk Classification and Model Comparison Module.
SIH26025 - NexGen | Mine Subsidence Risk Classifier

Supports:
- Chronological, leakage-free time-series train/val/test splitting
- Multi-model benchmarking: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM
- Priority focus on HIGH and CRITICAL safety recall
- Model evaluation reporting and artifact serialization
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from .config import app_config
from .utils import get_logger, load_artifact, save_artifact, save_json

logger = get_logger("mine_ai.classifier")

CLASS_NAMES = ["NORMAL", "WARNING", "HIGH", "CRITICAL"]
CLASS_TO_INT = {"NORMAL": 0, "WARNING": 1, "HIGH": 2, "CRITICAL": 3}
INT_TO_CLASS = {0: "NORMAL", 1: "WARNING", 2: "HIGH", 3: "CRITICAL"}


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict chronological time-series splitting to prevent forward-looking data leakage."""
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    logger.info("Chronological split: Train=%d, Val=%d, Test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


class MineRiskClassifier:
    """Trains, compares, and evaluates risk classifiers across candidate algorithms."""

    def __init__(self, model_type: str = "random_forest", config=None):
        self.config = config or app_config
        self.model_type = model_type
        self.model: Any = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.training_metadata: Dict[str, Any] = {}

    def _init_model(self, model_type: str) -> Any:
        rf_cfg = self.config.get("classification.hyperparameters.random_forest", {})
        xgb_cfg = self.config.get("classification.hyperparameters.xgboost", {})

        if model_type == "logistic_regression":
            return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        elif model_type == "decision_tree":
            return DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42)
        elif model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=rf_cfg.get("n_estimators", 120),
                max_depth=rf_cfg.get("max_depth", 12),
                min_samples_split=rf_cfg.get("min_samples_split", 4),
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=xgb_cfg.get("n_estimators", 120),
                max_depth=xgb_cfg.get("max_depth", 6),
                learning_rate=xgb_cfg.get("learning_rate", 0.08),
                subsample=xgb_cfg.get("subsample", 0.85),
                objective="multi:softprob",
                num_class=4,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.08,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "MineRiskClassifier":
        """Fit scaler and target classifier on training data."""
        self.feature_names = list(X_train.columns)
        self.model = self._init_model(self.model_type)

        y_encoded = y_train.map(CLASS_TO_INT).values
        X_scaled = self.scaler.fit_transform(X_train)

        start_t = time.perf_counter()
        self.model.fit(X_scaled, y_encoded)
        fit_duration = time.perf_counter() - start_t

        self.training_metadata = {
            "model_type": self.model_type,
            "train_samples": len(X_train),
            "num_features": len(self.feature_names),
            "fit_time_seconds": round(fit_duration, 4)
        }
        logger.info("Fitted %s in %.3f seconds.", self.model_type, fit_duration)
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class labels and per-class probabilities."""
        if self.model is None:
            raise ValueError("Model is not fitted or loaded.")

        X_ordered = X[self.feature_names].copy()
        X_scaled = self.scaler.transform(X_ordered)

        encoded_preds = self.model.predict(X_scaled)
        labels = np.array([INT_TO_CLASS.get(int(p), "NORMAL") for p in encoded_preds])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)
        else:
            probs = np.zeros((len(X), len(CLASS_NAMES)))
            for i, p in enumerate(encoded_preds):
                probs[i, int(p)] = 1.0

        return labels, probs

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Compute exhaustive evaluation metrics with special emphasis on High & Critical safety recall."""
        start_t = time.perf_counter()
        y_pred, probs = self.predict(X_test)
        inference_time_ms = ((time.perf_counter() - start_t) / max(1, len(X_test))) * 1000.0

        y_true = y_test.values
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        class_recalls = {}
        for c in CLASS_NAMES:
            c_mask = y_true == c
            if np.sum(c_mask) > 0:
                class_recalls[f"{c}_recall"] = float(recall_score(y_true == c, y_pred == c, zero_division=0))
            else:
                class_recalls[f"{c}_recall"] = 0.0

        cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)

        results = {
            "model_type": self.model_type,
            "accuracy": round(acc, 4),
            "weighted_precision": round(prec, 4),
            "weighted_recall": round(rec, 4),
            "weighted_f1": round(f1, 4),
            "high_recall": round(class_recalls.get("HIGH_recall", 0.0), 4),
            "critical_recall": round(class_recalls.get("CRITICAL_recall", 0.0), 4),
            "inference_time_ms_per_sample": round(inference_time_ms, 3),
            "per_class_recalls": class_recalls,
            "confusion_matrix": cm.tolist(),
            "class_names": CLASS_NAMES
        }
        return results

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist model and transformation pipeline."""
        data = {
            "model": self.model,
            "model_type": self.model_type,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "metadata": self.training_metadata
        }
        save_artifact(data, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path], config=None) -> "MineRiskClassifier":
        """Load persisted classifier artifact."""
        instance = cls(config=config)
        data = load_artifact(filepath)
        instance.model = data["model"]
        instance.model_type = data["model_type"]
        instance.scaler = data["scaler"]
        instance.feature_names = data["feature_names"]
        instance.training_metadata = data.get("metadata", {})
        return instance
