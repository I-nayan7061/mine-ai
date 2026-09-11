"""Unsupervised Anomaly Detection Module for Mine Strata Telemetry.
SIH26025 - NexGen | Isolation Forest & Statistical Baselines

Implements:
- Isolation Forest with configurable contamination
- Normalized continuous anomaly scoring [0.0 - 1.0]
- Statistical baseline detectors (Rolling Z-Score and Interquartile Range IQR)
- Model serialization & loading
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import app_config
from .utils import get_logger, load_artifact, save_artifact

logger = get_logger("mine_ai.anomaly")


class StatisticalBaselineDetector:
    """Baseline Z-Score and IQR anomaly detector for benchmarking."""

    def __init__(self, z_thresh: float = 3.0, iqr_multiplier: float = 1.5):
        self.z_thresh = z_thresh
        self.iqr_multiplier = iqr_multiplier
        self.means: Optional[pd.Series] = None
        self.stds: Optional[pd.Series] = None
        self.q1: Optional[pd.Series] = None
        self.q3: Optional[pd.Series] = None
        self.iqr: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame) -> "StatisticalBaselineDetector":
        self.means = X.mean()
        self.stds = X.std().replace(0.0, 1e-6)
        self.q1 = X.quantile(0.25)
        self.q3 = X.quantile(0.75)
        self.iqr = (self.q3 - self.q1).replace(0.0, 1e-6)
        return self

    def predict_zscore(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (is_anomaly, max_zscore)."""
        z_scores = ((X - self.means) / self.stds).abs()
        max_z = z_scores.max(axis=1).values
        is_anomaly = max_z > self.z_thresh
        return is_anomaly, max_z

    def predict_iqr(self, X: pd.DataFrame) -> np.ndarray:
        """Returns boolean mask of IQR outliers."""
        lower = self.q1 - self.iqr_multiplier * self.iqr
        upper = self.q3 + self.iqr_multiplier * self.iqr
        is_outlier = ((X < lower) | (X > upper)).any(axis=1).values
        return is_outlier


class MineAnomalyDetector:
    """Primary Unsupervised Anomaly Detector using Isolation Forest."""

    def __init__(self, config=None):
        self.config = config or app_config
        self.contamination = self.config.get("anomaly_detection.contamination", 0.08)
        self.n_estimators = self.config.get("anomaly_detection.n_estimators", 150)
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.baseline_detector: Optional[StatisticalBaselineDetector] = None

    def fit(self, X: pd.DataFrame) -> "MineAnomalyDetector":
        """Fit Isolation Forest and standard scaler on normal/baseline telemetry features."""
        self.feature_names = list(X.columns)
        logger.info("Fitting Isolation Forest on %d samples with %d features...", len(X), len(self.feature_names))

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled)

        # Fit baseline detector for comparison
        self.baseline_detector = StatisticalBaselineDetector()
        self.baseline_detector.fit(X)

        logger.info("Anomaly Detector successfully trained.")
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict binary anomaly flags and continuous normalized anomaly scores [0, 1].
        
        Returns:
            is_anomaly: boolean array (True if anomalous)
            anomaly_score: float array in [0.0, 1.0] (higher = more anomalous)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Anomaly detector model is not fitted or loaded.")

        # Ensure correct column ordering
        X_ordered = X[self.feature_names].copy()
        X_scaled = self.scaler.transform(X_ordered)

        # IsolationForest decision_function: lower values indicate abnormal points
        raw_scores = self.model.decision_function(X_scaled)

        # Convert raw decision score to [0, 1] anomaly probability:
        # Calibrated around decision boundary: raw_scores > 0.0 indicates normal inliers in sklearn
        # Shifts baseline so that healthy inliers produce near-zero anomaly scores
        anomaly_scores = np.clip(1.0 / (1.0 + np.exp((raw_scores - 0.02) * 10.0)), 0.0, 1.0)
        # Binary prediction: -1 is anomaly in sklearn
        raw_preds = self.model.predict(X_scaled)
        is_anomaly = raw_preds == -1

        return is_anomaly, anomaly_scores

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist trained model, scaler, and feature names."""
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "baseline": self.baseline_detector
        }
        save_artifact(data, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path], config=None) -> "MineAnomalyDetector":
        """Load persisted anomaly detector."""
        instance = cls(config=config)
        data = load_artifact(filepath)
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.feature_names = data["feature_names"]
        instance.contamination = data.get("contamination", 0.08)
        instance.baseline_detector = data.get("baseline")
        return instance
