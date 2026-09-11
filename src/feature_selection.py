"""Feature Selection and Optimization Pipeline.
SIH26025 - NexGen | Mine Subsidence Feature Engineering

Filters collinear and redundant features using:
1. Low-variance thresholding
2. High-correlation collinearity pruning (r > 0.95)
3. Mutual Information and Tree-based Feature Importance
4. Persisting optimal feature subset for reproducible inference
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from .config import app_config
from .utils import get_logger, save_json

logger = get_logger("mine_ai.feature_selection")


class FeatureSelector:
    """Selects top informative features for mine subsidence prediction."""

    def __init__(self, config=None):
        self.config = config or app_config
        self.max_features = self.config.get("features.max_features_selected", 30)
        self.selected_features: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Identify and rank the top features based on variance, collinearity, and importance."""
        logger.info("Starting feature selection on %d candidate features...", X.shape[1])

        # 1. Drop constant or near-constant features (variance < 1e-6)
        variances = X.var()
        non_constant = variances[variances > 1e-6].index.tolist()
        X_filtered = X[non_constant].copy()

        # 2. Prune highly collinear features (|r| > 0.95) with target awareness
        corr_matrix = X_filtered.corr().abs()
        # Encode string labels if necessary
        if y.dtype == object:
            y_encoded = pd.Categorical(y).codes
        else:
            y_encoded = y

        # Precompute simple target association to keep the more informative feature in each collinear pair
        target_relevance = X_filtered.apply(lambda col: abs(float(np.corrcoef(col, y_encoded)[0, 1])) if col.std() > 1e-6 else 0.0)
        target_relevance = target_relevance.fillna(0.0)

        to_drop = set()
        cols = list(X_filtered.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                if c1 not in to_drop and c2 not in to_drop:
                    if corr_matrix.loc[c1, c2] > 0.95:
                        # Drop the feature with lower relevance to target
                        if target_relevance.get(c1, 0.0) >= target_relevance.get(c2, 0.0):
                            to_drop.add(c2)
                        else:
                            to_drop.add(c1)

        X_filtered = X_filtered.drop(columns=list(to_drop))
        logger.info("Pruned %d collinear features; %d features remaining.", len(to_drop), X_filtered.shape[1])

        # 3. Supervised Importance via Random Forest & Mutual Information
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        # Encode string labels if necessary
        if y.dtype == object:
            y_encoded = pd.Categorical(y).codes
        else:
            y_encoded = y

        rf.fit(X_filtered, y_encoded)
        importances = pd.Series(rf.feature_importances_, index=X_filtered.columns)

        # Compute mutual information
        try:
            mi = mutual_info_classif(X_filtered, y_encoded, random_state=42)
            mi_series = pd.Series(mi, index=X_filtered.columns)
            # Combined rank score (normalized)
            norm_imp = importances / (importances.max() + 1e-8)
            norm_mi = mi_series / (mi_series.max() + 1e-8)
            composite_score = 0.6 * norm_imp + 0.4 * norm_mi
        except Exception:
            composite_score = importances

        ranked = composite_score.sort_values(ascending=False)
        self.selected_features = ranked.head(self.max_features).index.tolist()

        logger.info("Selected top %d features: %s", len(self.selected_features), self.selected_features[:5])
        return self.selected_features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter input DataFrame to only selected features."""
        if not self.selected_features:
            raise ValueError("FeatureSelector has not been fitted yet.")
        # Ensure any missing selected features are populated with 0
        missing = [f for f in self.selected_features if f not in X.columns]
        X_copy = X.copy()
        for m in missing:
            X_copy[m] = 0.0
        return X_copy[self.selected_features]

    def save(self, filepath: str | Path) -> None:
        """Save selected feature names to JSON."""
        save_json({"selected_features": self.selected_features, "count": len(self.selected_features)}, filepath)
