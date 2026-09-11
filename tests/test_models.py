"""Unit tests for ML models and explainability.
SIH26025 - NexGen
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.anomaly_model import MineAnomalyDetector
from src.classifier import CLASS_NAMES, MineRiskClassifier
from src.config import app_config
from src.explainability import MineExplainer
from src.utils import load_json


def get_models_dir() -> Path:
    return app_config.resolve_path("models")


def test_models_persisted_and_loadable():
    models_dir = get_models_dir()
    assert (models_dir / "isolation_forest.pkl").exists()
    assert (models_dir / "random_forest.pkl").exists()
    assert (models_dir / "xgboost_model.pkl").exists()
    assert (models_dir / "feature_names.json").exists()

    anom = MineAnomalyDetector.load(models_dir / "isolation_forest.pkl")
    assert anom.model is not None

    xgb_clf = MineRiskClassifier.load(models_dir / "xgboost_model.pkl")
    assert xgb_clf.model is not None


def test_model_predictions():
    models_dir = get_models_dir()
    feat_meta = load_json(models_dir / "feature_names.json")
    feat_names = feat_meta["selected_features"]

    # Dummy test row with zero values
    row_df = pd.DataFrame([{f: 0.05 for f in feat_names}])

    # 1. Anomaly Model
    anom = MineAnomalyDetector.load(models_dir / "isolation_forest.pkl")
    is_anom, scores = anom.predict(row_df)
    assert len(is_anom) == 1
    assert 0.0 <= scores[0] <= 1.0

    # 2. XGBoost Model
    xgb_clf = MineRiskClassifier.load(models_dir / "xgboost_model.pkl")
    labels, probs = xgb_clf.predict(row_df)
    assert len(labels) == 1
    assert labels[0] in CLASS_NAMES
    assert probs.shape == (1, 4)
    assert np.isclose(np.sum(probs[0]), 1.0, atol=1e-3)


def test_explainability():
    models_dir = get_models_dir()
    feat_meta = load_json(models_dir / "feature_names.json")
    feat_names = feat_meta["selected_features"]
    row_df = pd.DataFrame([{f: 0.5 for f in feat_names}])

    rf_clf = MineRiskClassifier.load(models_dir / "random_forest.pkl")
    explainer = MineExplainer(rf_clf.model, feat_names)
    explanation = explainer.explain_prediction(row_df, top_k=3)

    assert "top_factors" in explanation
    assert len(explanation["top_factors"]) == 3
    assert len(explanation["human_explanations"]) == 3
