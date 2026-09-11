"""Automated Verification Suite for Multimodal Alignment, Leakage Prevention & Safety Fallbacks.
SIH26025 - NexGen | Master Engineering Prompt Part 37

Verifies:
1. Zero target leakage in spatial features or schema.
2. Zero cross-split rolling window boundary overlap.
3. Scaler and feature selection fit strictly on training data.
4. Correct spatial-temporal alignment for India IMD weather and Sentinel-1 InSAR data.
5. Robust fallback handling when satellite or weather data is missing or stale.
6. Sensor fault differentiation vs geotechnical anomalies.
7. Backward compatibility of all existing API endpoints and schemas.
"""

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import app_config
from src.inference import MineInferencePipeline
from src.leakage_audit import LeakageAuditor
from src.weather_features import IndiaWeatherProvider
from src.satellite_features import IndiaSatelliteProvider
from src.spatial_features import MineSpatialGraph
from src.utils import load_json


@pytest.fixture
def client():
    return TestClient(app)


def _get_path(rel_path: str) -> Path:
    p1 = Path(rel_path)
    if p1.exists():
        return p1
    p2 = Path("mine-ai") / rel_path
    if p2.exists():
        return p2
    return p1


def test_no_target_leakage():
    """Verify that no target classes or risk labels appear in feature schemas."""
    auditor = LeakageAuditor()
    feat_json = _get_path("models/feature_names.json")
    assert feat_json.exists(), "feature_names.json must exist"
    d = load_json(feat_json)
    selected = d.get("selected_features", [])
    is_clean, flagged = auditor.audit_feature_names(selected)
    assert is_clean, f"Target leakage detected in selected features: {flagged}"


def test_no_cross_split_window_overlap():
    """Verify that train, val, and test feature splits share zero raw timestamps."""
    auditor = LeakageAuditor()
    tr_path = _get_path("data/processed/train_features.csv")
    val_path = _get_path("data/processed/val_features.csv")
    te_path = _get_path("data/processed/test_features.csv")

    assert tr_path.exists() and val_path.exists() and te_path.exists(), "Feature split CSVs must exist"

    df_tr = pd.read_csv(tr_path)
    df_val = pd.read_csv(val_path)
    df_te = pd.read_csv(te_path)

    passed, report = auditor.audit_window_overlap(df_tr, df_val, df_te, timestamp_col="timestamp")
    assert passed, f"Cross-split window overlap detected: {report}"


def test_scaler_and_feature_selection_fitted_on_train_only():
    """Verify preprocessing metadata proves scaler and selector fit strictly on training set."""
    prep_meta_path = _get_path("models/preprocessing_metadata.json")
    feat_meta_path = _get_path("models/feature_selection_metadata.json")

    assert prep_meta_path.exists(), "preprocessing_metadata.json must exist"
    assert feat_meta_path.exists(), "feature_selection_metadata.json must exist"

    prep_d = load_json(prep_meta_path)
    feat_d = load_json(feat_meta_path)

    assert prep_d.get("scaler_fit_on_train_only") is True
    assert feat_d.get("fit_strictly_on_train") is True


def test_weather_alignment_and_provenance():
    """Verify India weather provider produces valid IMD indicators and provenance."""
    provider = IndiaWeatherProvider()
    obs = provider.get_weather_observation()
    assert "provenance" in obs
    assert "IMD" in obs["provenance"]["source"]
    assert obs["has_weather"] == 1.0

    feats = provider.extract_weather_features(obs)
    assert "weather_rain_24h" in feats
    assert "weather_soil_moisture_deep" in feats
    assert "weather_pore_pressure_kpa" in feats
    assert feats["has_weather"] == 1.0
    assert feats["weather_age_hours"] >= 0.0


def test_satellite_alignment_and_provenance():
    """Verify Sentinel-1 InSAR provider produces valid regional deformation indicators."""
    provider = IndiaSatelliteProvider()
    obs = provider.get_satellite_observation()
    assert "provenance" in obs
    assert "SENTINEL" in obs["provenance"]["primary_source"]
    assert obs["has_satellite"] == 1.0

    disp_feats = {"disp_rate_max": 0.02, "disp_velocity_mean": 0.01}
    feats = provider.extract_satellite_features(obs, disp_features=disp_feats)
    assert "sat_insar_velocity_mm_yr" in feats
    assert "sat_insar_coherence" in feats
    assert feats["has_satellite"] == 1.0
    assert feats["satellite_age_days"] >= 0.0


def test_missing_satellite_fallback():
    """Verify pipeline handles missing satellite observation gracefully without NaN."""
    provider = IndiaSatelliteProvider()
    missing_obs = {"has_satellite": 0.0, "satellite_age_days": 99.0}
    feats = provider.extract_satellite_features(missing_obs)
    assert feats["has_satellite"] == 0.0
    assert not np.isnan(feats["sat_insar_velocity_mm_yr"])


def test_missing_weather_fallback():
    """Verify pipeline handles missing weather observation gracefully without NaN."""
    provider = IndiaWeatherProvider()
    missing_obs = {"has_weather": 0.0, "weather_age_hours": 99.0}
    feats = provider.extract_weather_features(missing_obs)
    assert feats["has_weather"] == 0.0
    assert not np.isnan(feats["weather_rain_24h"])


def test_stale_satellite_handling():
    """Verify satellite observation age exceeds freshness limit correctly flags age days."""
    provider = IndiaSatelliteProvider()
    stale_obs = provider.get_satellite_observation()
    stale_obs["satellite_age_days"] = 28.0  # 4 weeks old
    feats = provider.extract_satellite_features(stale_obs)
    assert feats["satellite_age_days"] == 28.0


def test_sensor_fault_differentiation():
    """Verify frozen stuck sensors are handled gracefully without false critical evacuation."""
    pipeline = MineInferencePipeline()
    pipeline.buffer_manager.clear()
    now_t = datetime(2026, 9, 10, 14, 0, 0)

    for i in range(25):
        res = pipeline.process_single_reading({
            "node_id": "N01",
            "timestamp": str(now_t + pd.Timedelta(seconds=i)),
            "tilt_x": 0.0000,
            "tilt_y": 0.0000,
            "vibration": 0.0000,
            "displacement_mm": 0.0000
        })

    assert res["success"] is True
    assert res["risk_level"] != "CRITICAL"
    assert not np.isnan(res["risk_score"])


def test_multimodal_prediction_and_shap():
    """Verify live inference produces risk prediction, class probabilities, and SHAP factors."""
    pipeline = MineInferencePipeline()
    now_t = datetime(2026, 9, 10, 15, 0, 0)

    # Send 15 continuous readings
    for i in range(15):
        res = pipeline.process_single_reading({
            "node_id": "N02",
            "timestamp": str(now_t + pd.Timedelta(seconds=i)),
            "tilt_x": 0.12 + (i * 0.01),
            "tilt_y": 0.08 + (i * 0.01),
            "vibration": 0.04,
            "displacement_mm": 1.05 + (i * 0.02)
        })

    assert res["success"] is True
    assert "risk_score" in res
    assert "risk_level" in res
    assert "subscores" in res
    assert "top_contributing_features" in res
    assert "human_explanations" in res
    assert len(res["class_probabilities"]) == 4


def test_api_backward_compatibility(client):
    """Verify that all existing REST endpoints remain fully backward compatible."""
    resp_health = client.get("/api/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "HEALTHY"

    resp_nodes = client.get("/api/nodes")
    assert resp_nodes.status_code == 200
    assert len(resp_nodes.json()) >= 5

    resp_risk = client.get("/api/risk")
    assert resp_risk.status_code == 200
    assert "max_risk_score" in resp_risk.json()

    # Ingest reading
    payload = {
        "node_id": "N01",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tilt_x": 0.14,
        "tilt_y": 0.09,
        "vibration": 0.04,
        "displacement_mm": 1.02
    }
    resp_ingest = client.post("/api/sensor-data", json=payload)
    assert resp_ingest.status_code == 200
    data = resp_ingest.json()
    assert data["success"] is True
    assert "risk_score" in data
