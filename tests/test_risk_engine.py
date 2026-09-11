"""Unit tests for the Risk Fusion Engine.
SIH26025 - NexGen
"""

import pytest
from src.risk_engine import RiskEngine


def test_risk_engine_normal():
    engine = RiskEngine()
    # Baseline normal values
    res = engine.calculate_risk(
        vibration_rms=0.04,
        tilt_magnitude=0.10,
        displacement_rate_mm_per_min=0.01,
        anomaly_score_norm=0.05,
        classifier_probs={"NORMAL": 0.95, "WARNING": 0.05, "HIGH": 0.0, "CRITICAL": 0.0}
    )
    assert res["risk_level"] == "NORMAL"
    assert res["risk_score"] <= 25.0


def test_risk_engine_critical():
    engine = RiskEngine()
    # Severe values
    res = engine.calculate_risk(
        vibration_rms=2.5,
        tilt_magnitude=6.5,
        displacement_rate_mm_per_min=2.0,
        anomaly_score_norm=0.98,
        classifier_probs={"NORMAL": 0.0, "WARNING": 0.0, "HIGH": 0.05, "CRITICAL": 0.95},
        temporal_trend_slope=1.2,
        spatial_abnormal_pct=1.0
    )
    assert res["risk_level"] == "CRITICAL"
    assert res["risk_score"] >= 75.0


def test_risk_engine_monotonicity():
    engine = RiskEngine()
    score_low = engine.calculate_risk(
        vibration_rms=0.05, tilt_magnitude=0.2, displacement_rate_mm_per_min=0.02, anomaly_score_norm=0.1
    )["risk_score"]

    score_med = engine.calculate_risk(
        vibration_rms=0.4, tilt_magnitude=1.5, displacement_rate_mm_per_min=0.3, anomaly_score_norm=0.5
    )["risk_score"]

    score_high = engine.calculate_risk(
        vibration_rms=1.5, tilt_magnitude=4.5, displacement_rate_mm_per_min=1.2, anomaly_score_norm=0.9
    )["risk_score"]

    assert score_low < score_med < score_high
