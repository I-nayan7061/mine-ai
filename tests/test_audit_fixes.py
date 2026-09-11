"""Strict Unit Tests for Forensic Audit Bug Fixes.
SIH26025 - NexGen | Verifying Absolute Displacement, Directional Tilt, Spatial Immunity & Stuck Detection
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.explainability import MineExplainer
from src.inference import MineInferencePipeline
from src.risk_engine import RiskEngine
from src.spatial_features import MineSpatialGraph
from src.tilt_features import extract_tilt_features


def test_static_roof_sag_triggers_critical_and_bypasses_blasting_cap():
    """Verify that a 25mm static roof sag with 0 velocity triggers CRITICAL and is NOT capped at 28.0."""
    engine = RiskEngine()
    
    # 25 mm static sag, but velocity is 0.01 mm/min (stationary creep)
    result = engine.calculate_risk(
        vibration_rms=0.04,
        tilt_magnitude=0.30,
        displacement_rate_mm_per_min=0.01,
        anomaly_score_norm=0.10,
        classifier_probs={"NORMAL": 0.1, "WARNING": 0.2, "HIGH": 0.3, "CRITICAL": 0.4},
        temporal_trend_slope=0.005,
        spatial_abnormal_pct=0.0,
        displacement_mm=25.0
    )

    # Must be CRITICAL or HIGH, definitely NOT capped at 28.0 (NORMAL)!
    assert result["risk_score"] >= 75.0, f"Expected critical score >= 75, got {result['risk_score']}"
    assert result["risk_level"] == "CRITICAL"


def test_blasting_vibration_does_not_contaminate_spatial_graph():
    """Verify that high vibration shock (0.4g) without displacement does NOT flag neighbor nodes as anomalous."""
    graph = MineSpatialGraph()

    # Node N02 and N03 experience 0.4g blasting vibration but zero displacement (<1.0mm) and nominal tilt (<0.5 deg)
    fleet_states = {
        "N01": {"displacement_mm": 1.0, "tilt_magnitude": 0.2, "vibration_rms": 0.05},
        "N02": {"displacement_mm": 1.0, "tilt_magnitude": 0.2, "vibration_rms": 0.45},
        "N03": {"displacement_mm": 1.0, "tilt_magnitude": 0.2, "vibration_rms": 0.50},
        "N04": {"displacement_mm": 1.0, "tilt_magnitude": 0.2, "vibration_rms": 0.05},
        "N05": {"displacement_mm": 1.0, "tilt_magnitude": 0.2, "vibration_rms": 0.05}
    }

    feats = graph.compute_spatial_features("N01", fleet_states)
    # N01 is connected to N02. N02 has high vibration but 0 displacement. It should NOT be marked abnormal!
    assert feats["spatial_num_abnormal_neighbors"] == 0.0
    assert feats["spatial_pct_abnormal_neighbors"] == 0.0


def test_directional_tilt_reversal_captured():
    """Verify that angular shear reversal from +3 deg to -3 deg computes non-zero net vector change."""
    tx = np.array([3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0])
    ty = np.zeros_like(tx)

    feats = extract_tilt_features(tx, ty)
    # Scalar net_angular_change on magnitude will be 0.0 because |+3| - |-3| = 0
    assert feats["tilt_net_angular_change"] == 0.0
    # But directional net_vector_change must capture the 6.0 degree reversal!
    assert feats["tilt_net_vector_change"] == 6.0


def test_stuck_sensor_with_floating_point_micro_noise_detected():
    """Verify that stuck sensor with 1 LSB floating point jitter is detected by data quality validator."""
    loader = DataLoader()

    # Generate 150 samples of stuck sensor with 1e-6 noise
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-09-11 10:00:00", periods=150, freq="1s")
    stuck_disp = 1.250000 + rng.uniform(-1e-6, 1e-6, size=150)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "node_id": ["N01"] * 150,
        "tilt_x": [0.1] * 150,
        "tilt_y": [0.1] * 150,
        "vibration": [0.04] * 150,
        "displacement_mm": stuck_disp
    })

    report = loader.validate_dataframe(df)
    assert len(report.stuck_sensors_detected) > 0
    assert any(s["sensor"] == "displacement_mm" for s in report.stuck_sensors_detected)
