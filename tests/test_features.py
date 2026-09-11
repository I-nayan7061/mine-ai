"""Unit tests for feature engineering modules.
SIH26025 - NexGen
"""

import numpy as np
import pytest

from src.displacement_features import extract_displacement_features
from src.spatial_features import MineSpatialGraph
from src.tilt_features import extract_tilt_features
from src.vibration_features import extract_vibration_features


def test_vibration_features():
    # Sine wave vibration
    t = np.linspace(0, 10, 100)
    vib = 0.5 * np.sin(2 * np.pi * 1.5 * t) + 0.1
    feats = extract_vibration_features(vib, sampling_rate_hz=10.0)

    assert "vib_rms" in feats
    assert "vib_peak" in feats
    assert "vib_crest_factor" in feats
    assert feats["vib_rms"] > 0.0
    assert feats["vib_peak"] >= feats["vib_rms"]


def test_vibration_empty():
    feats = extract_vibration_features(np.array([]))
    assert feats["vib_rms"] == 0.0


def test_tilt_features():
    tx = np.linspace(0.1, 1.5, 50)
    ty = np.linspace(0.1, 1.0, 50)
    feats = extract_tilt_features(tx, ty)

    assert "tilt_mag_current" in feats
    assert "tilt_rate_max" in feats
    assert "tilt_trend_slope" in feats
    assert feats["tilt_mag_current"] > 0.0
    # Positive trend slope since values increase
    assert feats["tilt_trend_slope"] > 0.0


def test_displacement_features():
    # Increasing displacement
    disp = np.array([1.0, 1.2, 1.5, 2.0, 2.8, 3.7])
    feats = extract_displacement_features(disp)

    assert "disp_current" in feats
    assert "disp_rate_max" in feats
    assert "disp_trend_slope" in feats
    assert feats["disp_current"] == 3.7
    assert feats["disp_rate_max"] > 0.0


def test_spatial_features():
    graph = MineSpatialGraph()
    neighbors = graph.get_neighbors("N02")
    assert "N01" in neighbors
    assert "N03" in neighbors

    dist = graph.get_distance("N01", "N02")
    assert dist == 25.0

    states = {
        "N01": {"displacement_mm": 1.0, "tilt_magnitude": 0.1, "risk_score": 10.0, "is_abnormal": False},
        "N02": {"displacement_mm": 1.5, "tilt_magnitude": 0.3, "risk_score": 20.0, "is_abnormal": False},
        "N03": {"displacement_mm": 8.0, "tilt_magnitude": 2.5, "risk_score": 85.0, "is_abnormal": True},
    }
    feats = graph.compute_spatial_features("N02", states)
    assert feats["spatial_num_abnormal_neighbors"] >= 1.0
    assert feats["spatial_disp_gradient_max"] > 0.0
