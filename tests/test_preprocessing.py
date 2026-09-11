"""Unit tests for preprocessing and time-windowing.
SIH26025 - NexGen
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import Preprocessor
from src.signal_processing import apply_butterworth_filter, apply_rolling_median_filter, compute_spectral_features


def test_rolling_median_filter():
    signal_with_spike = np.array([1.0, 1.0, 50.0, 1.0, 1.0])
    filtered = apply_rolling_median_filter(signal_with_spike, window_size=3)
    # The impulsive spike (50.0) should be attenuated
    assert filtered[2] < 10.0


def test_butterworth_filter_safe():
    short_signal = np.array([1.0, 2.0])
    # Should safely return copy without crashing
    res = apply_butterworth_filter(short_signal)
    assert len(res) == 2


def test_spectral_features_empty():
    feats = compute_spectral_features(np.array([0.0]))
    assert feats["dominant_frequency"] == 0.0


def test_preprocessor_cleaning():
    prep = Preprocessor()
    now = datetime(2026, 9, 10, 12, 0, 0)
    df = pd.DataFrame([
        {"timestamp": (now + timedelta(seconds=1)).isoformat(), "node_id": "N01", "tilt_x": 0.1, "tilt_y": 0.1, "vibration": 0.05, "displacement_mm": 1.0},
        {"timestamp": (now + timedelta(seconds=1)).isoformat(), "node_id": "N01", "tilt_x": 0.1, "tilt_y": 0.1, "vibration": 0.05, "displacement_mm": 1.0},  # Duplicate
        {"timestamp": (now + timedelta(seconds=2)).isoformat(), "node_id": "N01", "tilt_x": 999.0, "tilt_y": -999.0, "vibration": 50.0, "displacement_mm": 5000.0},  # Out of bounds
    ])

    cleaned = prep.clean_dataframe(df)
    assert len(cleaned) == 2  # Deduplicated
    # Check bounded
    assert cleaned["tilt_x"].max() <= 90.0
    assert cleaned["tilt_y"].min() >= -90.0
    assert cleaned["vibration"].max() <= 10.0


def test_window_generation():
    prep = Preprocessor()
    now = datetime(2026, 9, 10, 12, 0, 0)
    records = []
    for i in range(120):
        records.append({
            "timestamp": (now + timedelta(seconds=i)).isoformat(),
            "node_id": "N01",
            "tilt_x": 0.1,
            "tilt_y": 0.1,
            "vibration": 0.05,
            "displacement_mm": 1.0,
            "risk_label": "NORMAL"
        })
    df = pd.DataFrame(records)
    windows = prep.create_rolling_windows(df)
    assert len(windows) > 0
    assert windows[0]["node_id"] == "N01"
    assert len(windows[0]["data"]) >= 10
