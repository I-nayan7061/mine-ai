"""Brutal Verification of Cold-Start Emergency Interlock Bypass.
SIH26025 - NexGen | Eliminating Buffer Warm-up Blind Spot
"""

import pytest
from src.inference import MineInferencePipeline


def test_cold_start_normal_remains_buffering():
    """Verify that benign readings during cold-start warm up buffer without false alarms."""
    pipeline = MineInferencePipeline()
    pipeline.buffer_manager.clear("N99")

    benign_reading = {
        "node_id": "N99",
        "timestamp": "2026-09-11T12:00:00",
        "vibration": 0.03,
        "tilt_x": 0.1,
        "tilt_y": 0.08,
        "displacement_mm": 1.05
    }

    res = pipeline.process_single_reading(benign_reading)
    assert res["success"] is True
    assert res["status"] == "BUFFERING"
    assert res["risk_level"] == "NORMAL"
    assert res["risk_score"] == 10.0


def test_cold_start_extreme_displacement_triggers_emergency_bypass():
    """Verify that sudden catastrophic displacement on sample 1 instantly triggers CRITICAL."""
    pipeline = MineInferencePipeline()
    pipeline.buffer_manager.clear("N98")

    critical_reading = {
        "node_id": "N98",
        "timestamp": "2026-09-11T12:00:00",
        "vibration": 0.05,
        "tilt_x": 0.2,
        "tilt_y": 0.1,
        "displacement_mm": 12.5  # Critical roof collapse > 10mm
    }

    res = pipeline.process_single_reading(critical_reading)
    assert res["success"] is True
    assert res["status"] == "EMERGENCY_INTERLOCK_TRIGGERED"
    assert res["risk_level"] == "CRITICAL"
    assert res["risk_score"] >= 75.0
    assert res["anomaly"] is True


def test_cold_start_extreme_tilt_triggers_emergency_bypass():
    """Verify that severe angular tilt (>5 degrees) on sample 1 triggers CRITICAL."""
    pipeline = MineInferencePipeline()
    pipeline.buffer_manager.clear("N97")

    severe_tilt_reading = {
        "node_id": "N97",
        "timestamp": "2026-09-11T12:00:00",
        "vibration": 0.04,
        "tilt_x": 4.5,
        "tilt_y": 3.2,
        "displacement_mm": 2.0
    }

    res = pipeline.process_single_reading(severe_tilt_reading)
    assert res["success"] is True
    assert res["status"] == "EMERGENCY_INTERLOCK_TRIGGERED"
    assert res["risk_level"] == "CRITICAL"
    assert res["risk_score"] >= 75.0
