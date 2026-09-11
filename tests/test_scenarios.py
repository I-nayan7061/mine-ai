"""Geotechnical Scenario Verification Tests.
SIH26025 - NexGen | Automated Strata Movement Scenarios

Verifies end-to-end pipeline responses:
- SCENARIO A: Stable baseline -> NORMAL
- SCENARIO B: Small persistent roof sag -> WARNING
- SCENARIO C: Accelerated strata convergence & tilt -> HIGH
- SCENARIO D: Dynamic fracturing, rapid displacement jump & neighbour correlation -> CRITICAL
- SCENARIO E: High vibration blasting spike with no displacement -> False-Alarm Resistance
"""

from datetime import datetime, timedelta
import pytest
from src.inference import MineInferencePipeline


@pytest.fixture
def fresh_pipeline():
    pipe = MineInferencePipeline()
    pipe.buffer_manager.clear()
    return pipe


def test_scenario_a_stable_normal(fresh_pipeline):
    """Scenario A: Continuous stable sensor stream should maintain NORMAL status."""
    now = datetime(2026, 9, 10, 10, 0, 0)
    final_res = None
    for i in range(25):
        reading = {
            "node_id": "N01",
            "timestamp": (now + timedelta(seconds=i)).isoformat(),
            "tilt_x": 0.10,
            "tilt_y": 0.08,
            "vibration": 0.035,
            "displacement_mm": 1.00 + (i * 0.0005)  # Nominal static baseline
        }
        final_res = fresh_pipeline.process_single_reading(reading)

    assert final_res is not None
    assert final_res["risk_level"] == "NORMAL"
    assert final_res["risk_score"] < 25.0


def test_scenario_b_slow_sag_warning(fresh_pipeline):
    """Scenario B: Persistent slow roof sag should escalate to WARNING."""
    now = datetime(2026, 9, 10, 10, 0, 0)
    final_res = None
    for i in range(30):
        reading = {
            "node_id": "N02",
            "timestamp": (now + timedelta(seconds=i)).isoformat(),
            "tilt_x": 0.10 + (i * 0.015),
            "tilt_y": 0.08 + (i * 0.01),
            "vibration": 0.05,
            "displacement_mm": 1.00 + (i * 0.04)  # Slow continuous strata creep
        }
        final_res = fresh_pipeline.process_single_reading(reading)

    assert final_res is not None
    assert final_res["risk_level"] in ["WARNING", "HIGH"]
    assert final_res["risk_score"] >= 25.0


def test_scenario_c_accelerated_convergence_high(fresh_pipeline):
    """Scenario C: Accelerated convergence and significant tilt should elevate to HIGH."""
    now = datetime(2026, 9, 10, 10, 0, 0)
    final_res = None
    for i in range(35):
        reading = {
            "node_id": "N03",
            "timestamp": (now + timedelta(seconds=i)).isoformat(),
            "tilt_x": 0.10 + (i * 0.05),
            "tilt_y": 0.08 + (i * 0.04),
            "vibration": 0.15 + (i * 0.01),
            "displacement_mm": 1.00 + (i * 0.12)  # Accelerated strata separation
        }
        final_res = fresh_pipeline.process_single_reading(reading)

    assert final_res is not None
    assert final_res["risk_level"] in ["HIGH", "CRITICAL"]
    assert final_res["risk_score"] >= 50.0


def test_scenario_d_dynamic_fracture_critical(fresh_pipeline):
    """Scenario D: Dynamic tensile fracture, rapid displacement jump & seismic vibrations -> CRITICAL."""
    now = datetime(2026, 9, 10, 10, 0, 0)

    # Prime neighbors as abnormal
    fresh_pipeline.fleet_states["N02"] = {"displacement_mm": 5.0, "tilt_magnitude": 2.5, "vibration_rms": 0.6, "is_abnormal": True, "risk_score": 70.0}
    fresh_pipeline.fleet_states["N04"] = {"displacement_mm": 6.2, "tilt_magnitude": 3.0, "vibration_rms": 0.7, "is_abnormal": True, "risk_score": 75.0}

    final_res = None
    for i in range(35):
        reading = {
            "node_id": "N03",
            "timestamp": (now + timedelta(seconds=i)).isoformat(),
            "tilt_x": 0.10 + (i * 0.12),
            "tilt_y": 0.08 + (i * 0.08),
            "vibration": 0.40 + (i * 0.04),
            "displacement_mm": 1.00 + (i * 0.35)  # Sudden rapid convergence jump
        }
        final_res = fresh_pipeline.process_single_reading(reading)

    assert final_res is not None
    assert final_res["risk_level"] == "CRITICAL"
    assert final_res["risk_score"] >= 75.0
    assert final_res["anomaly"] is True
    assert len(final_res["top_contributing_features"]) > 0


def test_scenario_e_blasting_noise_resistance(fresh_pipeline):
    """Scenario E: Passing haulage / blasting causes high vibration for 5 seconds

    but zero displacement change; should NOT falsely trigger CRITICAL.
    """
    now = datetime(2026, 9, 10, 10, 0, 0)
    final_res = None
    for i in range(25):
        # Heavy vibration spike in middle
        vib_val = 1.8 if (10 <= i <= 15) else 0.04
        reading = {
            "node_id": "N01",
            "timestamp": (now + timedelta(seconds=i)).isoformat(),
            "tilt_x": 0.10,
            "tilt_y": 0.08,
            "vibration": vib_val,
            "displacement_mm": 1.00  # Zero ground displacement
        }
        final_res = fresh_pipeline.process_single_reading(reading)

    assert final_res is not None
    # Must NOT panic to CRITICAL when displacement is completely static
    assert final_res["risk_level"] != "CRITICAL"
