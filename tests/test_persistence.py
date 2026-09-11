"""Brutal Integration Tests for Persistent Database Layer.
SIH26025 - NexGen | SQLite WAL Persistence Tests
"""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
import pytest
import pytest_asyncio

from src.db import DatabaseManager


@pytest.fixture
def temp_db_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_database_init_and_wal_mode(temp_db_dir):
    db_path = temp_db_dir / "test_mine.db"
    mgr = DatabaseManager(db_path=db_path)
    await mgr.init_db()

    assert db_path.exists()

    # Check WAL mode
    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("PRAGMA journal_mode;") as cursor:
            row = await cursor.fetchone()
            assert row[0].lower() == "wal"


@pytest.mark.asyncio
async def test_reading_persistence_and_retrieval(temp_db_dir):
    db_path = temp_db_dir / "test_mine.db"
    mgr = DatabaseManager(db_path=db_path)
    await mgr.init_db()

    sample_reading = {
        "node_id": "N03",
        "timestamp": "2026-09-11T12:00:00",
        "sensor_values": {
            "vibration": 0.045,
            "tilt_x": 0.12,
            "tilt_y": 0.08,
            "displacement_mm": 1.25
        },
        "anomaly": False,
        "anomaly_score": 0.04,
        "risk_score": 14.5,
        "risk_level": "NORMAL",
        "confidence": 0.96,
        "class_probabilities": {"NORMAL": 0.96, "WARNING": 0.04, "HIGH": 0.0, "CRITICAL": 0.0},
        "subscores": {"vibration_severity": 10.0},
        "top_contributing_features": ["vib_rms"],
        "human_explanations": ["Ground conditions stable."],
        "neighbour_anomalies": 0
    }

    row_id = await mgr.save_reading(sample_reading)
    assert row_id > 0

    history = await mgr.get_recent_history(node_id="N03", limit=10)
    assert len(history) == 1
    assert history[0]["node_id"] == "N03"
    assert history[0]["risk_score"] == 14.5
    assert history[0]["sensor_values"]["displacement_mm"] == 1.25


@pytest.mark.asyncio
async def test_alert_persistence_and_retrieval(temp_db_dir):
    db_path = temp_db_dir / "test_mine.db"
    mgr = DatabaseManager(db_path=db_path)
    await mgr.init_db()

    alert = {
        "id": "alert-99",
        "node_id": "N03",
        "timestamp": "2026-09-11T12:05:00",
        "severity": "CRITICAL",
        "risk_score": 89.5,
        "reason": "Accelerated roof convergence detected",
        "top_factors": ["disp_rate_max", "tilt_rate_max"]
    }

    await mgr.save_alert(alert)

    alerts = await mgr.get_recent_alerts(limit=5)
    assert len(alerts) == 1
    assert alerts[0]["id"] == "alert-99"
    assert alerts[0]["severity"] == "CRITICAL"
    assert alerts[0]["risk_score"] == 89.5
    assert "disp_rate_max" in alerts[0]["top_factors"]


@pytest.mark.asyncio
async def test_fleet_node_upsert_and_recovery(temp_db_dir):
    db_path = temp_db_dir / "test_mine.db"
    mgr = DatabaseManager(db_path=db_path)
    await mgr.init_db()

    await mgr.upsert_node(
        node_id="N01",
        gallery="Gallery-North-1",
        x=10.0,
        y=20.0,
        latest_risk_score=15.0,
        latest_risk_level="NORMAL",
        last_updated="2026-09-11T12:00:00",
        online=True
    )

    nodes = await mgr.get_fleet_nodes()
    assert len(nodes) == 1
    assert nodes[0]["node_id"] == "N01"
    assert nodes[0]["latest_risk_score"] == 15.0

    # Upsert with new critical score
    await mgr.upsert_node(
        node_id="N01",
        gallery="Gallery-North-1",
        x=10.0,
        y=20.0,
        latest_risk_score=85.0,
        latest_risk_level="CRITICAL",
        last_updated="2026-09-11T12:01:00",
        online=True
    )

    nodes_updated = await mgr.get_fleet_nodes()
    assert len(nodes_updated) == 1
    assert nodes_updated[0]["latest_risk_score"] == 85.0
    assert nodes_updated[0]["latest_risk_level"] == "CRITICAL"
