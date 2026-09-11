"""Integration tests for FastAPI endpoints.
SIH26025 - NexGen
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["models_loaded"] is True


def test_get_nodes_endpoint():
    response = client.get("/api/nodes")
    assert response.status_code == 200
    nodes = response.json()
    assert len(nodes) >= 5
    assert any(n["node_id"] == "N01" for n in nodes)


def test_get_risk_endpoint():
    response = client.get("/api/risk")
    assert response.status_code == 200
    data = response.json()
    assert "max_risk_score" in data
    assert "total_nodes" in data


def test_sensor_data_ingestion():
    payload = {
        "node_id": "N01",
        "timestamp": "2026-09-10T12:00:00",
        "tilt_x": 0.12,
        "tilt_y": 0.08,
        "vibration": 0.035,
        "displacement_mm": 1.05
    }
    response = client.post("/api/sensor-data", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["node_id"] == "N01"
    assert "risk_score" in data
    assert "risk_level" in data
    assert "class_probabilities" in data


def test_sensor_data_validation_rejection():
    # Negative displacement far beyond physical bounds
    bad_payload = {
        "node_id": "N01",
        "timestamp": "2026-09-10T12:00:00",
        "tilt_x": 999.0,  # Invalid
        "tilt_y": 0.0,
        "vibration": -5.0,  # Invalid
        "displacement_mm": 99999.0  # Invalid
    }
    response = client.post("/api/sensor-data", json=bad_payload)
    assert response.status_code == 400
