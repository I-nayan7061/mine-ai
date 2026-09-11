"""Brutal Tests for WebSocket Real-Time Telemetry Streaming.
SIH26025 - NexGen | WebSocket Handshake, Snapshot & Broadcast Tests
"""

import pytest
from starlette.testclient import TestClient
from api.main import app


def test_websocket_connection_and_snapshot():
    """Verify WebSocket connects, receives initial snapshot, and responds to ping/pong."""
    client = TestClient(app)
    with client.websocket_connect("/api/ws/telemetry") as websocket:
        # 1. Receive initial snapshot
        data = websocket.receive_json()
        assert data["type"] == "INITIAL_SNAPSHOT"
        assert "nodes" in data
        assert len(data["nodes"]) >= 5

        # 2. Ping-pong keepalive
        websocket.send_text("ping")
        resp = websocket.receive_text()
        assert resp == "pong"


def test_websocket_broadcast_on_sensor_data_ingestion():
    """Verify that ingesting sensor data immediately broadcasts to all active WebSockets."""
    client = TestClient(app)
    with client.websocket_connect("/api/ws/telemetry") as websocket:
        # Flush initial snapshot
        init_data = websocket.receive_json()
        assert init_data["type"] == "INITIAL_SNAPSHOT"

        # Ingest reading via REST
        payload = {
            "node_id": "N03",
            "timestamp": "2026-09-11T12:00:00",
            "tilt_x": 0.15,
            "tilt_y": 0.08,
            "vibration": 0.04,
            "displacement_mm": 1.02
        }
        res = client.post("/api/sensor-data", json=payload)
        assert res.status_code == 200

        # Receive broadcast over WebSocket
        broadcast_msg = websocket.receive_json()
        assert broadcast_msg["type"] == "TELEMETRY_UPDATE"
        assert broadcast_msg["reading"]["node_id"] == "N03"
        assert broadcast_msg["node"]["node_id"] == "N03"
