"""Brutal Security and Hardening Tests.
SIH26025 - NexGen | Authentication & Boundary Validation
"""

import os
import pytest
from starlette.testclient import TestClient
from api.main import app
import api.routes as routes


def test_api_key_when_configured(monkeypatch):
    """Verify that when MINE_API_KEY is configured, requests without key are rejected with 401."""
    monkeypatch.setattr(routes, "REQUIRED_API_KEY", "super-secret-sih-key")
    client = TestClient(app)

    payload = {
        "node_id": "N01",
        "timestamp": "2026-09-11T12:00:00",
        "tilt_x": 0.15,
        "tilt_y": 0.08,
        "vibration": 0.04,
        "displacement_mm": 1.02
    }

    # 1. Without header -> 401 Unauthorized
    res_unauth = client.post("/api/sensor-data", json=payload)
    assert res_unauth.status_code == 401

    # 2. With wrong key -> 401 Unauthorized
    res_wrong = client.post("/api/sensor-data", json=payload, headers={"X-API-Key": "wrong-key"})
    assert res_wrong.status_code == 401

    # 3. With correct key -> 200 OK
    res_ok = client.post("/api/sensor-data", json=payload, headers={"X-API-Key": "super-secret-sih-key"})
    assert res_ok.status_code == 200


def test_cors_middleware_headers():
    """Verify CORS responds with valid Origin headers."""
    client = TestClient(app)
    res = client.options("/api/health", headers={"Origin": "https://example.com", "Access-Control-Request-Method": "GET"})
    assert res.status_code == 200
