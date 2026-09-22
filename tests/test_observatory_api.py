"""Test suite for Moonidih Mine Observatory and Multi-Modal Features.
SIH26025 - NexGen | Multi-Modal Subsidence Intelligence Tests
"""

import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app
from src.geo_fetcher import get_moonidih_environmental_snapshot, MOONIDIH_MINE_METADATA


@pytest.mark.asyncio
async def test_moonidih_observatory_endpoint():
    """Verify /api/mine/jharia-moonidih returns complete multi-modal dossier."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/mine/jharia-moonidih")
        assert res.status_code == 200
        data = res.json()
        assert "metadata" in data
        assert data["metadata"]["coalfield"] == "Jharia Coalfield"
        assert data["coordinates"]["latitude"] == 23.7438
        assert data["coordinates"]["longitude"] == 86.4172
        assert "weather" in data
        assert "rain_cum_24h_mm" in data["weather"]
        assert "soil_moisture_deep" in data["weather"]
        assert "satellite" in data
        assert "sentinel1_insar_velocity_mm_yr" in data["satellite"]
        assert "sentinel2_ndvi_anomaly" in data["satellite"]
        assert "landsat_thermal_anomaly_k" in data["satellite"]
        assert "geotech_indices" in data
        assert "physical_sensors" in data
        assert "nodes" in data["physical_sensors"]
        assert "N03" in data["physical_sensors"]["nodes"]
        assert "sensor_specs" in data
        assert "displacement" in data["sensor_specs"]
        assert "satellite_specs" in data
        assert "sentinel_1_insar" in data["satellite_specs"]
        assert "weather_specs" in data
        assert "accuracy_benchmarks" in data
        assert "full_multimodal_ai_model_d" in data["accuracy_benchmarks"]
        assert data["accuracy_benchmarks"]["full_multimodal_ai_model_d"]["accuracy"] == 98.35
        assert "sample_records" in data
        assert isinstance(data["sample_records"], list)


@pytest.mark.asyncio
async def test_moonidih_export_csv_endpoint():
    """Verify /api/mine/export-csv returns downloadable CSV stream."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/mine/export-csv")
        assert res.status_code == 200
        assert "text/csv" in res.headers.get("content-type", "")
        assert "attachment" in res.headers.get("content-disposition", "")
        content = res.text
        assert "timestamp" in content
        assert "node_id" in content
        assert "displacement_mm" in content
        assert "weather_rain_cum_24h_mm" in content
        assert "satellite_insar_velocity_mm_yr" in content


@pytest.mark.asyncio
async def test_multimodal_prediction_payload():
    """Verify /api/sensor-data accepts and merges weather/satellite overrides."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "node_id": "N03",
            "timestamp": "2026-09-11T22:20:00",
            "tilt_x": 0.25,
            "tilt_y": 0.15,
            "vibration": 0.05,
            "displacement_mm": 1.20,
            "weather": {
                "rain_rate_mm_h": 5.0,
                "rain_cum_24h_mm": 35.0,
                "rain_cum_72h_mm": 85.0,
                "soil_moisture_deep": 0.42
            },
            "satellite": {
                "sentinel1_insar_velocity_mm_yr": -28.5,
                "sentinel2_ndvi_anomaly": -0.18,
                "landsat_thermal_anomaly_k": 3.8
            }
        }
        res = await client.post("/api/sensor-data", json=payload)
        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert "weather_summary" in body
        assert body["weather_summary"]["rain_cum_24h_mm"] == 35.0
        assert "satellite_summary" in body
        assert body["satellite_summary"]["insar_velocity_mm_yr"] == -28.5
        assert "subscores" in body
        assert "weather_severity" in body["subscores"]
        assert "satellite_severity" in body["subscores"]
