"""FastAPI Route Handlers for Mine Subsidence AI System.
SIH26025 - NexGen | Real-Time Ingestion, WebSockets & Telemetry Endpoints
"""

import collections
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Security, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from .schemas import AlertItem, MineObservatoryData, NodeStatus, RiskPredictionResponse, SensorReadingRequest
from src.config import app_config
from src.db import db_manager
from src.geo_fetcher import MOONIDIH_MINE_METADATA, get_moonidih_environmental_snapshot
from src.inference import MineInferencePipeline
from src.utils import get_logger, load_json

logger = get_logger("mine_ai.api")

router = APIRouter(prefix="/api", tags=["Mine Monitoring"])

# Initialize singleton inference pipeline
pipeline = MineInferencePipeline()

# In-memory history and active alerts for low-latency cache
MAX_HISTORY = 1000
history_records = collections.deque(maxlen=MAX_HISTORY)
alerts_list = collections.deque(maxlen=200)

# Fleet registry
fleet_registry = {}
graph_nodes = app_config.get("spatial_graph.nodes", {})
for n_id, n_info in graph_nodes.items():
    fleet_registry[n_id] = {
        "node_id": n_id,
        "gallery": n_info.get("gallery", "Main Gallery"),
        "x": n_info.get("x", 0.0),
        "y": n_info.get("y", 0.0),
        "latest_risk_score": 10.0,
        "latest_risk_level": "NORMAL",
        "last_updated": None,
        "online": True
    }


# ==========================================================
# WEBSOCKET MANAGER FOR REAL-TIME STREAMING
# ==========================================================
class ConnectionManager:
    """Manages connected WebSocket clients and broadcasts live telemetry."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New WebSocket client connected. Active: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("WebSocket client disconnected. Active: %d", len(self.active_connections))

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast JSON message to all active dashboard connections."""
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        for dead in dead_connections:
            self.disconnect(dead)


ws_manager = ConnectionManager()


# ==========================================================
# SECURITY & AUTHENTICATION
# ==========================================================
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
REQUIRED_API_KEY = os.getenv("MINE_API_KEY", "")


async def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> Optional[str]:
    """Verify optional/mandatory API Key depending on MINE_API_KEY environment variable."""
    if REQUIRED_API_KEY and api_key != REQUIRED_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-API-Key authorization header"
        )
    return api_key


# ==========================================================
# TELEMETRY INGESTION & WEBSOCKET ENDPOINTS
# ==========================================================
@router.post("/sensor-data", response_model=RiskPredictionResponse)
async def ingest_sensor_data(
    reading: SensorReadingRequest,
    _auth: Optional[str] = Depends(verify_api_key)
):
    """Primary telemetry ingestion endpoint called by ESP32 microcontrollers or LoRa Gateways."""
    raw_dict = reading.model_dump()
    result = pipeline.process_single_reading(raw_dict)

    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("validation_errors", ["Invalid sensor data"]))

    # Update fleet registry
    node_id = reading.node_id
    if node_id in fleet_registry:
        fleet_registry[node_id]["latest_risk_score"] = result["risk_score"]
        fleet_registry[node_id]["latest_risk_level"] = result["risk_level"]
        fleet_registry[node_id]["last_updated"] = result["timestamp"]
        fleet_registry[node_id]["online"] = True
    else:
        fleet_registry[node_id] = {
            "node_id": node_id,
            "gallery": "Unassigned Sector",
            "x": 0.0,
            "y": 0.0,
            "latest_risk_score": result["risk_score"],
            "latest_risk_level": result["risk_level"],
            "last_updated": result["timestamp"],
            "online": True
        }

    # Store in in-memory history cache
    history_records.append(result)

    # Persist in SQLite database asynchronously
    try:
        await db_manager.save_reading(result)
        n_info = fleet_registry[node_id]
        await db_manager.upsert_node(
            node_id=node_id,
            gallery=n_info["gallery"],
            x=n_info["x"],
            y=n_info["y"],
            latest_risk_score=n_info["latest_risk_score"],
            latest_risk_level=n_info["latest_risk_level"],
            last_updated=n_info["last_updated"],
            online=n_info["online"]
        )
    except Exception as e:
        logger.warning("Failed to persist reading to SQLite: %s", e)

    # Check alert condition
    alert_entry = None
    if result["risk_level"] in ["WARNING", "HIGH", "CRITICAL"]:
        alert_entry = {
            "id": str(uuid.uuid4())[:8],
            "node_id": node_id,
            "timestamp": result["timestamp"],
            "severity": result["risk_level"],
            "risk_score": result["risk_score"],
            "reason": result["human_explanations"][0] if result.get("human_explanations") else "Geotechnical anomaly detected",
            "top_factors": result.get("top_contributing_features", [])
        }
        alerts_list.appendleft(alert_entry)
        try:
            await db_manager.save_alert(alert_entry)
        except Exception as e:
            logger.warning("Failed to persist alert to SQLite: %s", e)

    # Broadcast real-time update to all connected dashboard WebSocket clients
    await ws_manager.broadcast({
        "type": "TELEMETRY_UPDATE",
        "reading": result,
        "node": fleet_registry[node_id],
        "alert": alert_entry
    })

    return result


@router.post("/predict", response_model=RiskPredictionResponse)
async def predict_direct(reading: SensorReadingRequest, _auth: Optional[str] = Depends(verify_api_key)):
    """Direct inference route for on-demand diagnosis."""
    return await ingest_sensor_data(reading, _auth=_auth)


@router.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Real-time bi-directional WebSocket stream for live dashboard telemetry."""
    await ws_manager.connect(websocket)
    try:
        # Send initial full state snapshot to newly connected client
        initial_payload = {
            "type": "INITIAL_SNAPSHOT",
            "nodes": list(fleet_registry.values()),
            "recent_alerts": list(alerts_list)[:10],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await websocket.send_json(initial_payload)

        # Keep alive connection and handle client heartbeats
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.debug("WebSocket exception: %s", e)
        ws_manager.disconnect(websocket)


@router.get("/nodes", response_model=List[NodeStatus])
async def get_nodes():
    """Returns list of registered ESP32 mine monitoring nodes with spatial positions & risk status."""
    # Try reading from database if in-memory registry is fresh
    db_nodes = await db_manager.get_fleet_nodes()
    if db_nodes:
        # Merge with in-memory registry
        for n in db_nodes:
            if n["node_id"] in fleet_registry:
                fleet_registry[n["node_id"]].update(n)
            else:
                fleet_registry[n["node_id"]] = n
    return list(fleet_registry.values())


@router.get("/risk")
async def get_fleet_risk_summary():
    """Returns aggregated fleet risk summary and identified risk zones."""
    scores = [n["latest_risk_score"] for n in fleet_registry.values()]
    max_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0

    critical_nodes = [n["node_id"] for n in fleet_registry.values() if n["latest_risk_level"] == "CRITICAL"]
    high_nodes = [n["node_id"] for n in fleet_registry.values() if n["latest_risk_level"] == "HIGH"]
    warning_nodes = [n["node_id"] for n in fleet_registry.values() if n["latest_risk_level"] == "WARNING"]

    # Identify potential risk zone
    risk_zone = None
    if critical_nodes or high_nodes:
        affected = critical_nodes + high_nodes
        risk_zone = {
            "zone_id": "ZONE-NORTH-EXCAVATION",
            "active_nodes": affected,
            "severity": "CRITICAL" if critical_nodes else "HIGH",
            "description": f"Strata instability clustering across nodes {', '.join(affected)}"
        }

    return {
        "max_risk_score": round(max_score, 2),
        "average_risk_score": round(avg_score, 2),
        "total_nodes": len(fleet_registry),
        "critical_nodes_count": len(critical_nodes),
        "high_nodes_count": len(high_nodes),
        "warning_nodes_count": len(warning_nodes),
        "normal_nodes_count": len(fleet_registry) - len(critical_nodes) - len(high_nodes) - len(warning_nodes),
        "identified_risk_zone": risk_zone
    }


@router.get("/history")
async def get_history(node_id: Optional[str] = None, limit: int = Query(50, ge=1, le=500)):
    """Returns chronological sensor readings and risk evolution from DB / cache."""
    # First check DB
    try:
        db_history = await db_manager.get_recent_history(node_id=node_id, limit=limit)
        if db_history:
            return db_history
    except Exception as e:
        logger.debug("Falling back to in-memory history: %s", e)

    filtered = list(history_records)
    if node_id:
        filtered = [r for r in filtered if r["node_id"] == node_id]
    return filtered[-limit:]


@router.get("/alerts")
async def get_alerts(limit: int = Query(20, ge=1, le=100)):
    """Returns active risk alerts with root-cause explanations."""
    try:
        db_alerts = await db_manager.get_recent_alerts(limit=limit)
        if db_alerts:
            return db_alerts
    except Exception as e:
        logger.debug("Falling back to in-memory alerts: %s", e)

    return list(alerts_list)[:limit]


@router.get("/model-status")
async def get_model_status():
    """Returns loaded models, metadata, feature list, and accuracy benchmarks."""
    meta_path = app_config.resolve_path(app_config.get("paths.models_dir", "models")) / "model_metadata.json"
    try:
        metadata = load_json(meta_path)
    except Exception:
        metadata = {"status": "Model metadata not found"}

    return {
        "status": "OPERATIONAL",
        "primary_model": pipeline.classifier.model_type if pipeline.classifier else "None",
        "anomaly_detector": "IsolationForest",
        "selected_features_count": len(pipeline.selected_features),
        "selected_features": pipeline.selected_features,
        "metadata": metadata
    }


@router.get("/health")
async def get_health():
    """Health check endpoint."""
    return {
        "status": "HEALTHY",
        "project": "Mine Subsidence AI (SIH26025 - NexGen)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_loaded": pipeline.classifier is not None and pipeline.anomaly_detector is not None,
        "active_nodes": len(fleet_registry),
        "buffered_history_count": len(history_records),
        "websocket_active_clients": len(ws_manager.active_connections)
    }


@router.get("/mine/jharia-moonidih")
async def get_jharia_moonidih_observatory():
    """Returns complete multi-modal geotechnical dossier for Moonidih Colliery (Jharia Coalfield).
    Fuses real-time/cached Open-Meteo weather, Sentinel-1 InSAR subsidence,
    Sentinel-2 NDVI/NDMI, Landsat-9 LST, and subterranean gallery readings.
    """
    snapshot = get_moonidih_environmental_snapshot()

    # Load recent samples from synthetic / processed dataset for live telemetry explorer
    sample_records = []
    dataset_path = app_config.resolve_path("mine-ai/data/synthetic/synthetic_mine_subsidence_dataset.csv")
    if not dataset_path.exists():
        dataset_path = Path("mine-ai/data/synthetic/synthetic_mine_subsidence_dataset.csv")
    if not dataset_path.exists():
        dataset_path = Path("data/synthetic/synthetic_mine_subsidence_dataset.csv")

    if dataset_path.exists():
        import pandas as pd
        try:
            df = pd.read_csv(dataset_path)
            # Pick a representative sample across nodes (last 80 rows)
            sample_records = df.tail(80).to_dict(orient="records")
            for r in sample_records:
                if "timestamp" in r:
                    r["timestamp"] = str(r["timestamp"])
        except Exception as e:
            logger.warning("Error reading sample records: %s", e)

    return {
        "metadata": MOONIDIH_MINE_METADATA,
        "coordinates": MOONIDIH_MINE_METADATA["coordinates"],
        "weather": snapshot.get("weather", {}),
        "satellite": snapshot.get("satellite", {}),
        "geotech_indices": snapshot.get("geotech_indices", {}),
        "recent_records_count": len(sample_records),
        "sample_records": sample_records
    }


@router.get("/mine/export-csv")
async def export_mine_csv():
    """Generates a downloadable CSV containing all synchronized parameters across
    Hardware IoT, Surface Meteorology, and Satellite Earth Observation for Moonidih Colliery.
    """
    dataset_path = app_config.resolve_path("mine-ai/data/synthetic/synthetic_mine_subsidence_dataset.csv")
    if not dataset_path.exists():
        dataset_path = Path("mine-ai/data/synthetic/synthetic_mine_subsidence_dataset.csv")
    if not dataset_path.exists():
        dataset_path = Path("data/synthetic/synthetic_mine_subsidence_dataset.csv")

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    with open(dataset_path, "r", encoding="utf-8") as f:
        csv_data = f.read()

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=moonidih_jharia_multimodal_dataset.csv"
        }
    )


# ==========================================================
# PART 28: MULTI-MODAL & PROVENANCE REST ENDPOINTS
# ==========================================================
@router.get("/environment")
async def get_environment():
    """Returns real-time integrated surface environmental & remote-sensing telemetry."""
    from src.geo_fetcher import get_moonidih_environmental_snapshot
    return get_moonidih_environmental_snapshot()


@router.get("/weather")
async def get_weather():
    """Returns India Meteorological Department (IMD) / Open-Meteo calibrated weather feed with provenance."""
    from src.weather_features import IndiaWeatherProvider
    provider = IndiaWeatherProvider()
    return provider.get_weather_observation()


@router.get("/satellite")
async def get_satellite():
    """Returns Sentinel-1 InSAR & ISRO Bhoonidhi NISAR regional deformation metrics with provenance."""
    from src.satellite_features import IndiaSatelliteProvider
    provider = IndiaSatelliteProvider()
    return provider.get_satellite_observation()


@router.get("/multimodal-status")
async def get_multimodal_status():
    """Returns active synchronization status across IoT sensors, meteorology, and satellite layers."""
    meta_path = app_config.resolve_path("data/processed/multimodal_metadata.json")
    if not meta_path.exists():
        meta_path = Path("mine-ai/data/processed/multimodal_metadata.json")
    if meta_path.exists():
        try:
            return load_json(meta_path)
        except Exception:
            pass
    return {"status": "OPERATIONAL", "modalities": ["sensors_1hz", "weather_hourly", "satellite_insar_12d"]}


@router.get("/model-info")
async def get_model_info():
    """Returns complete model metadata, feature schema, latency benchmarks, and ablation summary."""
    meta_path = app_config.resolve_path("models/model_metadata.json")
    if not meta_path.exists():
        meta_path = Path("mine-ai/models/model_metadata.json")
    if meta_path.exists():
        try:
            return load_json(meta_path)
        except Exception:
            pass
    return {"status": "OPERATIONAL", "model": "xgboost_multimodal_v4"}


@router.get("/data-provenance")
async def get_data_provenance():
    """Returns explicit data provenance for all external meteorological, radar, and optical sources."""
    from src.weather_features import IndiaWeatherProvider
    from src.satellite_features import IndiaSatelliteProvider
    w_prov = IndiaWeatherProvider().default_provenance
    s_prov = IndiaSatelliteProvider().default_provenance
    return {
        "mine_aoi": "Moonidih Underground Colliery, Jharia Coalfield, BCCL, Dhanbad, Jharkhand",
        "coordinates": {"latitude": 23.7438, "longitude": 86.4172},
        "weather_data": w_prov,
        "satellite_data": s_prov,
        "underground_sensors": {
            "vibration": "MPU6050 Accelerometer (g) - 1.0 Hz",
            "tilt": "MPU6050 Dual-Axis Inclinometer (deg) - 1.0 Hz",
            "displacement": "VL53L1X Time-of-Flight Strata Convergence (mm) - 1.0 Hz"
        },
        "data_provenance_disclaimer": "Satellite InSAR provides regional surface subsidence context (12-day pass); subterranean IoT sensors provide real-time 1.0 Hz strata dynamics. Synthetic scenario datasets are utilized for prototype calibration."
    }

