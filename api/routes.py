"""FastAPI Route Handlers for Mine Subsidence AI System.
SIH26025 - NexGen | Real-Time Ingestion & Telemetry Endpoints
"""

import collections
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .schemas import AlertItem, NodeStatus, RiskPredictionResponse, SensorReadingRequest
from src.config import app_config
from src.inference import MineInferencePipeline
from src.utils import get_logger, load_json

logger = get_logger("mine_ai.api")

router = APIRouter(prefix="/api", tags=["Mine Monitoring"])

# Initialize singleton inference pipeline
pipeline = MineInferencePipeline()

# In-memory history and active alerts
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


@router.post("/sensor-data", response_model=RiskPredictionResponse)
async def ingest_sensor_data(reading: SensorReadingRequest):
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

    # Store in history
    history_records.append(result)

    # Check alert condition
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

    return result


@router.post("/predict", response_model=RiskPredictionResponse)
async def predict_direct(reading: SensorReadingRequest):
    """Direct inference route for on-demand diagnosis."""
    return await ingest_sensor_data(reading)


@router.get("/nodes", response_model=List[NodeStatus])
async def get_nodes():
    """Returns list of registered ESP32 mine monitoring nodes with spatial positions & risk status."""
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
    """Returns chronological sensor readings and risk evolution."""
    filtered = list(history_records)
    if node_id:
        filtered = [r for r in filtered if r["node_id"] == node_id]
    return filtered[-limit:]


@router.get("/alerts")
async def get_alerts(limit: int = Query(20, ge=1, le=100)):
    """Returns active risk alerts with root-cause explanations."""
    return list(alerts_list)[:limit]


@router.get("/model-status")
async def get_model_status():
    """Returns loaded models, metadata, feature list, and accuracy benchmarks."""
    meta_path = app_config.get("paths.models_dir", "mine-ai/models") + "/model_metadata.json"
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
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": pipeline.classifier is not None and pipeline.anomaly_detector is not None,
        "active_nodes": len(fleet_registry),
        "buffered_history_count": len(history_records)
    }
