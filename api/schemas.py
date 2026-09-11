"""Pydantic Request and Response Schemas for Mine AI API.
SIH26025 - NexGen | FastAPI Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SensorReadingRequest(BaseModel):
    """Payload format sent by ESP32 sensor nodes."""
    node_id: str = Field(..., examples=["N01"], description="Identifier of the reporting ESP32 node")
    timestamp: str = Field(..., examples=["2026-09-10T10:00:00"], description="ISO 8601 timestamp")
    tilt_x: float = Field(..., examples=[0.15], description="X-axis tilt angle in degrees")
    tilt_y: float = Field(..., examples=[0.08], description="Y-axis tilt angle in degrees")
    vibration: float = Field(..., examples=[0.04], description="Vibration magnitude / acceleration in g")
    displacement_mm: float = Field(..., examples=[1.02], description="Roof-to-floor displacement in mm")

    # Optional fields
    accel_x: Optional[float] = None
    accel_y: Optional[float] = None
    accel_z: Optional[float] = None
    gyro_x: Optional[float] = None
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None
    battery_voltage: Optional[float] = None
    signal_quality: Optional[float] = None

    # Multi-Modal Weather & Satellite Overrides (Optional, defaults to Jharia live feed if omitted)
    weather: Optional[Dict[str, float]] = None
    satellite: Optional[Dict[str, float]] = None


class RiskPredictionResponse(BaseModel):
    """Standardized response matching SIH master specification."""
    success: bool
    timestamp: str
    node_id: str
    sensor_values: Dict[str, float]
    weather_summary: Optional[Dict[str, float]] = None
    satellite_summary: Optional[Dict[str, float]] = None
    anomaly: bool
    anomaly_score: float
    risk_score: float
    risk_level: str
    class_probabilities: Dict[str, float]
    subscores: Optional[Dict[str, float]] = None
    top_contributing_features: List[str]
    human_explanations: List[str]
    neighbour_anomalies: int
    confidence: float


class NodeStatus(BaseModel):
    node_id: str
    gallery: str
    x: float
    y: float
    latest_risk_score: float
    latest_risk_level: str
    last_updated: Optional[str] = None
    online: bool = True


class AlertItem(BaseModel):
    id: str
    node_id: str
    timestamp: str
    severity: str
    risk_score: float
    reason: str
    top_factors: List[str]


class MineObservatoryData(BaseModel):
    """Complete multi-modal dump for Moonidih Colliery, Jharia Coalfield."""
    metadata: Dict[str, Any]
    coordinates: Dict[str, float]
    weather: Dict[str, Any]
    satellite: Dict[str, Any]
    geotech_indices: Dict[str, Any]
    recent_records_count: int
    sample_records: List[Dict[str, Any]]
