# API Documentation: Mine Subsidence Monitoring Backend
**SIH26025 - NexGen | FastAPI Specification**

---

## Base URL
`http://localhost:8000` (or host IP on network)  
Interactive Swagger Docs: `http://localhost:8000/docs`

---

## Endpoints

### 1. Ingest Sensor Reading
- **Endpoint**: `POST /api/sensor-data`
- **Description**: Main ingestion route called by ESP32 sensor nodes.
- **Request Body**:
  ```json
  {
    "node_id": "N01",
    "timestamp": "2026-09-10T14:30:00",
    "tilt_x": 0.15,
    "tilt_y": 0.08,
    "vibration": 0.04,
    "displacement_mm": 1.02
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "timestamp": "2026-09-10T14:30:00",
    "node_id": "N01",
    "sensor_values": {
      "vibration": 0.04,
      "tilt_x": 0.15,
      "tilt_y": 0.08,
      "displacement_mm": 1.02
    },
    "anomaly": false,
    "anomaly_score": 0.052,
    "risk_score": 11.4,
    "risk_level": "NORMAL",
    "class_probabilities": {
      "NORMAL": 0.96,
      "WARNING": 0.04,
      "HIGH": 0.0,
      "CRITICAL": 0.0
    },
    "top_contributing_features": ["strata_equilibrium"],
    "human_explanations": ["Ground conditions are stable within baseline geotechnical tolerances."],
    "neighbour_anomalies": 0,
    "confidence": 0.96
  }
  ```

---

### 2. Fleet Nodes Status
- **Endpoint**: `GET /api/nodes`
- **Description**: Returns all registered sensor nodes, spatial coordinates, online status, and latest risk level.

---

### 3. Fleet Risk Aggregation
- **Endpoint**: `GET /api/risk`
- **Description**: Returns max fleet risk, average risk, node severity counts, and identified risk clusters/zones.

---

### 4. Historical Telemetry
- **Endpoint**: `GET /api/history?node_id=N03&limit=50`
- **Description**: Returns time-series telemetry and risk evolution for plotting and trend analysis.

---

### 5. Active Alerts Feed
- **Endpoint**: `GET /api/alerts?limit=20`
- **Description**: Returns active Warning, High, and Critical alerts with SHAP root-cause diagnostic explanations.

---

### 6. System Health & Model Status
- **Endpoints**: `GET /api/health`, `GET /api/model-status`
- **Description**: Liveness probe and loaded ML model metadata.
