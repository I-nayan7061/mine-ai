# Teammate Integration Guide: Connecting Dashboard & Hardware to AI/ML Model
**SIH 2026 | Problem Statement: SIH26025 | Team: NexGen**

---

## 1. Quick Overview
The AI/ML model runs as a high-performance HTTP microservice.  
Whenever your hardware or dashboard receives new sensor readings, send an **HTTP POST request** to the prediction URL, and the AI will return the **Risk Level**, **Risk Score (0-100)**, **Confidence**, and **SHAP root-cause explanations**.

---

## 2. API Endpoint URL

- **If running on the same laptop as the dashboard**:
  ```text
  POST http://localhost:8000/api/predict
  ```
- **If running on different laptops over Wi-Fi / LAN**:
  ```text
  POST http://<AI_TEAMMATE_IP>:8000/api/predict
  ```

---

## 3. What You Send (Input JSON)
Send this JSON payload with header `Content-Type: application/json`:

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

### Field Definitions:
- `node_id` (string): Sensor node ID (e.g. `"N01"`, `"N02"`, `"N03"`).
- `timestamp` (string): ISO 8601 timestamp string.
- `tilt_x` (float): X-axis tilt in degrees from MPU6050.
- `tilt_y` (float): Y-axis tilt in degrees from MPU6050.
- `vibration` (float): Acceleration magnitude in $g$ from MPU6050.
- `displacement_mm` (float): Current relative distance/convergence in $mm$ from VL53L1X.

---

## 4. What the AI Model Returns (Output JSON)

```json
{
  "success": true,
  "node_id": "N01",
  "timestamp": "2026-09-10T14:30:00",
  "sensor_values": {
    "vibration": 0.04,
    "tilt_x": 0.15,
    "tilt_y": 0.08,
    "displacement_mm": 1.02
  },
  "risk_level": "NORMAL",
  "risk_score": 11.4,
  "anomaly": false,
  "anomaly_score": 0.052,
  "confidence": 0.98,
  "class_probabilities": {
    "NORMAL": 0.98,
    "WARNING": 0.02,
    "HIGH": 0.0,
    "CRITICAL": 0.0
  },
  "subscores": {
    "vibration_severity": 0.0,
    "tilt_severity": 0.0,
    "displacement_severity": 0.0,
    "anomaly_score": 5.2,
    "temporal_trend": 0.0,
    "spatial_correlation": 0.0
  },
  "top_contributing_features": [
    "strata_equilibrium"
  ],
  "human_explanations": [
    "Ground conditions are stable within baseline geotechnical tolerances."
  ],
  "neighbour_anomalies": 0
}
```

---

## 5. How to Display This on Your Dashboard

| Response Field | What to show on your UI | Recommended UI Component |
| :--- | :--- | :--- |
| `risk_level` | `NORMAL` $\rightarrow$ 🟢 Green<br>`WARNING` $\rightarrow$ 🟡 Yellow<br>`HIGH` $\rightarrow$ 🟠 Orange<br>`CRITICAL` $\rightarrow$ 🔴 Red | Status Badge / Node Color on Mine Map |
| `risk_score` | Continuous value from `0.0` to `100.0` | Gauge Meter / Progress Bar |
| `confidence` | Model confidence e.g. `98.0%` | Small text pill next to status |
| `human_explanations` | List of natural language diagnostic reasons | Modal popup: *"Why is this Node at Risk?"* |
| `neighbour_anomalies` | Number of adjacent abnormal sensors | Spatial cluster badge |

---

## 6. Ready-to-Use Code Snippets

### JavaScript / Fetch (for Web Dashboard):
```javascript
async function sendSensorTelemetry(sensorData) {
  const response = await fetch("http://localhost:8000/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(sensorData)
  });
  const aiResult = await response.json();
  console.log("Risk Level:", aiResult.risk_level);
  console.log("Risk Score:", aiResult.risk_score);
  return aiResult;
}
```

### Python (for Gateway / Backend):
```python
import requests

payload = {
    "node_id": "N01",
    "timestamp": "2026-09-10T14:30:00",
    "tilt_x": 0.15,
    "tilt_y": 0.08,
    "vibration": 0.04,
    "displacement_mm": 1.02
}

response = requests.post("http://localhost:8000/api/predict", json=payload)
ai_result = response.json()
print("Risk Level:", ai_result["risk_level"])
print("Risk Score:", ai_result["risk_score"])
print("Explanation:", ai_result["human_explanations"])
```

### ESP32 C++ (HTTPClient):
```cpp
#include <WiFi.h>
#include <HTTPClient.h>

void sendToAI(String nodeId, float tiltX, float tiltY, float vib, float disp) {
  HTTPClient http;
  http.begin("http://192.168.1.15:8000/api/predict"); // Replace with AI laptop IP
  http.addHeader("Content-Type", "application/json");

  String json = "{\"node_id\":\"" + nodeId + "\","
                "\"timestamp\":\"2026-09-10T14:30:00\","
                "\"tilt_x\":" + String(tiltX) + ","
                "\"tilt_y\":" + String(tiltY) + ","
                "\"vibration\":" + String(vib) + ","
                "\"displacement_mm\":" + String(disp) + "}";

  int httpCode = http.POST(json);
  if (httpCode > 0) {
    String payload = http.getString();
    Serial.println("AI Response: " + payload);
  }
  http.end();
}
```
