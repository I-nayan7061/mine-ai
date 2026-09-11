# AI-Enabled Real-Time Mine Subsidence Monitoring, Prediction & Early Warning System
**SIH 2026 Problem Statement ID:** SIH26025  
**Theme:** Disaster Management | **Category:** Hardware & AI/IoT  
**Team:** NexGen

---

## 1. Project Overview
Underground coal mining extraction alters in-situ geotechnical stresses, creating strata convergence, roof bed separation, pillar spalling, and sudden subsidence. Conventional monitoring relies on periodic visual inspections or expensive proprietary instrumentation, leaving active galleries vulnerable to undetected failures.

This project delivers a **low-cost, real-time, AI-enabled monitoring, prediction, and early warning system** sensing ground movement across multiple underground sensor nodes via:
1. **Vibration** (MPU6050 Accelerometer magnitude, RMS, and spectral FFT features)
2. **Dual-Axis Tilt** (MPU6050 Inclinometer $\theta_x, \theta_y$, tilt velocity, and trend slopes)
3. **Displacement** (VL53L1X Time-of-Flight relative strata convergence, velocity, and acceleration)

---

## 2. System Architecture & End-to-End Flow

```text
┌──────────────────────────────────────────────────────────┐
│              UNDERGROUND COAL MINE GALLERY               │
│                                                          │
│  ESP32 Node 1      ESP32 Node 2      ESP32 Node 3 (Face) │
│  (Vib, Tilt, Disp) (Vib, Tilt, Disp) (Vib, Tilt, Disp)   │
└────────────────────────────┬─────────────────────────────┘
                             │ LoRa / Wi-Fi Telemetry
                             ↓
                 ┌───────────────────────┐
                 │     LORA GATEWAY      │
                 └───────────┬───────────┘
                             │ MQTT / HTTP REST
                             ↓
                 ┌───────────────────────┐
                 │    FastAPI BACKEND    │
                 └───────────┬───────────┘
                             │
     ┌───────────────────────┴────────────────────────┐
     ↓                                                ↓
┌─────────────────────────┐              ┌─────────────────────────┐
│     DATA VALIDATION     │              │     IN-MEMORY BUFFER    │
│  - Bounds & Stuck Sensor│              │  - Per-Node Ring Buffer │
└────────────┬────────────┘              └────────────┬────────────┘
             │                                        │
             └───────────────────┬────────────────────┘
                                 ↓
                 ┌───────────────────────────────┐
                 │  ROLLING WINDOW PREPROCESSING │
                 │  - Median & Butterworth Filter│
                 │  - Chronological 60s Windows  │
                 └───────────────┬───────────────┘
                                 ↓
                 ┌───────────────────────────────┐
                 │      FEATURE ENGINEERING      │
                 │  - Vibration (RMS, FFT, Peaks)│
                 │  - Tilt (Rates, Slope, Mag)   │
                 │  - Disp (Velocity, Accel)     │
                 │  - Spatial Gallery Graph      │
                 └───────────────┬───────────────┘
                                 ↓
                 ┌───────────────────────────────┐
                 │   FEATURE SELECTION (Top 30)  │
                 │   Pruned Collinearity (|r|<.95│
                 └───────────────┬───────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ↓                               ↓
 ┌───────────────────────────────┐ ┌───────────────────────────────┐
 │   ISOLATION FOREST (ANOMALY)  │ │   XGBOOST RISK CLASSIFIER     │
 │   - Unsupervised Outlier Score│ │   - 4-Tier Class Probabilities│
 └───────────────┬───────────────┘ └───────────────┬───────────────┘
                 │                                 │
                 └───────────────┬─────────────────┘
                                 ↓
                 ┌───────────────────────────────┐
                 │    MULTI-CRITERIA RISK ENGINE │
                 │  - Physical Severity Subscores│
                 │  - Anomaly + ML Fusion [0-100]│
                 │  - Spatial-Temporal Coupling  │
                 └───────────────┬───────────────┘
                                 ↓
                 ┌───────────────────────────────┐
                 │   EXPLAINABLE AI (SHAP XAI)   │
                 │  - "Why is Node at Risk?"     │
                 │  - Top Factor Impact Breakdown│
                 └───────────────┬───────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ↓                               ↓
 ┌───────────────────────────────┐ ┌───────────────────────────────┐
 │   REAL-TIME WEB DASHBOARD     │ │      EARLY WARNING ALERTS     │
 │  - 2D Mine Gallery Map        │ │  - Immediate Operator Warning │
 │  - Live Chart.js Telemetry    │ │  - SMS / MQTT / Webhook Hooks │
 └───────────────────────────────┘ └───────────────────────────────┘
```

---

## 3. Risk Level Definitions (Prototype & Experimental)
The continuous Risk Score ($0 - 100$) maps to four operational categories:
- 🟢 **NORMAL (0.0 – 24.9)**: Stable baseline strata conditions; ambient vibrations and minor sensor noise within baseline geotechnical tolerances.
- 🟡 **WARNING (25.0 – 49.9)**: Detectable low-velocity roof convergence ($\sim 0.04\text{ mm/min}$) or minor tilt drift without immediate seismic shock.
- 🟠 **HIGH RISK (50.0 – 74.9)**: Accelerated bed separation, steep trend slopes, elevated vibration RMS, and multi-sensor coupling.
- 🔴 **CRITICAL (75.0 – 100.0)**: Rapid strata displacement jump, severe tilt exceeding $3.5^\circ$, seismic fracturing bursts, and adjacent gallery nodes also exhibiting abnormal convergence.

> **Important Clarification**: These thresholds are prototype engineering calibration values and must be field-validated using geotechnical standards before operational mine safety deployment.

---

## 4. Multi-Model Benchmark Results

Evaluated on chronologically separated, completely unseen test cycle (`Cycle_04`, 809 windows):

| Model | Accuracy | Weighted F1 | High Recall | Critical Recall | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Recommended)** | **99.38%** | **0.9938** | **98.97%** | **100.00%** | **0.027 ms** |
| **LightGBM** | **99.01%** | **0.9901** | **99.48%** | **100.00%** | **0.015 ms** |
| **Random Forest (Baseline)** | **96.29%** | **0.9632** | **97.42%** | **100.00%** | **0.110 ms** |
| **Logistic Regression** | 96.17% | 0.9618 | 97.42% | 98.20% | 0.004 ms |
| **Decision Tree** | 92.83% | 0.9288 | 94.33% | 99.10% | 0.004 ms |

**Unsupervised Anomaly Concordance**: Isolation Forest achieved **92.71% test concordance** on anomalous test windows without requiring labelled ground collapse data.

---

## 5. Quickstart Guide

### 5.1 Installation
```bash
cd mine-ai
pip install -r requirements.txt
```

### 5.2 Run Automated Tests
```bash
python -m pytest tests/ -v
```

### 5.3 Launch FastAPI Backend & Real-Time Dashboard
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Open your web browser at:
👉 **`http://localhost:8000/`** (or `http://localhost:8000/dashboard`)

---

## 6. Live Hardware Integration (ESP32)
ESP32 nodes send HTTP POST requests or MQTT messages to `/api/sensor-data`:
```json
POST /api/sensor-data
Content-Type: application/json

{
  "node_id": "N01",
  "timestamp": "2026-09-10T14:30:00",
  "tilt_x": 0.12,
  "tilt_y": 0.08,
  "vibration": 0.035,
  "displacement_mm": 1.02
}
```

Response:
```json
{
  "success": true,
  "node_id": "N01",
  "timestamp": "2026-09-10T14:30:00",
  "anomaly": false,
  "anomaly_score": 0.051,
  "risk_score": 10.2,
  "risk_level": "NORMAL",
  "confidence": 0.98,
  "top_contributing_features": ["strata_equilibrium"],
  "human_explanations": ["Ground conditions are stable within baseline geotechnical tolerances."]
}
```
