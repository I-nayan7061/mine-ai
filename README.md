# AI-Enabled Low Cost Real Time Mine Subsidence Monitoring, Prediction & Early Warning System
**SIH 2026 Problem Statement ID:** SIH26025  
**Theme:** Disaster Management | **Category:** Hardware & AI/IoT  
**Team:** NexGen  
**Primary Pilot Site:** Moonidih Underground Colliery, Jharia Coalfield, Bharat Coking Coal Limited (BCCL / Coal India Ltd.), Dhanbad, Jharkhand  

---

## 1. Project Overview & Operational Context
Underground coal extraction alters in-situ geotechnical stresses, creating strata convergence, roof bed separation, pillar spalling, and surface subsidence. Conventional monitoring relies on periodic visual inspections or expensive proprietary instrumentation, leaving active galleries vulnerable to undetected failures.

This project delivers a **low-cost, real-time, AI-enabled monitoring, prediction, and early warning system** sensing ground movement across multiple underground sensor nodes synchronized with regional meteorology and satellite radar remote sensing:
1. **Subterranean IoT Telemetry (1.0 Hz):**
   - **Vibration:** MPU6050 Accelerometer magnitude, RMS, Crest Factor, and spectral FFT energy bands.
   - **Dual-Axis Tilt:** MPU6050 Inclinometer $\theta_x, \theta_y$, tilt velocity, and trend slopes.
   - **Displacement:** VL53L1X Time-of-Flight relative strata convergence, velocity, and acceleration.
2. **Surface Meteorology (IMD / Open-Meteo):**
   - India Meteorological Department (IMD) 0.25° gridded rainfall (24h, 72h, 7d) and root-zone deep soil moisture percolation.
3. **Satellite Earth Observation (Copernicus / ISRO):**
   - Sentinel-1 C-Band SAR InSAR line-of-sight surface deformation velocity (-18.4 mm/yr) and ISRO Bhoonidhi NISAR L2 GUNW interferometric products.

---

## 2. End-to-End System Architecture

```text
┌──────────────────────────────────────────────────────────┐
│              UNDERGROUND COAL MINE GALLERY               │
│                                                          │
│  ESP32 Node 1      ESP32 Node 2      ESP32 Node 3 (Face) │
│  (Vib, Tilt, Disp) (Vib, Tilt, Disp) (Vib, Tilt, Disp)   │
└────────────────────────────┬─────────────────────────────┘
                             │ LoRa / Wi-Fi Telemetry
                             ▼
┌──────────────────────────────────────────────────────────┐
│       FASTAPI BACKEND GATEWAY (api/main.py)              │
│  - REST Ingestion & Bidirectional WebSockets             │
│  - Async SQLite WAL Persistence (data/mine_ai.db)        │
└────────────────────────────┬─────────────────────────────┘
                             │
       ┌─────────────────────┴─────────────────────┐
       ▼                                           ▼
┌───────────────────────────┐         ┌───────────────────────────┐
│     SIGNAL FILTERING      │         │   REGIONAL CONTEXT CACHE  │
│  - Bounds & Stuck Sensor  │         │  - IMD Gridded Rainfall   │
│  - Median & Butterworth   │         │  - Sentinel-1 InSAR LOS   │
│  - Chronological Windows  │         │  - ISRO Bhoonidhi NISAR   │
└──────────────┬────────────┘         └─────────────┬─────────────┘
               │                                    │
               └─────────────────┬──────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────┐
│      MULTI-MODAL FEATURE EXTRACTION (29 Features)        │
│  - Vibration (RMS, FFT Bands, Crest Factor, Entropy)     │
│  - Tilt (Magnitude, Velocity, Acceleration, Trend Slope) │
│  - Displacement (Velocity mm/min, Acceleration, Stability│
│  - Spatial Gallery Graph (Distance-Weighted Correlation) │
│  - Freshness Metadata (sensor_age_s, weather_age_h, sat_d│
└────────────────────────────┬─────────────────────────────┘
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│ ISOLATION FOREST ANOMALY  │ │   XGBOOST CLASSIFIER      │
│ - Unsupervised Outlier    │ │ - 4-Tier Class Probability│
└──────────────┬────────────┘ └─────────────┬─────────────┘
               │                            │
               └─────────────┬──────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│         MULTI-CRITERIA RISK ENGINE & INTERLOCKS          │
│  - Geotechnical Weighted Fusion [0 - 100 Score]          │
│  - Blasting / Machine Transient Noise Interlock (<= 28)  │
│  - Strata Physical Equilibrium Interlock (<= 24)         │
│  - Critical Confirmation Interlock                       │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│          EXPLAINABLE AI (C++ Native TreeSHAP)            │
│  - Sub-millisecond factor attribution                    │
│  - Plain-English geotechnical root causes                │
└────────────────────────────┬─────────────────────────────┘
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│  REAL-TIME WEB DASHBOARD  │ │   OPERATOR EARLY WARNING  │
│  - SVG Mine Gallery Map   │ │ - Web Audio API Siren     │
│  - Leaflet InSAR Satellite│ │ - Evacuation Banner       │
│  - Live Chart.js Charts   │ │ - SMS / MQTT / Webhooks   │
└───────────────────────────┘ └───────────────────────────┘
```

---

## 3. Risk Level Definitions & Physical Sanity Interlocks

The continuous Risk Score ($0 - 100$) maps to four operational categories:
- 🟢 **NORMAL (0.0 – 24.9)**: Strata stable within geotechnical baseline tolerances.
- 🟡 **WARNING (25.0 – 49.9)**: Detectable low-velocity roof convergence ($\sim 0.04\text{ mm/min}$) or minor tilt drift without immediate seismic shock.
- 🟠 **HIGH RISK (50.0 – 74.9)**: Accelerated bed separation, steep trend slopes, elevated vibration RMS, and multi-sensor coupling.
- 🔴 **CRITICAL (75.0 – 100.0)**: Rapid strata displacement jump, severe tilt exceeding $2.5^\circ$, seismic fracturing bursts, and adjacent gallery nodes also exhibiting abnormal convergence.

### Physical Safety Interlocks
1. **False-Positive Blasting / Haulage Interlock:** High vibration without physical displacement rate or tilt is classified as operational noise and capped at `28.0` (eliminating false alarms).
2. **Strata Equilibrium Interlock:** When local roof convergence is negligible ($< 0.25\text{ mm/min}$) and tilt is stable ($< 0.50^\circ$), regional weather or satellite baseline context cannot artificially elevate a stable node into `WARNING`.
3. **Critical Confirmation Interlock:** True `CRITICAL` risk ($R \ge 75.0$) requires physical confirmation (convergence velocity $\ge 0.50\text{ mm/min}$, severe tilt $\ge 2.5^\circ$, or active spatial cluster correlation).

---

## 4. Multi-Modal Ablation Study Results
Evaluated on chronologically separated, completely unseen operational shift (`Cycle_04`, 809 windows):

| Model Identifier | Modalities Included | Features | Accuracy | Macro F1 | Weighted F1 | High Recall | Critical Recall | Model Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Sensors Baseline)** | Subterranean Vibration, Tilt, Displacement, Spatial Graph | 29 | 98.25% | 0.9824 | 0.9825 | 98.45% | **100.00%** | 0.8120 ms |
| **Model B (Sensors + Weather)** | Sensors + IMD Gridded Rainfall, Soil Moisture, Pore Pressure | 29 | 98.35% | 0.9838 | 0.9835 | 98.45% | **100.00%** | 0.8250 ms |
| **Model C (Sensors + Satellite)** | Sensors + Sentinel-1 InSAR LOS Velocity, Gradient, Coherence | 29 | 98.35% | 0.9838 | 0.9835 | 98.45% | **100.00%** | 0.8290 ms |
| **Model D (Full Multimodal — Active)**| Sensors + IMD Weather + Sentinel-1 InSAR / NISAR Radar | 29 | **98.35%** | **0.9838** | **0.9835** | **98.45%** | **100.00%** | **0.8295 ms** |
| **Model E (Multimodal + Terrain)** | Multimodal + Overburden Depth (320m), Seam Thickness (4.2m) | 29 | 98.35% | 0.9838 | 0.9835 | 98.45% | **100.00%** | 0.8350 ms |

> **Research & Prototype Disclosure:**  
> The 100% Critical Recall was obtained on the controlled test shift `Cycle_04`. This is a prototype benchmark result and does NOT constitute a guarantee of zero false negatives under uncalibrated field conditions.

---

## 5. Computational Latency Profile
- **Model-Only Latency:** p50 = `0.8295 ms` | p95 = `1.3742 ms` | p99 = `2.8152 ms`
- **Full End-to-End Pipeline Latency:** p50 = `46.53 ms` | p95 = `54.97 ms` | p99 = `56.55 ms`  
  *(Includes JSON parsing, bounds check, rolling ring buffer, Butterworth filter, FFT spectral analysis, spatial graph aggregation, ML classification, multi-criteria risk engine, and TreeSHAP attribution)*.
- **Throughput:** Supports ~100 to 200 simultaneous 1.0 Hz ESP32 gallery nodes on a standard edge gateway.
- **External Ingestion:** Satellite and weather ingestion is asynchronous/cached (0.00 ms added to live sensor loop).

---

## 6. Quickstart Guide

### 6.1 Installation
```bash
cd mine-ai
pip install -r requirements.txt
```

### 6.2 Run Automated Tests (60 Tests Across 13 Test Suites)
```bash
python -m pytest tests/ -v
```

### 6.3 Launch FastAPI Backend & Real-Time Dashboard
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Or double-click `run_dashboard.bat` in the repository root.

Open your browser at:
- **Interactive SCADA Dashboard:** [http://localhost:8000/](http://localhost:8000/)
- **Swagger / OpenAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **System Health Status:** [http://localhost:8000/api/health](http://localhost:8000/api/health)
- **Data Provenance:** [http://localhost:8000/api/data-provenance](http://localhost:8000/api/data-provenance)

---

## 7. Live Telemetry API Specification
ESP32 microcontrollers send HTTP POST requests to `/api/sensor-data`:
```json
POST /api/sensor-data
Content-Type: application/json

{
  "node_id": "N03",
  "timestamp": "2026-09-11T14:30:00Z",
  "tilt_x": 0.42,
  "tilt_y": 0.28,
  "vibration": 0.082,
  "displacement_mm": 2.45
}
```

Response:
```json
{
  "success": true,
  "node_id": "N03",
  "timestamp": "2026-09-11T14:30:00Z",
  "anomaly": true,
  "anomaly_score": 0.421,
  "risk_score": 38.4,
  "risk_level": "WARNING",
  "confidence": 0.984,
  "top_contributing_features": ["disp_rate_max", "disp_trend_slope", "tilt_mag_current"],
  "human_explanations": [
    "Accelerated roof-to-floor convergence velocity (+0.42 mm/min)",
    "Upward linear roof sag trend detected"
  ],
  "neighbour_anomalies": 0
}
```

---

## 8. Data Provenance & Statutory Disclosures
- **Subterranean Sensors:** Physics-grounded synthetic telemetry modeling Jharia coalfield geotechnical parameters.
- **Surface Weather:** IMD Gridded Rainfall (0.25°) and Temperature (0.5°) / Open-Meteo IMD-calibrated grid.
- **Satellite InSAR:** Copernicus Sentinel-1 SAR & ISRO Bhoonidhi NISAR L2 GUNW regional deformation layer.
- **Statutory Notice:** This project is a student research prototype developed for Smart India Hackathon 2026 (SIH26025). Operational mine safety deployment requires formal statutory certification by DGMS and CIMFR.
