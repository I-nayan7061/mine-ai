# Pre-Modification Architecture Audit
**Project:** AI-Enabled Low Cost Real Time Mine Subsidence Monitoring, Prediction & Early Warning System (SIH26025)  
**Team:** NexGen | **Pilot Mine Context:** Moonidih Underground Colliery, Jharia Coalfield, Dhanbad, Jharkhand  
**Audit Date:** 2026-09-11 | **Status:** Baseline Audited & Verified

---

## 1. Executive Summary
This document establishes the frozen baseline of the existing codebase prior to the execution of the Master Engineering Prompt enhancements. All currently operational functionalities (FastAPI endpoints, WebSocket streaming, SCADA web dashboard, physical interlocks, and 49 automated pytest test suites) have been validated and will be strictly preserved.

---

## 2. Inventory of System Entry Points & Components

### 2.1 Entry Points
- **Batch Launcher:** 
un_dashboard.bat (launches uvicorn server at http://localhost:8000/)
- **FastAPI Core Application:** mine-ai/api/main.py
  - Lifespan initialization: SQLite DB initialization (src/db.py)
  - Route handlers: mine-ai/api/routes.py
  - Static Web UI mount: /static -> mine-ai/dashboard/
- **Training Orchestration:**
  - mine-ai/make_dataset.py (synthetic scenario generation & rolling window feature extraction)
  - mine-ai/src/train_pipeline.py (cycle-isolated training, feature selection, multi-model benchmark)
- **Automated Test Suite:** mine-ai/tests/ (12 test modules, 49 tests, all currently passing)

### 2.2 Model Artifacts Baseline (Archived in models/archive/)
- models/xgboost_model.pkl: Primary supervised classifier (4 classes: NORMAL, WARNING, HIGH, CRITICAL)
- models/lightgbm_model.pkl: Gradient boosted candidate model
- models/random_forest.pkl: Interpretable ensemble baseline
- models/isolation_forest.pkl: Unsupervised anomaly detector
- models/scaler.pkl: StandardScaler fit on feature subset
- models/feature_names.json: Top 30 features list
- models/model_metadata.json: Model version, training parameters, and benchmark scores

### 2.3 Existing Configuration
- mine-ai/config.yaml: Defines spatial graph topology (nodes N01-N05, coordinates, connections), risk engine weights, severity scale parameters, thresholds, and buffer capacities.

### 2.4 Existing API Endpoints
| HTTP Method | Route | Description |
| :--- | :--- | :--- |
| GET | / and /dashboard | Serves the interactive SCADA web dashboard (index.html) |
| GET | /docs | OpenAPI / Swagger interactive API documentation |
| POST | /api/sensor-data | Telemetry ingestion called by ESP32 microcontrollers / LoRa gateway |
| POST | /api/predict | Direct inference endpoint for on-demand predictions |
| GET | /api/nodes | Registry of 5 mine monitoring nodes with spatial coordinates & status |
| GET | /api/risk | Fleet-level aggregated risk summary and identified risk zones |
| GET | /api/history | Chronological sensor readings and risk evolution |
| GET | /api/alerts | Active risk alerts with root-cause explanations |
| GET | /api/model-status | Model status, loaded algorithms, and feature schema |
| GET | /api/health | System health check (models loaded, nodes online, DB status) |
| GET | /api/mine/jharia-moonidih| Geotechnical dossier for Moonidih Colliery |
| GET | /api/mine/export-csv | CSV download of multimodal parameters |
| WS | /api/ws/telemetry | Bi-directional WebSocket stream for live dashboard broadcasting |

### 2.5 Dashboard Dependencies
- Frontend stack: Pure vanilla JavaScript (dashboard/app.js), CSS (dashboard/styles.css), and HTML5 (dashboard/index.html).
- External CDN dependencies:
  - Chart.js 4.4.1 (live telemetry charts)
  - Leaflet 1.9.4 (satellite GIS map)
- Hardware Audio: Web Audio API synthesized emergency siren alarm for critical events.

---

## 3. Pre-Modification Verification Status
- Pytest execution command: python -m pytest tests/ -v
- Result: **49 passed, 0 failed, 0 errors** (Execution time: 43.54s).
- Persistence layer: SQLite database mine-ai/data/mine_ai.db in WAL mode operational.
- All existing model artifacts verified and backed up to mine-ai/models/archive/.

---

## 4. Preservation Invariant
In accordance with Rule 1 of the Master Engineering Prompt:
1. No existing API route or schema contract will be broken.
2. The interactive 3-tab dashboard will remain functional with all existing controls (scenario buttons, Leaflet layers, Chart.js graphs).
3. The existing 49 tests will continue to pass alongside newly developed tests for multimodal alignment, leakage verification, and fault isolation.
