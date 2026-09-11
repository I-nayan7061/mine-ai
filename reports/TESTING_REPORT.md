# Automated Quality Assurance & Testing Report
**Project:** AI-Enabled Mine Subsidence Monitoring & Early Warning (SIH26025)  
**Team:** NexGen | **Testing Framework:** Pytest 8.4.2 / Asyncio  
**Execution Date:** 2026-09-11 | **Status:** 100% Passing (60 / 60 Tests)

---

## 1. Test Suite Summary

| Test Suite | File | Tests | Status | Scope |
| :--- | :--- | :---: | :---: | :--- |
| **API Endpoints** | `test_api.py` | 5 | PASSED | Health check, node fleet registry, aggregated risk, JSON ingestion |
| **Brutal Stress Tests** | `test_brutal_stress.py` | 8 | PASSED | Blasting shock immunity, NaN/Inf injection, stuck sensors, white noise jitter |
| **Emergency Bypass** | `test_emergency_bypass.py` | 5 | PASSED | Instantaneous displacement / tilt threshold bypass during cold-start |
| **Explainability (XAI)**| `test_fast_xai.py` | 3 | PASSED | Fast TreeSHAP latency (< 10 ms), human-readable geomechanical explanations |
| **Feature Extraction** | `test_features.py` | 4 | PASSED | Vibration FFT bands, dual-axis tilt kinematics, laser displacement velocity |
| **Model Architectures**| `test_models.py` | 4 | PASSED | Multiclass classification accuracy, model-only latency, isolation forest |
| **Observatory API** | `test_observatory_api.py` | 3 | PASSED | Moonidih Colliery dossier, CSV export, multimodal prediction payload |
| **Data Persistence** | `test_persistence.py` | 4 | PASSED | SQLite database initialization, WAL mode, reading & alert persistence |
| **Preprocessing** | `test_preprocessing.py` | 4 | PASSED | Data validation bounds, rolling window segmentation, stuck sensor check |
| **Risk Engine** | `test_risk_engine.py` | 4 | PASSED | Multi-criteria formula, safety interlocks, monotonicity under creep |
| **Mine Scenarios** | `test_scenarios.py` | 5 | PASSED | Normal, Slow Sag, Bed Separation, Dynamic Fracture, Blasting Noise |
| **Security & Auth** | `test_security.py` | 2 | PASSED | API Key verification (`X-API-Key`), CORS headers |
| **WebSocket Stream** | `test_websocket.py` | 2 | PASSED | Client connection, initial snapshot, real-time telemetry broadcast |
| **Multimodal & Leakage**| `test_multimodal_leakage_and_alignment.py` | 11 | PASSED | Target leakage check, window overlap, IMD weather, InSAR fallback |
| **TOTAL** | **13 Suites** | **60 Tests** | **ALL PASSED** | **Execution Time: 38.35 seconds** |
