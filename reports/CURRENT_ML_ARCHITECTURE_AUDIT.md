# Current ML Architecture Audit
**Project:** AI-Enabled Mine Subsidence Monitoring & Early Warning (SIH26025)  
**Team:** NexGen | **Pilot Context:** Moonidih Colliery, Jharia Coalfield, BCCL, Dhanbad, Jharkhand  
**Audit Date:** 2026-09-11 | **Auditor Role:** Senior ML & Remote-Sensing Systems Auditor

---

## 1. System Overview & Entry Points
- **HTTP REST & WebSocket Gateway:** FastAPI application (`api/main.py`, `api/routes.py`) serving live prediction, fleet management, historical records, and bidirectional telemetry streaming.
- **In-Memory Buffer Layer:** Per-node rolling ring buffer (`NodeBufferManager`, capacity 300 readings) preserving temporal ordering.
- **Signal Preprocessing:** Bounds verification, stuck-sensor fault detection, moving median filtering, and Butterworth low-pass filtering.
- **Multi-Domain Feature Engineering:** Vibration kinematics & FFT spectrum, dual-axis tilt kinematics & regression, displacement convergence velocity & stability index, multi-node spatial gallery graph, India IMD meteorology, and Sentinel-1 InSAR radar deformation.
- **Dual-Model Inference:** Unsupervised `IsolationForest` outlier detector + Multiclass `XGBoost` (with `LightGBM` and `RandomForest` candidates).
- **Multi-Criteria Risk Fusion:** Weighted geotechnical scoring with physical interlocks for false alarm suppression.
- **Explainability:** Fast TreeSHAP factor impact attribution and human-readable natural language explanations.

---

## 2. Preprocessing & Windowing Verification
- **Window Size:** 60 seconds.
- **Step Stride:** 10 seconds.
- **Cycle-Isolated Windowing:** Windows are generated strictly within individual operational cycle boundaries (Cycle 1+2 Train, Cycle 3 Val, Cycle 4 Test).
- **Leakage Prevention:** Cross-split rolling window overlap has been eliminated, confirmed via automated testing (`test_no_cross_split_window_overlap`).

---

## 3. Machine Learning Models & Calibration
- **Model Types Evaluated:**
  - Primary Supervised: XGBoost (`xgboost_model.pkl`)
  - Gradient Boosted Candidate: LightGBM (`lightgbm_model.pkl`)
  - Interpretable Ensemble: Random Forest (`random_forest.pkl`)
  - Linear Baseline: Logistic Regression
  - Tree Baseline: Decision Tree
  - Unsupervised Anomaly Detector: Isolation Forest (`isolation_forest.pkl`)
- **Scaling:** Standalone `StandardScaler` (`scaler.pkl`) fitted strictly on training data.
- **Feature Selection:** Filtered using correlation threshold (|r| < 0.95), mutual information, and tree importance, fitted strictly on training data.

---

## 4. Multi-Modal Context Integration
- **Surface Weather Feed:** India Meteorological Department (IMD) 0.25° gridded rainfall and 0.5° temperature + Open-Meteo IMD-calibrated grid. Tracks cumulative precipitation (24h, 72h, 7d), deep root-zone soil saturation, and overburden hydrostatic pore-water pressure.
- **Satellite Earth Observation:** Copernicus Sentinel-1 C-Band SAR InSAR line-of-sight surface deformation time series and ISRO Bhoonidhi NISAR L2 GUNW interface. Tracks surface velocity (-18.4 mm/yr), coherence quality masks, and spatial deformation gradient.
- **Freshness Metadata:** Every record maintains `sensor_age_seconds`, `weather_age_hours`, `satellite_age_days`, `has_weather`, and `has_satellite` flags to respect varying observation temporal scales.
