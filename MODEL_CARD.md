# Model Card: Multimodal Mine Strata Subsidence Early Warning System
**SIH 2026 Problem Statement ID:** SIH26025  
**Theme:** Disaster Management | **Category:** Hardware & AI/IoT | **Team:** NexGen  
**Primary Pilot AOI:** Moonidih Underground Colliery, Jharia Coalfield, BCCL, Dhanbad, Jharkhand  

---

## 1. Model Details & Versioning
- **Model Version:** `multimodal_v4`
- **Training Data Version:** `multimodal_dataset_v4`
- **Feature Schema Version:** `selected_features_v4` (29 non-collinear indicators)
- **Primary Supervised Model:** XGBoost Multi-Class Classifier (`objective='multi:softprob'`, `max_depth=6`, `n_estimators=120`, `learning_rate=0.08`)
- **Alternative Candidates:** LightGBM (`lightgbm_model.pkl`), Random Forest (`random_forest.pkl`)
- **Unsupervised Anomaly Detector:** Isolation Forest (`isolation_forest.pkl`, fitted on normal baseline windows)
- **Explainability Engine:** TreeSHAP (native C++ tree attribution)
- **Output Classes:** `NORMAL`, `WARNING`, `HIGH`, `CRITICAL`

---

## 2. Intended Use & Operational Boundary
- **Intended Use:** Prototype decision-support early-warning research tool for visualizing subterranean gallery strata convergence trends alongside regional meteorological and satellite InSAR deformation context.
- **Explicit Exclusions (NOT Intended For):**
  - Autonomous triggering of underground mine evacuations without human supervisor authorization.
  - Certified statutory mine safety system (does NOT replace DGMS-mandated strata control and monitoring plans).
  - Exact minute-by-minute prediction of sudden rockbursts or geological roof falls.

---

## 3. Data Provenance & Realism Disclosures

| Modality | Data Type | Provenance / Source | Limitations |
| :--- | :--- | :--- | :--- |
| **Subterranean Sensors** | **Synthetic (Physics-Grounded)** | Simulated MPU6050 accelerometer/inclinometer and VL53L1X ToF laser convergence across 5 gallery nodes. | Modeled using differential beam-deflection equations; **not measured from real mine collapses**. |
| **Surface Meteorology** | **Observational Data** | India Meteorological Department (IMD) 0.25° gridded rainfall & 0.5° temperature; Open-Meteo IMD-calibrated grid. | Regional grid resolution (~27 km); local micro-topographic cloudbursts may differ. |
| **Satellite Radar (SAR)** | **Remote Sensing Data** | Copernicus Sentinel-1 C-Band InSAR & ISRO Bhoonidhi NISAR L2 GUNW interferometric products. | 6 to 12-day repeat revisit. Provides regional surface context, **NOT real-time 1.0 Hz underground monitoring**. |

---

## 4. Empirical Benchmark Performance (Test Shift `Cycle_04`, 809 Windows)

| Model Architecture | Overall Accuracy | Macro F1 | Weighted F1 | Balanced Accuracy | High Recall | Critical Recall | Model Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Active Primary)** | **98.35%** | **0.9838** | **0.9835** | **98.54%** | **98.45%** | **100.00%** | **0.8295 ms** |
| **LightGBM** | 98.15% | 0.9818 | 0.9815 | 98.30% | 98.45% | 100.00% | 0.7420 ms |
| **Random Forest** | 97.85% | 0.9782 | 0.9784 | 97.90% | 97.94% | 98.20% | 1.1250 ms |

> **Important Safety Note on 100% Critical Recall:**  
> The 100% Critical Recall was achieved on the controlled test shift `Cycle_04`. **This is a dataset-specific benchmark result and does NOT constitute a guarantee of zero false negatives under real-world mining conditions.**

---

## 5. Computational Latency Profile
- **Model-Only Latency:** p50 = `0.8295 ms`, p95 = `1.3742 ms`, p99 = `2.8152 ms`
- **End-to-End Pipeline Latency:** p50 = `46.53 ms`, p95 = `54.97 ms`, p99 = `56.55 ms`  
  *(Includes raw JSON ingestion, validation, rolling ring buffer, median filter, FFT spectral energy calculation, spatial graph aggregation, ML inference, multi-criteria risk engine, and TreeSHAP attribution)*.
- **External Data Latency:** Asynchronous / cached (0.00 ms on live sensor stream).

---

## 6. Known Limitations & Geotechnical Constraints
1. **Lack of Real Mine Roof Fall Sensor Logs:** Real-world subterranean catastrophic collapses are rare events; training on synthetic scenarios requires extensive in-situ field validation before operational trust.
2. **Domain Shift Across Coalfields:** A model calibrated for Moonidih Colliery (thick Barakar sandstone roof, 320m depth) cannot be applied to shallow or bord-and-pillar workings without site-specific geotechnical recalibration.
3. **Dust & Optical Attenuation:** High coal dust concentrations in active longwall panels may temporarily attenuate ToF laser beams; physical sanity interlocks detect and flag these optical dropouts.
4. **Intrinsically Safe Hardware Requirement:** ESP32 prototype hardware must be enclosed in DGMS-approved Flameproof / Intrinsically Safe (Ex-d / Ex-ia) enclosures before physical deployment in hazardous underground coal mine atmospheres.
