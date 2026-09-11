# Comprehensive Data Leakage Audit Report
**Project:** AI-Enabled Mine Subsidence Monitoring & Early Warning (SIH26025)  
**Team:** NexGen | **Pilot Context:** Moonidih Colliery, Jharia Coalfield  
**Audit Date:** 2026-09-11 | **Compliance Standard:** Zero Tolerance for Target & Temporal Contamination

---

## 1. Executive Summary & Leakage Status
A comprehensive audit across all datasets, feature extractors, spatial graph calculations, preprocessing pipelines, and model training routines was executed.

| Leakage Dimension | Initial Status | Remediation Applied | Final Audit Result |
| :--- | :---: | :--- | :---: |
| **Target Leakage** | VULNERABLE | Removed residual target-derived feature `spatial_avg_neighbor_risk` | **PASS (Clean)** |
| **Temporal Window Overlap** | VULNERABLE | Isolated rolling-window segmentation per cycle split | **PASS (Clean)** |
| **Preprocessing Contamination** | COMPLIANT | Verified `StandardScaler` fitted strictly on training split only | **PASS (Clean)** |
| **Feature Selection Leakage** | COMPLIANT | Verified `FeatureSelector` fitted strictly on training split only | **PASS (Clean)** |
| **Spatial Graph Leakage** | COMPLIANT | Computed solely from physical sensors (displacement, tilt, vibration) | **PASS (Clean)** |
| **Future Satellite/Weather** | COMPLIANT | Freshness metadata & timestamp causality verified | **PASS (Clean)** |

---

## 2. Detailed Findings & Corrections

### Issue 1: Target Leakage via Residual Schema Feature
- **Location:** `models/feature_names.json`, line 9 (`spatial_avg_neighbor_risk`).
- **Why Leakage:** An earlier prototype implementation had computed average neighbor risk using ground-truth target labels. Although `src/spatial_features.py` was refactored to compute physical anomaly thresholds, the feature name persisted in the saved JSON schema.
- **Correction:** Replaced with purely physical metric `spatial_avg_neighbor_vib`. Verified via automated regex check in `src/leakage_audit.py`.

### Issue 2: Cross-Split Rolling Window Overlap in `make_dataset.py`
- **Location:** `make_dataset.py`, lines 112-128.
- **Why Leakage:** Rolling windows (60s window, 10s stride) were generated across the concatenated multi-cycle time series before applying chronological splitting. This allowed windows near the train/val and val/test boundaries to share up to 50 seconds of identical raw sensor records.
- **Correction:** Replaced with strict cycle-isolated splitting (Train = Cycle 1+2, Val = Cycle 3, Test = Cycle 4). Each cycle split is windowed independently. No window shares raw timestamps across splits. Verified via `test_no_cross_split_window_overlap`.

### Issue 3: Preprocessing & Scaler Verification
- **Verification:** `StandardScaler` and `FeatureSelector` were checked to confirm they are never fitted on validation or test sets. `models/preprocessing_metadata.json` and `models/feature_selection_metadata.json` explicitly log `scaler_fit_on_train_only = true` and `fit_strictly_on_train = true`.
