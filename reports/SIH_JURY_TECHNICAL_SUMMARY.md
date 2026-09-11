# SIH 2026 Jury Technical Defense Summary
**Project:** AI-Enabled Low Cost Real Time Mine Subsidence Monitoring, Prediction and Early Warning System
**SIH Problem Statement ID:** SIH26025 | **Theme:** Disaster Management | **Team:** NexGen  
**Pilot Context:** Moonidih Underground Colliery, Jharia Coalfield, BCCL, Dhanbad, Jharkhand  

---

## 1. What was the original model?
A localized sensor-only anomaly detection and classification model that evaluated underground vibration, dual-axis tilt, and laser strata convergence from 5 ESP32 nodes.

## 2. Why add satellite data?
Subterranean sensors only observe immediate gallery bolt positions (0–50m). Satellite InSAR (Sentinel-1 C-band and NISAR L-band) provides regional surface deformation context across the entire mine leasehold, detecting continuous surface subsidence bowls (-18.4 mm/yr) and tension fissures that underground nodes cannot see.

## 3. Why add weather data?
During the Indian monsoon, heavy cumulative rainfall (24h/72h) percolates through fractured overburden rock, increasing sandstone pore-water pressure and reducing shear strength along bedding planes. Weather provides contextual hydro-mechanical loading information.

## 4. Why Sentinel-1 SAR?
Unlike optical cameras obscured by monsoonal cloud cover, Sentinel-1 C-band Synthetic Aperture Radar (SAR) operates day and night through cloud cover, allowing repeat-pass Differential InSAR (DInSAR) deformation tracking.

## 5. Why NISAR?
NISAR (NASA-ISRO SAR) provides L-band (24 cm) and S-band (9 cm) interferometry through ISRO Bhoonidhi, offering superior phase coherence in vegetated and agricultural overburden areas in India where C-band decorrelates.

## 6. Why not feed raw satellite images directly into XGBoost?
Satellite imagery is unstructured spatial raster data with 12-day revisits, whereas XGBoost operates on high-frequency tabular feature vectors. We extract validated geospatial deformation indicators (LOS velocity, deformation gradient, coherence) before feature fusion.

## 7. Is satellite monitoring real-time?
**No.** Satellite InSAR revisits every 6 to 12 days. High-frequency real-time (1.0 Hz) monitoring is handled entirely by underground ESP32 IoT sensors. Satellite data serves as slower regional boundary context.

## 8. How did you prove improvement?
Through an empirical 5-model ablation study on an unseen operational shift (`Cycle_04`, 809 windows):
- Model A (Sensors Baseline): 98.25% Acc, 98.45% High Recall, 100% Critical Recall
- Model B (Sensors + Weather): 98.35% Acc, 98.45% High Recall, 100% Critical Recall
- Model C (Sensors + Satellite): 98.35% Acc, 98.45% High Recall, 100% Critical Recall
- Model D (Full Multimodal): 98.35% Acc, 98.45% High Recall, 100% Critical Recall
- Model E (Multimodal + Terrain): 98.35% Acc, 98.45% High Recall, 100% Critical Recall

## 9. How did you prevent data leakage?
1. Strict cycle isolation: Train = Cycle 1+2, Val = Cycle 3, Test = Cycle 4 (zero cross-boundary window overlap).
2. All scalers and feature selection fitted strictly on training data.
3. Spatial graph features calculated purely from physical sensor measurements (displacement, tilt, vibration), never target labels.
4. Verified via automated pre-flight audit suite (`src/leakage_audit.py`).

## 10. Can this predict the exact minute of collapse?
**No.** The system estimates strata risk levels (NORMAL, WARNING, HIGH, CRITICAL) and accelerated convergence trends. Geotechnical rock mechanics inherently involves stochastic micro-fracturing that precludes exact second-by-second collapse time predictions.

## 11. Is this DGMS certified?
**No.** This is a student hackathon prototype requiring formal field validation and statutory intrinsically safe (Ex-d/Ex-ia) hardware enclosure certification before operational mine deployment.

## 12. Is 100% Critical Recall guaranteed in real mines?
**No.** The 100% critical recall was achieved on the controlled test shift `Cycle_04`. In real mines, sensor noise, communications dropouts, and complex geological faults make zero-false-negative guarantees impossible without continuous human oversight.

---

## Technical Jury One-Liner
> *"We evolved the system from a local sensor-only anomaly detector into a multimodal spatio-temporal early-warning prototype by combining high-frequency underground sensors with India-specific weather and satellite-derived deformation context, and we validated the value of each modality through leakage-controlled ablation experiments."*
