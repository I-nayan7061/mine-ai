# Dataset Audit & Provenance Report
**Project:** AI-Enabled Mine Subsidence Monitoring & Early Warning (SIH26025)  
**Team:** NexGen | **Pilot Context:** Moonidih Colliery, Jharia Coalfield, BCCL, Dhanbad, Jharkhand  
**Audit Date:** 2026-09-11

---

## 1. Dataset Provenance & Modalities

| Modality | Data Type | Source / Instrument | Frequency | Spatial Resolution |
| :--- | :--- | :--- | :---: | :---: |
| **Subterranean Sensors** | Synthetic (Physics-Grounded) | MPU6050 Accelerometer / Inclinometer, VL53L1X ToF Laser | 1.0 Hz | 5 Nodes across Gallery North & Crosscuts |
| **Surface Meteorology** | Observational Grid | India Meteorological Department (IMD) / Open-Meteo IMD-Calibrated | Hourly / Daily | 0.25° x 0.25° (Rainfall), 0.5° x 0.5° (Temp) |
| **Satellite Radar (SAR)** | Remote Sensing | Copernicus Sentinel-1 C-Band SAR / ISRO Bhoonidhi NISAR L2 GUNW | 6 - 12 days | 20m x 20m pixel coherence |
| **Satellite Optical** | Remote Sensing Context | Sentinel-2 MSI (NDVI/NDMI) & Landsat-9 TIRS (LST) | 5 - 16 days | 10m - 30m multispectral |

---

## 2. Dataset Records & Split Distribution
- **Raw Sensor Telemetry:** 4 complete operational shifts (Cycles 01 to 04), 8,000 seconds per node across 5 nodes = 40,000 records.
- **Preprocessed Rolling Windows (60s window, 10s step):**
  - Training Set (Cycles 01 & 02): 1,940 windows
  - Validation Set (Cycle 03): 970 windows
  - Test Set (Cycle 04): 809 windows (chronologically unseen)
  - Total Windows: 3,719 windows
- **Master Multimodal File:** `data/processed/multimodal_training_dataset.csv` (3,719 rows, 68 columns).

---

## 3. Strict Limitation Disclosure
- The subterranean sensor data in this prototype is physics-grounded synthetic time series modeling real differential strata convergence equations.
- **It is NOT field-validated on real mine roof collapse events.** Real mine deployment requires statutory in-situ calibration under DGMS Tech Circular standards.
