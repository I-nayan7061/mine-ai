# Dataset Guide & Geotechnical Calibration
**SIH26025 - NexGen | Mine Subsidence Monitoring System**

---

## 1. Multi-Source Data Strategy
Per the project requirements, the system supports three distinct tiers of data:
1. **Real In-Situ Project Data**: Real telemetry streamed from physical ESP32 nodes deployed in experimental test setups.
2. **External / Academic Research Benchmarks**: Geotechnical subsidence and microseismic datasets from mining repositories.
3. **Physics-Grounded Synthetic Simulation Data**: Explicitly labeled `DATA_TYPE = SYNTHETIC`, generated via geomechanical strata deformation models.

---

## 2. Benchmark Synthetic Dataset Specification
- **File**: `data/synthetic/synthetic_mine_subsidence_dataset.csv`
- **Metadata**: `data/synthetic/synthetic_mine_subsidence_metadata.json`
- **Total Records**: 40,000 samples across 4 complete operational shifts/cycles.
- **Sampling Frequency**: Nominal 1.0 Hz (1 sample per second per node).
- **Nodes Included**: 5 ESP32 nodes (`N01`, `N02`, `N03`, `N04`, `N05`) in a simulated coal seam gallery.

### Class Distribution
| Risk Class | Sample Count | Percentage | Geotechnical Description |
| :--- | :---: | :---: | :--- |
| **NORMAL** | 18,000 | 45.0% | Steady-state quiescent strata, ambient micro-vibrations |
| **WARNING** | 9,200 | 23.0% | Gradual bed separation creep ($\sim 0.04\text{ mm/min}$) |
| **HIGH** | 8,000 | 20.0% | Accelerated convergence, noticeable bolt tensioning |
| **CRITICAL**| 4,800 | 12.0% | Dynamic tensile fracture, rapid roof deflection jump |

---

## 3. Sensor Field Specifications & Units

| Column | Data Type | Units | Physical Bounds | Primary Sensor |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | ISO 8601 string | UTC Time | Monotonic | ESP32 RTC / Gateway |
| `node_id` | String (`N01` - `N05`) | Identifier | Categorical | Node Firmware Config |
| `tilt_x` | Float | Degrees (°) | $[-90.0, +90.0]$ | MPU6050 Accelerometer |
| `tilt_y` | Float | Degrees (°) | $[-90.0, +90.0]$ | MPU6050 Accelerometer |
| `vibration`| Float | Acceleration ($g$) | $[0.0, 10.0]$ | MPU6050 Accelerometer |
| `displacement_mm`| Float | Millimeters ($mm$) | $[-100.0, 2000.0]$| VL53L1X Time-of-Flight |

---

## 4. Time-Series Splitting Strategy
To strictly prevent forward-looking data leakage, records are segmented chronologically across distinct operational shifts:
- **Training Set (60%)**: Cycles 1 & 2 (Earlier time period)
- **Validation Set (20%)**: Cycle 3 (Middle time period)
- **Test Set (20%)**: Cycle 4 (Latest unseen time period)

Adjacent rolling windows from the test set are completely isolated from training data.

---

## 5. Processed Feature-Engineered Dataset (`data/processed/`)
For direct machine learning and tabular model training without requiring real-time signal recomputation, the raw telemetry is windowed (60-second window, 10-second step) and transformed into 30 selected multi-domain features:

- **`featured_training_dataset.csv`**: All 4,045 window samples with 30 selected features and ground truth labels.
- **`train_features.csv`**: 2,427 chronological training samples (60%).
- **`val_features.csv`**: 809 validation samples (20%).
- **`test_features.csv`**: 809 testing samples (20%).
- **`processed_metadata.json`**: Feature column definitions, units, and class distributions.

### Selected 30 Geotechnical Features
1. **Spatial Deformation**: `spatial_disp_gradient_max`, `spatial_avg_neighbor_disp`, `spatial_avg_neighbor_risk`, `spatial_num_abnormal_neighbors`, `spatial_pct_abnormal_neighbors`
2. **Tilt Inclinometer**: `tilt_x_current`, `tilt_mag_std`, `tilt_trend_r2`, `tilt_accel_max`, `tilt_stability_index`, `tilt_rate_max`
3. **Convergence Displacement**: `disp_trend_r2`, `disp_accel_max`, `disp_rate_max`, `disp_stability_index`
4. **Seismic Vibration**: `vib_mean`, `vib_variance`, `vib_std`, `vib_min`, `vib_cov`, `vib_skewness`, `vib_crest_factor`, `vib_kurtosis`, `vib_spectral_energy_mid`, `vib_dominant_frequency`, `vib_spectral_centroid`, `vib_spectral_energy_low`, `vib_spectral_energy_high`, `vib_spectral_entropy`, `vib_second_dominant_frequency`

---

## 6. How to Generate & Train

### Generate Dataset
```bash
# Generate raw telemetry (40,000 samples) and processed feature matrices (4,045 windows)
python make_dataset.py

# Custom generation (e.g. 6 cycles, with immediate model training)
python make_dataset.py --cycles 6 --train
```

### Train Models Directly
```bash
# Execute end-to-end multi-model benchmarking (Isolation Forest, RF, XGBoost, LightGBM)
python -m src.train_pipeline
```

### Quick Data Loading in Python
```python
import pandas as pd

# Load raw telemetry stream
df_raw = pd.read_csv("data/synthetic/synthetic_mine_subsidence_dataset.csv")

# Load preprocessed training split directly for model fitting
train_df = pd.read_csv("data/processed/train_features.csv")
X_train = train_df.drop(columns=["risk_label", "timestamp"])
y_train = train_df["risk_label"]
```
