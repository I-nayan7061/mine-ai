# Model Validation & Benchmark Report (Clean & Leakage-Free)
**Project:** SIH 2026 | Problem Statement SIH26025 | Team NexGen  
**Dataset Version:** Clean Chronologically Partitioned Subsidence Dataset  
**Test Set:** `Cycle_04` (Completely unseen physical sequence: 809 windows, 10 sensor nodes)  
**Verification Date:** September 2026  
**Status:** Certified 100% Passed (34 / 34 Test Suite Pass)

---

## 1. Dataset Partitioning & Cycle Isolation

To guarantee realistic out-of-sample generalization without temporal or cycle contamination:
- **Total Windows:** 4,045 rolling windows (each 30 samples, 5-second slide step)
- **Train Set (`Cycle_01` + `Cycle_02`):** 2,427 windows (60.0%)
- **Validation Set (`Cycle_03`):** 809 windows (20.0%)
- **Test Set (`Cycle_04`):** 809 windows (20.0% — completely unseen)

```
+------------------------------------+------------------+------------------+
| Train: Cycle_01 & Cycle_02         | Val: Cycle_03    | Test: Cycle_04   |
| 2,427 windows                      | 809 windows      | 809 windows      |
+------------------------------------+------------------+------------------+
  [Scalers & Selection Fitted Here]   [Zero Contamination] [Zero Contamination]
```

---

## 2. Test Set Benchmark Comparison (`Cycle_04`)

Evaluated on 809 unseen test windows with ground truth distribution:
- **NORMAL:** 279 windows
- **WARNING:** 225 windows
- **HIGH:** 194 windows
- **CRITICAL:** 111 windows

| Model | Accuracy | Weighted F1 | High Recall | Critical Recall | False Positives (Normal $\to$ Warn/High/Crit) | Model Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Recommended)** | **99.38%** | **0.9938** | **98.97%** | **100.00%** | **1 / 279 (0.36%)** | **0.027 ms** |
| **LightGBM** | 99.01% | 0.9901 | 99.48% | 100.00% | 0 / 279 (0.00%) | 0.015 ms |
| **Random Forest** | 96.29% | 0.9632 | 97.42% | 100.00% | 16 / 279 (5.73%) | 0.110 ms |
| **Logistic Regression** | 96.17% | 0.9618 | 97.42% | 98.20% | 9 / 279 (3.23%) | 0.004 ms |
| **Decision Tree** | 92.83% | 0.9288 | 94.33% | 99.10% | 24 / 279 (8.60%) | 0.004 ms |

### Unsupervised Anomaly Detection Concordance
* **Model:** Isolation Forest (`contamination=0.08`, fitted on training data only)
* **Test Concordance with Physical Ground Truth:** **92.71%**
* **Role in Pipeline:** Operates as an independent second-opinion safety interlock. If XGBoost predicts Normal but Isolation Forest detects high novelty/anomaly, a warning flag is attached to the API response.

---

## 3. Confusion Matrices (`Cycle_04` Test Set)

### XGBoost (Accuracy: 99.38%)
```
Predicted --->   NORMAL   WARNING   HIGH   CRITICAL   |  Recall
Actual NORMAL      278         1      0          0   |   99.64%
Actual WARNING       2       223      0          0   |   99.11%
Actual HIGH          0         2    192          0   |   98.97%
Actual CRITICAL      0         0      0        111   |  100.00%
```
*Note: Critical Recall is 100.00% (111 / 111). Zero false negatives on catastrophic collapse.*

### LightGBM (Accuracy: 99.01%)
```
Predicted --->   NORMAL   WARNING   HIGH   CRITICAL   |  Recall
Actual NORMAL      279         0      0          0   |  100.00%
Actual WARNING       7       218      0          0   |   96.89%
Actual HIGH          0         1    193          0   |   99.48%
Actual CRITICAL      0         0      0        111   |  100.00%
```

### Random Forest (Accuracy: 96.29%)
```
Predicted --->   NORMAL   WARNING   HIGH   CRITICAL   |  Recall
Actual NORMAL      263        16      0          0   |   94.27%
Actual WARNING       2       216      7          0   |   96.00%
Actual HIGH          0         5    189          0   |   97.42%
Actual CRITICAL      0         0      0        111   |  100.00%
```

---

## 4. Latency Profiling Breakdown

Measured on standard CPU runtime (Python 3.14 on Windows, single-threaded execution):

| Pipeline Stage | Operations Executed | Execution Time (ms) | % of Total |
| :--- | :--- | :---: | :---: |
| **Ingestion & Buffering** | Pydantic validation, FIFO deque insert | 0.42 ms | 0.8% |
| **Preprocessing & Filtering** | 3-point rolling median filter | 1.15 ms | 2.3% |
| **Feature Extraction** | 30 features (FFT, linregress, gradients, spatial mesh) | 41.20 ms | 81.5% |
| **Model Inference** | Standardized scaling + XGBoost `predict_proba` | 0.03 ms | 0.1% |
| **Explainability (TreeSHAP)** | TreeSHAP exact computation (top 3 features) | 7.60 ms | 15.0% |
| **Risk Scoring & Interlocks** | Multi-factor physics interlock & hysteresis check | 0.16 ms | 0.3% |
| **Total End-to-End Latency** | Full `/api/sensor-data` processing cycle | **50.56 ms** | **100.0%** |

*Throughput Capacity:* **~20 requests/second per CPU core**. Easily capable of handling a 10-node mesh network operating at 1.0 Hz telemetry reporting.

---

## 5. Geotechnical Physical Scenario Validation

All 5 core mining failure and operational scenarios were verified via automated end-to-end integration tests:

1. **Scenario A (Stable Baseline):**
   - *Parameters:* Displacement $< 0.5\text{ mm}$, Tilt $< 0.3^\circ$, Vibration $< 0.05\text{ g}$.
   - *Result:* 100% `NORMAL`, Risk Index $< 15.0$.
2. **Scenario B (Slow Roof Sag / Bed Separation):**
   - *Parameters:* Steady displacement creep ($0.05\text{ mm/s}$), low tilt, negligible vibration.
   - *Result:* Correctly triggers `WARNING` at $2.0\text{ mm}$, with SHAP attributing primary weight to `disp_trend_r2` and `disp_rate_max`.
3. **Scenario C (Accelerated Convergence / Pillar Yielding):**
   - *Parameters:* Multi-node displacement rate acceleration ($0.3\text{ mm/s}$), angular tilt $> 2.0^\circ$.
   - *Result:* Escalates to `HIGH` risk, spatial mesh neighbor escalation features active.
4. **Scenario D (Dynamic Roof Fracture & Imminent Collapse):**
   - *Parameters:* Explosive displacement rate ($> 1.0\text{ mm/s}$), tilt spike ($> 5^\circ$), micro-seismic acoustic burst ($> 0.8\text{ g}$).
   - *Result:* **100% immediate trigger of `CRITICAL` alert**, Risk Index $= 95\text{--}100$, emergency evacuation recommendation generated.
5. **Scenario E (Blasting Shockwave / Haulage Noise Resistance):**
   - *Parameters:* Sudden high vibration spike ($1.8\text{ g}$) lasting 1.5 seconds, but **zero sustained roof displacement** and zero tilt change.
   - *Result:* **Physics Interlock activates.** Model caps risk score at $28.0$ (`WARNING` / Operational Noise). **Zero false alarms for catastrophic collapse.**

---

## 6. Brutal Stress & Edge Case Test Results

A dedicated brutal stress suite (`tests/test_brutal_stress.py`) validated edge cases:
- `test_blasting_shock_false_positive_immunity`: Passed (Caps risk at 28.0 under 2.5g blast).
- `test_nan_and_inf_injection`: Passed (Linear interpolation sanitizes corrupt bytes).
- `test_stuck_dead_sensor_behavior`: Passed (Zero-variance freeze detected without crashing).
- `test_high_frequency_noise_jitter`: Passed (95% Normal stability under continuous white noise).
- `test_monotonic_creep_progression`: Passed (Risk strictly increases as displacement advances).
- `test_boundary_threshold_determinism`: Passed (Hysteresis prevents flickering).
- `test_high_throughput_burst`: Passed (100 sequential packets handled in $< 2.0$ seconds).
- `test_spatial_correlation_amplification`: Passed (Multi-node failure amplifies risk score).

---

## 7. Operational Disclaimer & Safety Boundaries

1. **Synthetic Data Disclaimer:** The training and evaluation dataset was generated using geomechanical simulation models adhering to empirical underground coal mining rock mechanics. It is designed for prototype evaluation and testing, not yet certified on live colliery production strata.
2. **Threshold Disclaimer:** The alert thresholds ($2.0\text{ mm}$, $5.0\text{ mm}$, etc.) represent prototype engineering calibration values and must be recalibrated with DGMS (Directorate General of Mines Safety) statutory requirements and mine-specific geotechnical strata control guidelines prior to commercial underground deployment.
3. **Early Warning Scope:** The system predicts **risk escalation levels and probability of instability**, not an exact millisecond timestamp of structural collapse.
