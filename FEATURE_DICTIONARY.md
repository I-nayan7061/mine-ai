# Mine Subsidence AI Feature Dictionary
**Project:** SIH 2026 | Problem Statement SIH26025 | Team NexGen  
**Total Selected Features:** 30  
**Feature Selection Method:** Mutual Information Maximization & Variance Thresholding on Training Set (`Cycle_01` + `Cycle_02`)  
**Data Leakage Status:** 100% Leakage-Free (Computed strictly from physical sensor observables)

---

## Overview Table

| # | Feature Name | Domain | Formula / Method | Units | Physical Interpretation | Live Compute |
|---|---|---|---|---|---|---|
| 1 | `spatial_disp_gradient_max` | Spatial | $\max_{j \in \mathcal{N}(i)} \frac{\|d_i - d_j\|}{\text{dist}(i, j)}$ | mm/m | Maximum differential roof sag rate across adjacent nodes | Circular buffer + mesh cache |
| 2 | `tilt_x_current` | Tilt | $\theta_{x}[-1]$ | deg ($^\circ$) | Instantaneous lateral tilt angle of mine roof / prop | Direct read |
| 3 | `vib_mean` | Vibration | $\frac{1}{N}\sum_{k=1}^N a_k$ | g | Baseline acceleration offset / constant load offset | 30-sample mean |
| 4 | `vib_variance` | Vibration | $\frac{1}{N}\sum_{k=1}^N (a_k - \mu)^2$ | $\text{g}^2$ | Micro-seismic mechanical energy spread | 30-sample variance |
| 5 | `vib_std` | Vibration | $\sqrt{\text{vib\_variance}}$ | g | Seismic signal standard deviation / dynamic energy | Standard deviation |
| 6 | `spatial_avg_neighbor_disp` | Spatial | $\frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} d_j$ | mm | Mean displacement of adjacent roof support pillars | Mesh spatial context |
| 7 | `spatial_avg_neighbor_risk` | Spatial | $\frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} \text{Elevated}_j$ | [0, 1] | Fraction of adjacent nodes with physically elevated sensors | Physical threshold check |
| 8 | `tilt_mag_std` | Tilt | $\text{std}(\sqrt{\theta_x^2 + \theta_y^2})$ | deg ($^\circ$) | Dynamic instability in net angular inclination | 30-sample tilt std |
| 9 | `vib_min` | Vibration | $\min_k a_k$ | g | Minimum dynamic acceleration trough | Window min |
| 10 | `tilt_trend_r2` | Tilt | $R^2$ of $\theta_{\text{mag}}(t)$ vs $t$ | [0, 1] | Linearity/steadiness of angular creep progression | Linear regression $R^2$ |
| 11 | `disp_trend_r2` | Displacement | $R^2$ of $d(t)$ vs $t$ | [0, 1] | Deterministic continuous roof convergence confidence | Linear regression $R^2$ |
| 12 | `spatial_num_abnormal_neighbors` | Spatial | $\sum_{j \in \mathcal{N}(i)} \mathbb{I}(\text{Elevated}_j)$ | count | Integer count of neighbours exceeding physical safety thresholds | Physical threshold sum |
| 13 | `tilt_accel_max` | Tilt | $\max \left\|\frac{\Delta^2 \theta}{\Delta t^2}\right\|$ | $\text{deg/s}^2$ | Rotational jerk / rapid angular snap prior to failure | 2nd discrete diff |
| 14 | `tilt_stability_index` | Tilt | $\frac{1}{1 + \text{std}(\theta) + |\text{slope}(\theta)|}$ | [0, 1] | Geometric structural stability metric (1=stable, 0=unstable) | Derivative + std |
| 15 | `spatial_pct_abnormal_neighbors` | Spatial | $\frac{\text{num\_abnormal\_neighbors}}{|\mathcal{N}(i)|} \times 100$ | % | Percentage of cluster exhibiting physical warning signs | Percentage |
| 16 | `tilt_rate_max` | Tilt | $\max \left\|\frac{\Delta \theta}{\Delta t}\right\|$ | deg/s | Peak angular velocity of tilting roof / pillar | 1st discrete diff |
| 17 | `disp_accel_max` | Displacement | $\max \frac{\Delta^2 d}{\Delta t^2}$ | $\text{mm/s}^2$ | Peak downward acceleration of roof sag | 2nd discrete diff |
| 18 | `disp_rate_max` | Displacement | $\max \frac{\Delta d}{\Delta t}$ | mm/s | Peak downward velocity (convergence rate) | 1st discrete diff |
| 19 | `vib_cov` | Vibration | $\frac{\text{std}}{\text{mean}}$ | dimensionless | Coefficient of variation of acceleration signal | std / mean |
| 20 | `vib_skewness` | Vibration | $\frac{1}{N}\sum \left(\frac{a_k - \mu}{\sigma}\right)^3$ | dimensionless | Asymmetry of micro-fracturing impulses | 3rd standardized moment |
| 21 | `disp_stability_index` | Displacement | $\frac{1}{1 + \text{std}(d) + |\text{slope}(d)|}$ | [0, 1] | Geomechanical roof stability metric | Derivative + std |
| 22 | `vib_crest_factor` | Vibration | $\frac{\max \|a_k\|}{\text{RMS}(a)}$ | dimensionless | Ratio of impulsive peak amplitude to continuous RMS | Peak / RMS |
| 23 | `vib_kurtosis` | Vibration | $\frac{1}{N}\sum \left(\frac{a_k - \mu}{\sigma}\right)^4 - 3$ | dimensionless | Heavy-tailedness indicator for micro-rock fracturing spikes | 4th standardized moment |
| 24 | `vib_spectral_energy_mid` | Vibration | $\sum_{f=10\text{Hz}}^{30\text{Hz}} |X(f)|^2$ | $\text{g}^2$ | Energy in intermediate band (equipment / mechanical vibrations) | FFT discrete power |
| 25 | `vib_dominant_frequency` | Vibration | $\arg\max_f |X(f)|$ | Hz | Dominant resonant frequency of the rock strata | FFT peak index |
| 26 | `vib_spectral_centroid` | Vibration | $\frac{\sum f \cdot |X(f)|}{\sum |X(f)|}$ | Hz | Spectral center of mass; shifts upward during crack propagation | Spectral weighted avg |
| 27 | `vib_spectral_energy_low` | Vibration | $\sum_{f=0.5\text{Hz}}^{10\text{Hz}} |X(f)|^2$ | $\text{g}^2$ | Energy in sub-10Hz band (geological settling, macro-movement) | FFT discrete power |
| 28 | `vib_spectral_energy_high` | Vibration | $\sum_{f=30\text{Hz}}^{50\text{Hz}} |X(f)|^2$ | $\text{g}^2$ | Energy in high band (acoustic emissions, microscopic fractures) | FFT discrete power |
| 29 | `vib_spectral_entropy` | Vibration | $-\sum p(f) \ln p(f)$ | nats | Complexity/disorder of vibration spectrum | Normalized power entropy |
| 30 | `vib_second_dominant_frequency`| Vibration | 2nd peak frequency in $|X(f)|$ | Hz | Secondary resonance harmonic in fractured strata | FFT 2nd peak |

---

## Detailed Physical Justifications

### 1. Vibration Domain Features
* **Micro-fracturing Signatures (`vib_kurtosis`, `vib_crest_factor`, `vib_skewness`):** As tensile and shear stresses exceed rock strength, microscopic fissures emit impulsive acoustic bursts. These transients produce sharp peaks above background noise, heavily inflating kurtosis ($> 3.0$) and crest factor ($> 3.5$).
* **Spectral Shift (`vib_spectral_centroid`, `vib_spectral_energy_high`):** Stable mine strata exhibit low-frequency mechanical background noise ($< 10\text{ Hz}$). Progressive bed separation and micro-crack propagation shift seismic energy into higher frequencies ($30\text{--}50\text{ Hz}$), increasing the spectral centroid.

### 2. Tilt Domain Features
* **Angular Creep (`tilt_trend_r2`, `tilt_stability_index`):** Roof props and strata tilt progressively under differential loading. A high $R^2$ ($> 0.85$) combined with a steady slope indicates monotonic structural yield rather than transient disturbance.
* **Dynamic Failure Snap (`tilt_rate_max`, `tilt_accel_max`):** In the tertiary creep stage prior to shear failure, angular displacement accelerates sharply.

### 3. Displacement Domain Features
* **Roof Convergence Rate (`disp_rate_max`, `disp_trend_r2`):** Continuous downward movement measured by time-of-flight laser or linear potentiometers. A linear or parabolic trend indicates active bed separation.
* **Subsidence Acceleration (`disp_accel_max`):** Tertiary creep inflection point ($\frac{d^2d}{dt^2} > 0$), serving as a deterministic trigger for evacuation.

### 4. Spatial Mesh Topology Features
* **Spatial Gradient (`spatial_disp_gradient_max`):** Differential displacement between adjacent nodes indicates shear strain and bending moment across roof beams.
* **Cluster Escalation (`spatial_pct_abnormal_neighbors`):** Isolated sensor movement can occur from local rock spalling or sensor bump. When multiple adjacent nodes concurrently observe physical anomalies, risk escalates from local to systemic subsidence.

---

## Live Inference Compute Architecture
During real-time execution in the FastAPI `/api/sensor-data` endpoint:
1. Every incoming packet is appended to that node's FIFO circular buffer (capacity 30 samples).
2. The buffer is cleaned via a 3-point rolling median filter.
3. If buffer size $< 10$ samples, safe fallback defaults are used.
4. When buffer size $\ge 10$, FFT, discrete differentiation, and SciPy linear regressions are computed within **$0.05$ seconds**, feeding directly into the 30-feature `StandardScaler` and XGBoost/Random Forest models.
