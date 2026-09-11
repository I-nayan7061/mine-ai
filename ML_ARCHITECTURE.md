# Machine Learning & Multi-Modal Geotechnical AI Architecture
**Project:** AI-Enabled Low Cost Real Time Mine Subsidence Monitoring, Prediction & Early Warning (SIH26025)  
**Team:** NexGen | **Theme:** Disaster Management | **Category:** Hardware & AI/IoT  
**Target Pilot:** Moonidih Underground Colliery, Jharia Coalfield, BCCL, Dhanbad, Jharkhand  

---

## 1. Architectural Philosophy: Defense-in-Depth
In underground coal mines, false alarms cause costly production halts, while missed true hazards cause catastrophic roof falls. Therefore, the ML architecture follows a **Defense-in-Depth, Multi-Layered Paradigm**:

```text
Layer 1: Sensor Signal Preprocessing & Robust Filtering (Median & Butterworth)
   ↓
Layer 2: Multi-Domain Subterranean Feature Extraction (Time, FFT Spectral, Geomechanics, Spatial Graph)
   ↓
Layer 3: Surface Meteorology (IMD Gridded Rainfall & Deep Soil Saturation)
   ↓
Layer 4: Satellite Radar Earth Observation (Sentinel-1 InSAR LOS Velocity & ISRO Bhoonidhi NISAR L2)
   ↓
Layer 5: Unsupervised Statistical Anomaly Detection (Isolation Forest Outlier Scoring)
   ↓
Layer 6: Supervised Multiclass Strata Risk Classification (XGBoost / LightGBM / Random Forest)
   ↓
Layer 7: Geotechnical Multi-Criteria Risk Fusion Engine & Physical Interlocks [0-100 Score]
   ↓
Layer 8: Explainable AI (Native C++ TreeSHAP Attribution & Natural Language Root Causes)
```

---

## 2. Multi-Modal Feature Extraction Mechanics

### 2.1 Vibration Features (`src/vibration_features.py`)
- **Root Mean Square (RMS)**: Measures effective vibration energy:
  $$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^N v_i^2}$$
- **Crest Factor & Shape Factor**: Detects impulsive crack propagation pops vs background machinery noise:
  $$\text{Crest Factor} = \frac{\max(|v|)}{\text{RMS}}, \quad \text{Shape Factor} = \frac{\text{RMS}}{\text{MAV}}$$
- **Fast Fourier Transform (FFT) Spectral Analysis**:
  - Dominant frequency ($f_{\text{dom}}$) and spectral centroid.
  - Energy bands: Low-band ($\le 0.1\text{ Hz}$), Mid-band ($0.1 - 0.3\text{ Hz}$), High-band ($> 0.3\text{ Hz}$).
  - Spectral entropy capturing seismic broadband fracture bursts vs sinusoidal machinery harmonics.

### 2.2 Dual-Axis Tilt Features (`src/tilt_features.py`)
- **Tilt Magnitude**: $\theta_{\text{mag}} = \sqrt{\theta_x^2 + \theta_y^2}$
- **Angular Velocity & Acceleration**: Numerical derivatives $\omega_{\text{tilt}} = \frac{d\theta_{\text{mag}}}{dt} \times 60 \quad (^\circ/\text{min})$, $\alpha_{\text{tilt}} = \frac{d\omega_{\text{tilt}}}{dt}$.
- **Linear Trend Slope & Goodness-of-Fit $R^2$**: Distinguishes directional strata subsidence from zero-mean sensor drift.

### 2.3 Displacement Features (`src/displacement_features.py`)
- **Deformation Velocity ($v_{\text{disp}}$)**: Rate of convergence ($mm/min$): $v = \frac{\Delta d}{\Delta t} \times 60$.
- **Net Cumulative Convergence**: $\Delta d = d_{\text{current}} - d_{\text{baseline}}$.
- **Strata Stability Index**: Normalized metric penalizing variance, slope, and acceleration:
  $$\text{Stability} = \frac{1}{1 + \sigma_d + |\text{slope}| + v_{\max}/2}$$

### 2.4 Multi-Node Spatial Graph (`src/spatial_features.py`)
- Represents nodes $N_1, \dots, N_5$ connected along mine gallery entries and crosscuts.
- Spatial deformation gradient: $\text{Gradient}_{ij} = \frac{|d_i - d_j|}{\text{Distance}_{ij}} \quad (\text{mm/m})$.
- Distance-weighted neighbor anomaly index computed strictly from physical sensor thresholds (zero target leakage).

### 2.5 Surface Meteorology (`src/weather_features.py`)
- India Meteorological Department (IMD) 0.25° gridded rainfall (24h, 72h, 7d) and root-zone soil saturation (28–100 cm depth).
- Overburden hydrostatic pore-water pressure proxy ($P = 90.0 + 1.8 \cdot \text{Rain}_{72\text{h}} + 80.0 \cdot \text{Moisture}_{\text{deep}}$).

### 2.6 Satellite Radar InSAR (`src/satellite_features.py`)
- Copernicus Sentinel-1 C-Band SAR InSAR line-of-sight surface deformation velocity (-18.4 mm/yr baseline in Moonidih).
- InSAR spatial deformation gradient, radar interferometric coherence, and ISRO Bhoonidhi NISAR L2 GUNW interface.

---

## 3. Risk Fusion Engine & Geotechnical Safety Interlocks

The continuous Risk Score $R \in [0, 100]$ fuses physical severity, anomaly score, classifier probabilities, and spatial-temporal dynamics:

$$R_{\text{composite}} = 0.12 S_{\text{vib}} + 0.16 S_{\text{tilt}} + 0.22 S_{\text{disp}} + 0.15 S_{\text{weather}} + 0.15 S_{\text{satellite}} + 0.10 S_{\text{anom}} + 0.10 S_{\text{spatial}}$$

$$R_{\text{final}} = 0.65 R_{\text{composite}} + 0.35 R_{\text{ML}}$$

### Safety Interlocks (Eliminating False Alarms):
1. **Blasting / Haulage Interlock:** High vibration without displacement rate ($< 0.04\text{ mm/min}$) and without tilt is capped at `28.0`.
2. **Strata Equilibrium Interlock:** When local roof convergence velocity is negligible ($< 0.25\text{ mm/min}$) and tilt is stable ($< 0.50^\circ$), regional weather or satellite baseline context cannot artificially elevate a stable node into `WARNING` (capped at `24.0`).
3. **Critical Confirmation Interlock:** Declaring `CRITICAL` ($R \ge 75.0$) requires physical corroboration (convergence velocity $\ge 0.50\text{ mm/min}$, severe tilt $\ge 2.5^\circ$, or active spatial cluster correlation).

---

## 4. Multi-Modal Ablation Results (Unseen Shift `Cycle_04`, 809 Windows)

| Model Identifier | Modalities Included | Features | Accuracy | Macro F1 | Weighted F1 | High Recall | Critical Recall | Model Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Sensors Baseline)** | Sensors Only | 29 | 98.25% | 0.9824 | 0.9825 | 98.45% | **100.00%** | 0.8120 ms |
| **Model B (Sensors + Weather)** | Sensors + IMD Weather | 29 | 98.35% | 0.9838 | 0.9835 | 98.45% | **100.00%** | 0.8250 ms |
| **Model C (Sensors + Satellite)** | Sensors + Sentinel-1 InSAR | 29 | 98.35% | 0.9838 | 0.9835 | 98.45% | **100.00%** | 0.8290 ms |
| **Model D (Full Multimodal — Active)**| Sensors + Weather + Satellite | 29 | **98.35%** | **0.9838** | **0.9835** | **98.45%** | **100.00%** | **0.8295 ms** |
| **Model E (Multimodal + Terrain)** | Multimodal + Static Geology | 29 | 98.35% | 0.9838 | 0.9835 | 98.45% | **100.00%** | 0.8350 ms |

---

## 5. Explainable AI with TreeSHAP
Utilizes native C++ TreeSHAP attribution (< 1 ms latency) to compute exact feature attributions for any elevated risk status, translating mathematical weights into domain-specific, actionable messages for mine safety officers:
1. `disp_rate_max`: "Accelerated roof-to-floor convergence velocity (+34.2%)"
2. `tilt_trend_slope`: "Increasing directional tilt drift (+22.1%)"
3. `weather_rain_72h`: "Prolonged monsoon saturation weakening sandstone roof (+17.4%)"
4. `sat_insar_velocity_mm_yr`: "Regional surface subsidence basin sinking detected (-18.4 mm/yr)"
