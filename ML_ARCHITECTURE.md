# Machine Learning & Geotechnical AI Architecture
**Project:** AI-Enabled Mine Subsidence Monitoring, Prediction & Early Warning (SIH26025)  
**Team:** NexGen | **Theme:** Disaster Management

---

## 1. Architectural Philosophy
In underground coal mines, false alarms cause costly production halts, while missed true hazards cause catastrophic roof falls. Therefore, the ML architecture follows a **Defense-in-Depth, Multi-Layered Paradigm**:

```
Layer 1: Sensor Signal Preprocessing & Robust Filtering
   ↓
Layer 2: Multi-Domain Feature Extraction (Time, Frequency, Geomechanics, Spatial)
   ↓
Layer 3: Unsupervised Statistical Anomaly Detection (Isolation Forest)
   ↓
Layer 4: Supervised Tabular Strata Risk Classification (XGBoost / Random Forest)
   ↓
Layer 5: Multi-Node Spatial Graph & Neighborhood Risk Zone Analysis
   ↓
Layer 6: Geotechnical Multi-Criteria Risk Fusion Engine [0-100 Score]
   ↓
Layer 7: Explainable AI (SHAP Tree Attribution)
```

---

## 2. Feature Extraction Mechanics

### 2.1 Vibration Features (`src/vibration_features.py`)
- **Root Mean Square (RMS)**: Measures effective vibration energy:
  $$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^N v_i^2}$$
- **Crest Factor & Shape Factor**: Detects impulsive crack propagation pops vs background machinery noise:
  $$\text{Crest Factor} = \frac{\max(|v|)}{\text{RMS}}, \quad \text{Shape Factor} = \frac{\text{RMS}}{\text{MAV}}$$
- **Fast Fourier Transform (FFT) Spectral Analysis**:
  - Dominant frequency ($f_{\text{dom}}$) and spectral centroid.
  - Frequency band energy ratios: Low-band ($\le 0.1\text{ Hz}$), Mid-band ($0.1 - 0.3\text{ Hz}$), High-band ($> 0.3\text{ Hz}$).
  - Spectral entropy to capture seismic broadband noise vs sinusoidal machinery harmonics.

### 2.2 Tilt Features (`src/tilt_features.py`)
- **Tilt Magnitude**:
  $$\theta_{\text{mag}} = \sqrt{\theta_x^2 + \theta_y^2}$$
- **Tilt Rate & Angular Acceleration**: First and second numerical derivatives:
  $$\omega_{\text{tilt}} = \frac{d\theta_{\text{mag}}}{dt} \quad (\text{deg/min}), \quad \alpha_{\text{tilt}} = \frac{d\omega_{\text{tilt}}}{dt}$$
- **Linear Trend Slope & Goodness-of-Fit $R^2$**: Distinguishes directional strata subsidence from zero-mean sensor noise.

### 2.3 Displacement Features (`src/displacement_features.py`)
- **Deformation Velocity ($v_{\text{disp}}$)**: Rate of convergence ($mm/min$):
  $$v = \frac{\Delta d}{\Delta t} \times 60$$
- **Net Cumulative Convergence**: $\Delta d = d_{\text{current}} - d_{\text{baseline}}$.
- **Strata Stability Index**: Normalized metric penalizing variance, slope, and acceleration:
  $$\text{Stability} = \frac{1}{1 + \sigma_d + |\text{slope}| + v_{\max}/2}$$

### 2.4 Multi-Node Spatial Graph (`src/spatial_features.py`)
- Represents nodes $N_1, \dots, N_k$ connected along mine gallery entries and crosscuts.
- Calculates spatial deformation gradient across physical tunnel spans:
  $$\text{Gradient}_{ij} = \frac{|d_i - d_j|}{\text{Distance}_{ij}} \quad (\text{mm/m})$$
- Distance-weighted neighbor anomaly:
  $$A_{\text{spatial}} = \frac{\sum_j w_{ij} \cdot \mathbb{I}(\text{Node } j \text{ abnormal})}{\sum_j w_{ij}}, \quad w_{ij} = \frac{1}{\text{dist}_{ij} + \epsilon}$$

---

## 3. Risk Fusion Engine Formula

The final continuous Risk Score $R \in [0, 100]$ fuses physical severity, anomaly score, classifier probabilities, and spatial-temporal dynamics:

$$R_{\text{composite}} = w_{\text{vib}} S_{\text{vib}} + w_{\text{tilt}} S_{\text{tilt}} + w_{\text{disp}} S_{\text{disp}} + w_{\text{anom}} S_{\text{anom}} + w_{\text{temp}} S_{\text{temp}} + w_{\text{spat}} S_{\text{spat}}$$

Where default configurable weights in `config.yaml` are:
- $w_{\text{disp}} = 0.25$ (Primary indicator of roof convergence)
- $w_{\text{tilt}} = 0.20$ (Inclinometer ground deflection)
- $w_{\text{anom}} = 0.20$ (Isolation Forest unsupervised outlier score)
- $w_{\text{vib}} = 0.15$ (Seismic / fracturing acoustic energy)
- $w_{\text{temp}} = 0.10$ (Persistence and trend slope)
- $w_{\text{spat}} = 0.10$ (Cluster neighbor correlation)

---

## 4. Explainability with SHAP
Rather than outputting a black-box number, the system utilizes **SHAP (SHapley Additive exPlanations)** TreeExplainer to compute exact feature attributions for any elevated risk status, translating math into domain-specific geomechanical insights:
1. `disp_rate_max`: "Accelerated roof-to-floor convergence velocity (+34.2%)"
2. `tilt_trend_slope`: "Increasing directional tilt drift (+22.1%)"
3. `vib_rms`: "Seismic vibration energy exceeding ambient baseline (+18.4%)"
4. `spatial_pct_abnormal_neighbors`: "Adjacent gallery nodes also deforming (+14.3%)"
