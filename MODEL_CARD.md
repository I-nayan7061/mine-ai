# Model Card: Underground Mine Strata Risk Classifier & Anomaly Detector
**SIH26025 - NexGen | Disaster Management**

---

## 1. Model Details
- **Primary Advanced Model**: XGBoost Multiclass Classifier (`multi:softprob`, 120 estimators, max_depth=6).
- **Primary Interpretable Baseline**: Random Forest Classifier (balanced class weights, 120 estimators).
- **Unsupervised Anomaly Detector**: Isolation Forest (150 estimators, contamination=0.08).
- **Explainability**: SHAP TreeExplainer.
- **Input Dimensions**: 30 engineered features selected from 69 candidate indicators.
- **Output Classes**: `NORMAL`, `WARNING`, `HIGH`, `CRITICAL`.

---

## 2. Intended Use & Safety Scope
- **Primary Intention**: Real-time early warning decision-support for underground coal mine safety engineers.
- **Out of Scope**: This prototype does NOT predict the exact millisecond of a geological collapse and must NOT supersede statutory mine ventilation and roof control plans approved by DGMS (Directorate General of Mines Safety).

---

## 3. Evaluation Metrics & Safety Recall

| Evaluation Metric | Random Forest (Baseline) | XGBoost (Advanced) | LightGBM |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 96.42% | **99.13%** | 98.76% |
| **Weighted F1-Score** | 0.964 | **0.991** | 0.988 |
| **High Risk Recall** | 97.42% | **100.00%** | 99.48% |
| **Critical Risk Recall** | **100.00%** | **100.00%** | **100.00%** |
| **Inference Time per Sample** | 0.128 ms | **0.011 ms** | 0.018 ms |

### Safety Philosophy: Zero False Negatives on Critical Events
In mining safety, a false alarm (lowering machinery speed temporarily) is far less dangerous than a false negative (failing to alert workers before a roof collapse). Both **Random Forest** and **XGBoost** achieved **100.00% Recall on the CRITICAL class** on unseen test cycles.

---

## 4. Limitations & Edge Cases
1. **ToF Optical Obstruction**: VL53L1X requires an unobscured line-of-sight to the reference target; dense coal dust clouds may temporarily attenuate optical range signals (detected by the data loader's bound validator).
2. **In-Situ Calibration**: Sensor baseline mounting angle offsets must be zeroed upon physical installation in the gallery.
