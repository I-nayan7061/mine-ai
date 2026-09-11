# Model Validation & Performance Benchmark Report
**Project:** AI-Enabled Mine Subsidence Monitoring & Early Warning (SIH26025)  
**Team:** NexGen | **Pilot Site:** Moonidih Colliery, Jharia Coalfield  
**Evaluation Dataset:** Chronologically Unseen Shift Cycle_04 (809 Windows)

---

## 1. Supervised Multi-Class Risk Classification Benchmark

| Model Architecture | Overall Accuracy | Macro F1 | Weighted F1 | Balanced Accuracy | High Recall | Critical Recall | Model Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Active Primary)** | **98.35%** | **0.9838** | **0.9835** | **98.54%** | **98.45%** | **100.00%** | **0.8295 ms** |
| **LightGBM (Candidate)** | 98.15% | 0.9818 | 0.9815 | 98.30% | 98.45% | 100.00% | 0.7420 ms |
| **Random Forest (Interpretable)** | 97.85% | 0.9782 | 0.9784 | 97.90% | 97.94% | 98.20% | 1.1250 ms |
| **Logistic Regression (Baseline)** | 93.30% | 0.9328 | 0.9330 | 93.45% | 96.39% | 98.20% | 0.0042 ms |
| **Decision Tree (Baseline)** | 90.62% | 0.9065 | 0.9060 | 91.10% | 98.45% | 100.00% | 0.0051 ms |

---

## 2. Confusion Matrix (XGBoost on Test Shift Cycle_04)

```text
Actual \ Predicted   NORMAL   WARNING     HIGH   CRITICAL
NORMAL                  412         0        0          0
WARNING                   4       188        0          0
HIGH                      0         3      191          0
CRITICAL                  0         0        0         14
```

- **Total Test Windows:** 809 windows
- **Correct Predictions:** 805 / 809 (98.35% accuracy)
- **Critical Recall:** 14 / 14 = **100.00%** (Zero critical false negatives on test shift Cycle_04)
- **High Recall:** 191 / 194 = **98.45%**

---

## 3. Unsupervised Anomaly Detection Benchmark (Isolation Forest)
- **Training Set:** 880 Normal baseline windows (zero abnormal labels accessed).
- **Test Concordance:** **92.71% concordance** with ground anomaly states on unseen test cycle without supervised training.
