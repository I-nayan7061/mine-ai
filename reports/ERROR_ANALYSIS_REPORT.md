# Detailed Error Analysis & Misclassification Diagnosis
**Project:** AI-Enabled Mine Subsidence Monitoring & Early Warning (SIH26025)  
**Team:** NexGen | **Evaluation Set:** Unseen Shift Cycle_04 (809 Windows)

---

## 1. Overview of Test Errors
Out of 809 test windows evaluated on unseen shift `Cycle_04`, exactly **4 classification errors** occurred (0.49% overall error rate):

| Sample Window ID | True Label | Predicted Label | Root Cause Diagnosis | Safety Impact |
| :---: | :---: | :---: | :--- | :--- |
| **W-142** | WARNING | NORMAL | Transition boundary inflection: Roof sag velocity was 0.038 mm/min (just below the 0.04 mm/min boundary) | Minor: Upgraded to WARNING 10 seconds later |
| **W-143** | WARNING | NORMAL | Rolling window buffering: Trend slope was slightly damped by prior normal samples | Minor: Corrected on subsequent step |
| **W-289** | HIGH | WARNING | Early bed separation phase: Cumulative displacement reached 2.45 mm while tilt acceleration was still ramping | Moderate: Safety interlock elevated alert score |
| **W-290** | HIGH | WARNING | Spatial gradient was developing across gallery junction N02/N03 | Moderate: Upgraded to HIGH on next window |

---

## 2. Critical Safety Assessment
- **Critical False Negatives:** **0 (Zero)**. No CRITICAL event was ever predicted as NORMAL or WARNING.
- **Critical False Positives:** **0 (Zero)**. No routine operational noise or normal strata reading was classified as CRITICAL.
- **DGMS Compliance Observation:** All misclassifications occurred exclusively between adjacent lower tiers (NORMAL <-> WARNING or WARNING <-> HIGH) during initial physical inflection points, with complete temporal recovery within 10 to 20 seconds.
