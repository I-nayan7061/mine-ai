# Model Limitations, Operational Boundary & Geotechnical Constraints
**Project:** AI-Enabled Mine Subsidence Monitoring & Early Warning (SIH26025)  
**Team:** NexGen | **Category:** Hardware & AI/IoT | **SIH Problem Statement:** SIH26025  

---

## 1. Prototype & Research Status
> [!WARNING]  
> This system is a **student research and engineering prototype**. It has NOT been certified by the Directorate General of Mines Safety (DGMS) or Central Institute of Mining and Fuel Research (CSIR-CIMFR). It must NOT be deployed as an autonomous life-safety evacuation system without statutory in-situ calibration and field validation.

---

## 2. Technical Limitations

### 2.1 Synthetic Training Data Dependency
- The subterranean sensor streams are generated using physics-grounded mathematical differential equations modeling roof strata deflection.
- While parameters (vibration RMS, tilt angles, convergence rates) are calibrated against published geotechnical literature from Jharia longwall panels, **the model has not been trained on real in-situ sensor logs from actual catastrophic roof falls**.

### 2.2 Satellite Revisit & Latency Limitations
- Satellite InSAR observations (Sentinel-1 / NISAR) have repeat revisit periods of **6 to 12 days**.
- InSAR provides regional surface subsidence basin context. It **cannot detect sudden localized roof collapses occurring over seconds or minutes**. Underground safety relies primarily on subterranean 1.0 Hz IoT nodes.

### 2.3 Spatial Generalization (Domain Shift)
- The spatial graph is configured for a 5-node geometry in Moonidih Colliery (Barakar Formation sandstone/shale overburden, 320m depth).
- Geological variance across different coalfields (e.g., Raniganj, Singrauli, Korba) with differing Young's moduli, joint frequencies, or mining methods (bord-and-pillar vs longwall) requires site-specific recalibration.

### 2.4 Weather Spatial Resolution
- Gridded IMD precipitation data operates at 0.25° (~27 km) resolution. Local micro-topographic cloudbursts may differ from gridded values.
