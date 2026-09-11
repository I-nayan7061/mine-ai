# Spatio-Temporal Multimodal Data Alignment Specification
**Project:** AI-Enabled Mine Subsidence Monitoring & Early Warning (SIH26025)  
**Team:** NexGen | **Pilot Site:** Moonidih Colliery, Jharia Coalfield  

---

## 1. Multi-Scale Alignment Strategy
A critical challenge in multi-modal mining AI is fusing observations spanning vastly different temporal and spatial scales:

```
Subterranean IoT:   1.0 Hz (1 second)          Local Gallery (0 - 50m)
Surface Weather:    Hourly / Daily             Regional Grid (0.1° - 0.25°)
Satellite Radar:    6 - 12 Days Revisit        Regional Surface AOI (km²)
```

---

## 2. Spatio-Temporal Join Architecture
Data alignment is executed via a deterministic multi-key join:
1. **Spatial Key:** Coordinates `(23.7438° N, 86.4172° E)` + Mine Leasehold Polygon `BCCL-MOONIDIH-01`.
2. **Temporal Key:** Rolling window timestamp $t_{	ext{end}}$.
   - Nearest preceding valid IMD weather record within 24 hours.
   - Nearest preceding valid Sentinel-1 InSAR scene within 14 days.
3. **Freshness Tracking:**
   - `sensor_age_seconds`: Age of newest subterranean sensor reading ($< 2	ext{ s}$).
   - `weather_age_hours`: Hours since last meteorological station observation.
   - `satellite_age_days`: Days since last satellite SAR acquisition pass.
4. **Missing Modality Handling:**
   - `has_weather`: Binary indicator (1.0 = observed, 0.0 = imputed).
   - `has_satellite`: Binary indicator (1.0 = observed, 0.0 = imputed).

---

## 3. Master Multimodal Dataset
- **File:** `data/processed/multimodal_training_dataset.csv`
- **Schema:**
  `timestamp, mine_id, node_id, state, district, latitude, longitude, sensor_features (29), weather_features (11), satellite_features (8), terrain_features (4), risk_label`
- **Integrity Guarantee:** Verified zero row-number joins; all records joined by explicit spatial coordinates and timestamp windows.
