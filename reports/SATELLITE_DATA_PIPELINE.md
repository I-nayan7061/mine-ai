# Satellite Earth Observation & InSAR Pipeline Architecture
**Project:** AI-Enabled Mine Subsidence Monitoring & Early Warning (SIH26025)  
**Team:** NexGen | **Pilot Site:** Moonidih Colliery, Jharia Coalfield, BCCL, Dhanbad, Jharkhand  

---

## 1. Radar Remote Sensing Core: Sentinel-1 InSAR
- **Constellation:** Copernicus Sentinel-1 C-Band Synthetic Aperture Radar (SAR) (5.546 cm wavelength).
- **Orbit Geometry:** Descending Track 121, IW (Interferometric Wide Swath) mode, VV polarization.
- **Repeat Pass Revisit:** 12 days.
- **InSAR Processing Chain:**
  1. Sentinel-1 SLC Pair Acquisition -> Precise Orbit Ephemerides application.
  2. Sub-pixel Image Co-registration (< 0.001 pixel accuracy).
  3. Interferogram Generation & Flat-Earth phase subtraction.
  4. Topographic Phase Removal using SRTM 30m DEM.
  5. Goldstein Adaptive Phase Filtering.
  6. Minimum Cost Flow (MCF) 2D Phase Unwrapping.
  7. Phase-to-Displacement Conversion & Geocoding to WGS-84 UTM Zone 45N.
  8. Spatial Aggregation & Zonal Statistics around Moonidih Colliery Leasehold AOI.

---

## 2. Secondary Satellite Layer: ISRO / NASA NISAR
- **Mission:** NASA-ISRO Synthetic Aperture Radar (NISAR).
- **Instrument:** Dual-frequency L-band (24 cm) and S-band (9 cm) SweepSAR.
- **Product Utilized:** L2 Geocoded Unwrapped Interferogram (GUNW) through ISRO Bhoonidhi.
- **Advantage in Jharia:** L-band penetration through dense vegetation and soil moisture with superior temporal coherence compared to C-band.

---

## 3. Optical Secondary Context: Sentinel-2 & Landsat-9
- **Sentinel-2 MSI:** Normalized Difference Vegetation Index (NDVI) anomaly and Normalized Difference Moisture Index (NDMI). Surface tension cracks damage topsoil vegetation, manifesting as a local NDVI drop (ΔNDVI < -0.15).
- **Landsat-9 TIRS:** Land Surface Temperature (LST) thermal anomalies (ΔT > +3.0 K) identifying subsurface coal seam spontaneous combustion heating.

---

## 4. Freshness & Integration Rule
- Satellite passes occur every 6 to 12 days; therefore, **satellite observations provide regional deformation context, NOT real-time 1.0 Hz underground roof monitoring**.
- The pipeline explicitly tracks `satellite_age_days` and provides model-compatible fallback when passes are delayed or obscured by low coherence.
