"""Environmental & Satellite Remote Sensing Feature Engineering.
SIH26025 - NexGen | Multi-Modal Subsidence Early Warning

Extracts and normalizes features from:
1. Surface meteorological station (Rainfall, 24h/72h cumulative precipitation, soil moisture).
2. Satellite Earth Observation (Sentinel-1 InSAR velocity, Sentinel-2 NDVI/NDMI, Landsat-9 LST).
3. Hydro-mechanical coupling indicators (interaction of surface water saturation with subterranean strata sag).
"""

from typing import Any, Dict, Optional
import numpy as np


def extract_environmental_features(
    weather_data: Optional[Dict[str, Any]] = None,
    satellite_data: Optional[Dict[str, Any]] = None,
    disp_features: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """Extract standard numerical feature dictionary for weather and satellite streams.
    
    Guarantees deterministic, leakage-free numerical values with safety defaults.
    """
    w = weather_data or {}
    s = satellite_data or {}
    d = disp_features or {}

    # 1. Weather domain features
    rain_rate = float(w.get("rain_rate_mm_h", 0.0))
    rain_24h = float(w.get("rain_cum_24h_mm", 12.0))
    rain_72h = float(w.get("rain_cum_72h_mm", 40.0))
    soil_deep = float(w.get("soil_moisture_deep", 0.35))
    temp_c = float(w.get("ambient_temp_c", 28.0))
    humidity = float(w.get("relative_humidity_pct", 75.0))
    pressure = float(w.get("surface_pressure_hpa", 990.0))

    # Pore-water pressure proxy (kPa) on mine overburden
    pore_pressure_kpa = float(w.get("pore_pressure_kpa", 90.0 + (rain_72h * 1.8) + (soil_deep * 80.0)))

    # 2. Satellite Earth Observation domain features
    insar_vel = float(s.get("sentinel1_insar_velocity_mm_yr", -18.4))
    insar_cum = float(s.get("sentinel1_cum_los_displacement_mm", -32.6))
    ndvi_anomaly = float(s.get("sentinel2_ndvi_anomaly", -0.14))
    ndmi_moisture = float(s.get("sentinel2_ndmi_moisture", 0.22))
    thermal_anomaly = float(s.get("landsat_thermal_anomaly_k", 3.4))

    # 3. Hydro-Mechanical Coupling
    # Strata roof convergence rate accelerates when rainwater weakens overburden cohesion
    disp_rate = abs(float(d.get("disp_rate_max", d.get("disp_velocity_mean", 0.01))))
    hydro_mechanical_risk = float(disp_rate * (1.0 + (rain_72h / 50.0)) * (1.0 + soil_deep))

    # Surface-to-subsurface kinematic alignment
    # InSAR sinking rate (mm/yr converted to mm/day) vs gallery convergence (mm/day)
    insar_subsidence_rate_daily = abs(insar_vel) / 365.0
    subsurface_disp_daily = disp_rate * 60.0 * 24.0  # mm/min -> mm/day
    surface_subsurface_ratio = float(subsurface_disp_daily / (insar_subsidence_rate_daily + 1e-4))

    return {
        "env_rain_rate_mm_h": rain_rate,
        "env_rain_cum_24h_mm": rain_24h,
        "env_rain_cum_72h_mm": rain_72h,
        "env_soil_moisture_deep": soil_deep,
        "env_pore_pressure_kpa": pore_pressure_kpa,
        "sat_insar_velocity_mm_yr": insar_vel,
        "sat_insar_cum_los_mm": insar_cum,
        "sat_ndvi_anomaly": ndvi_anomaly,
        "sat_ndmi_moisture": ndmi_moisture,
        "sat_thermal_anomaly_k": thermal_anomaly,
        "hydro_mechanical_coupling_risk": round(hydro_mechanical_risk, 4),
        "surface_subsurface_convergence_ratio": round(surface_subsurface_ratio, 4)
    }
