# India Meteorological Data Pipeline Architecture
**Project:** AI-Enabled Mine Subsidence Monitoring & Early Warning (SIH26025)  
**Team:** NexGen | **Pilot Site:** Moonidih Colliery, Dhanbad, Jharkhand  

---

## 1. Primary Source: India Meteorological Department (IMD)
- **Datasets:** IMD Daily Gridded Rainfall (0.25° x 0.25° resolution) and Maximum/Minimum Temperature (0.5° x 0.5° resolution).
- **AOI Grid Point:** Latitude 23.75° N, Longitude 86.50° E (covering Moonidih Colliery leasehold).
- **Features Extracted:**
  - `weather_rain_24h`: 24-hour cumulative rainfall (mm).
  - `weather_rain_72h`: 72-hour cumulative precipitation representing short-term percolation memory.
  - `weather_rain_7d` & `weather_rain_14d`: Extended monsoon saturation indicators.
  - `weather_temp_mean` & `weather_temp_range`: Diurnal thermal cycles.

---

## 2. Supplementary Hourly Source: ERA5-Land & Open-Meteo IMD-Calibrated Grid
- **Source:** Open-Meteo IMD-Calibrated Grid / ECMWF ERA5-Land.
- **Resolution:** 0.1° x 0.1° hourly surface reanalysis.
- **Features Extracted:**
  - `weather_soil_moisture_deep`: Volumetric soil moisture at 28–100 cm root zone (m³/m³).
  - `weather_soil_saturation_index`: Soil saturation ratio relative to field capacity.
  - `weather_pore_pressure_kpa`: Overburden hydrostatic pore-water pressure proxy ($P = 90.0 + 1.8 \cdot 	ext{Rain}_{72	ext{h}} + 80.0 \cdot 	ext{Moisture}_{	ext{deep}}$).

---

## 3. Freshness & Data Quality Handling
- Ground sensors update at 1.0 Hz; weather data updates hourly/daily.
- The pipeline tracks `weather_age_hours` (typically 1.0 to 3.0 hours) and sets `has_weather = 1.0`.
- In offline or disconnected network environments, the system falls back gracefully to a calibrated local meteorological profile.
