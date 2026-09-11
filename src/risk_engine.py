"""Multi-Criteria Strata Risk Fusion Engine with Physical Consistency Interlocks.
SIH26025 - NexGen | Real-Time Geotechnical Risk Assessment

Combines:
1. Physical sensor severity scores (vibration RMS, tilt magnitude, displacement velocity)
2. Unsupervised Anomaly Score (Isolation Forest [0-100])
3. Supervised ML Classifier Probabilities
4. Temporal persistence and acceleration trends
5. Multi-node spatial neighborhood correlation

Safety Interlocks:
- False-Positive Immunity: High vibration without physical displacement/tilt deformation
  is flagged as Operational Vibration Noise and capped below HIGH/CRITICAL strata risk.
- True Critical Subsidence requires correlated strata convergence and angular tilt.

Outputs a continuous Risk Score in [0, 100] mapped to:
- NORMAL   (0.0 - 24.9)
- WARNING  (25.0 - 49.9)
- HIGH     (50.0 - 74.9)
- CRITICAL (75.0 - 100.0)
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from .config import app_config
from .utils import get_logger

logger = get_logger("mine_ai.risk_engine")


class RiskEngine:
    """Configurable fusion engine for continuous strata subsidence risk scoring."""

    def __init__(self, config=None):
        self.config = config or app_config
        self.weights = self.config.risk_weights
        self.thresholds = self.config.risk_thresholds

        # Physical calibration parameters from config
        self.vib_nom_rms = self.config.get("risk_engine.severity_scale_params.vibration_nominal_rms", 0.10)
        self.vib_crit_rms = self.config.get("risk_engine.severity_scale_params.vibration_critical_rms", 2.0)
        self.tilt_nom_deg = self.config.get("risk_engine.severity_scale_params.tilt_nominal_deg", 0.50)
        self.tilt_crit_deg = self.config.get("risk_engine.severity_scale_params.tilt_critical_deg", 5.0)
        self.disp_rate_nom = self.config.get("risk_engine.severity_scale_params.displacement_rate_nominal", 0.02)
        self.disp_rate_crit = self.config.get("risk_engine.severity_scale_params.displacement_rate_critical", 1.50)

    def compute_sensor_subscores(
        self,
        vibration_rms: float,
        tilt_magnitude: float,
        displacement_rate_mm_per_min: float,
        displacement_mm: float = 0.0
    ) -> Tuple[float, float, float]:
        """Convert raw physical parameters to normalized 0-100 severity subscores."""
        # Vibration subscore
        vib_score = np.clip(
            ((vibration_rms - self.vib_nom_rms) / max(0.01, self.vib_crit_rms - self.vib_nom_rms)) * 100.0,
            0.0, 100.0
        )

        # Tilt subscore
        tilt_score = np.clip(
            ((tilt_magnitude - self.tilt_nom_deg) / max(0.1, self.tilt_crit_deg - self.tilt_nom_deg)) * 100.0,
            0.0, 100.0
        )

        # Displacement rate subscore
        rate_score = np.clip(
            ((displacement_rate_mm_per_min - self.disp_rate_nom) / max(0.01, self.disp_rate_crit - self.disp_rate_nom)) * 100.0,
            0.0, 100.0
        )

        # Absolute cumulative displacement subscore (2.0mm baseline, >= 10.0mm critical)
        abs_disp_score = np.clip(
            ((displacement_mm - 2.0) / max(1.0, 10.0 - 2.0)) * 100.0,
            0.0, 100.0
        )

        disp_score = max(rate_score, abs_disp_score)

        return float(vib_score), float(tilt_score), float(disp_score)

    def calculate_risk(
        self,
        vibration_rms: float,
        tilt_magnitude: float,
        displacement_rate_mm_per_min: float,
        anomaly_score_norm: float,  # [0.0, 1.0]
        classifier_probs: Optional[Dict[str, float]] = None,
        temporal_trend_slope: float = 0.0,
        spatial_abnormal_pct: float = 0.0,
        weather_features: Optional[Dict[str, float]] = None,
        satellite_features: Optional[Dict[str, float]] = None,
        displacement_mm: float = 0.0
    ) -> Dict[str, Any]:
        """Compute holistic risk score fusing all diagnostic dimensions with physical interlocks."""
        vib_sub, tilt_sub, disp_sub = self.compute_sensor_subscores(
            vibration_rms, tilt_magnitude, displacement_rate_mm_per_min, displacement_mm=displacement_mm
        )

        anomaly_sub = float(np.clip(anomaly_score_norm * 100.0, 0.0, 100.0))
        temporal_sub = float(np.clip(temporal_trend_slope * 50.0, 0.0, 100.0))
        spatial_sub = float(np.clip(spatial_abnormal_pct * 100.0, 0.0, 100.0))

        # Weather subscore (monsoon precipitation accumulation & root-zone saturation)
        wf = weather_features or {}
        rain_72h = float(wf.get("env_rain_cum_72h_mm", wf.get("rain_cum_72h_mm", 20.0)))
        soil_deep = float(wf.get("env_soil_moisture_deep", wf.get("soil_moisture_deep", 0.30)))
        weather_sub = float(np.clip((rain_72h / 120.0) * 60.0 + (soil_deep / 0.55) * 40.0, 0.0, 100.0))

        # Satellite subscore (InSAR subsidence velocity + NDVI tension crack anomaly + thermal anomaly)
        sf = satellite_features or {}
        insar_vel = abs(float(sf.get("sat_insar_velocity_mm_yr", sf.get("sentinel1_insar_velocity_mm_yr", -15.0))))
        ndvi_anom = abs(float(sf.get("sat_ndvi_anomaly", sf.get("sentinel2_ndvi_anomaly", -0.05))))
        thermal_anom = float(sf.get("sat_thermal_anomaly_k", sf.get("landsat_thermal_anomaly_k", 1.5)))
        sat_sub = float(np.clip((insar_vel / 40.0) * 50.0 + (ndvi_anom / 0.30) * 30.0 + (thermal_anom / 6.0) * 20.0, 0.0, 100.0))

        # Classifier probability contribution
        ml_score = 0.0
        if classifier_probs:
            ml_score = (
                classifier_probs.get("NORMAL", 0.0) * 5.0 +
                classifier_probs.get("WARNING", 0.0) * 35.0 +
                classifier_probs.get("HIGH", 0.0) * 65.0 +
                classifier_probs.get("CRITICAL", 0.0) * 92.0
            )

        # Weighted combination
        w = self.weights
        composite_score = (
            w.get("vibration", 0.12) * vib_sub +
            w.get("tilt", 0.16) * tilt_sub +
            w.get("displacement", 0.22) * disp_sub +
            w.get("weather", 0.15) * weather_sub +
            w.get("satellite", 0.15) * sat_sub +
            w.get("anomaly", 0.10) * anomaly_sub +
            w.get("spatial", 0.10) * spatial_sub
        )

        if classifier_probs:
            final_risk = 0.65 * composite_score + 0.35 * ml_score
        else:
            final_risk = composite_score

        # Physical Dominance Override:
        # If absolute physical roof sag is severe (displacement_mm >= 8.0mm or abs_disp_score >= 80%),
        # the roof has undergone major mechanical subsidence. Do not dilute through linear averaging.
        abs_disp_score = float(np.clip(
            ((displacement_mm - 2.0) / max(1.0, 10.0 - 2.0)) * 100.0,
            0.0, 100.0
        ))
        if abs_disp_score >= 80.0:
            final_risk = max(final_risk, abs_disp_score * 0.82)
        elif abs_disp_score >= 50.0:
            final_risk = max(final_risk, abs_disp_score * 0.70)
        elif disp_sub >= 80.0 and (tilt_sub >= 20.0 or vib_sub >= 20.0 or spatial_sub >= 20.0):
            # Dynamic movement corroborated by multiple physical sensing modalities
            final_risk = max(final_risk, disp_sub * 0.75)

        # ====================================================================
        # GEOTECHNICAL PHYSICAL INTERLOCKS (Eliminating False Positives)
        # ====================================================================
        # 1. False Positive Blasting / Drilling Interlock:
        # If there is ZERO strata convergence velocity (< 0.04 mm/min) and ZERO tilt rate,
        # high vibration is an operational acoustic event, NOT strata collapse!
        # Only cap if absolute displacement is also safe (< 3.0 mm)
        if displacement_rate_mm_per_min < 0.04 and tilt_magnitude < 0.6 and abs(temporal_trend_slope) < 0.02 and displacement_mm < 3.0:
            if final_risk > 28.0:
                final_risk = min(final_risk, 28.0)

        # 2. Strata Physical Equilibrium Interlock:
        # Only apply equilibrium cap if displacement is within baseline (< 2.5 mm)
        if (displacement_rate_mm_per_min < 0.25 or abs(temporal_trend_slope) < 0.08) and tilt_magnitude < 0.50 and vibration_rms < 0.20 and displacement_mm < 2.5:
            if final_risk > 24.0:
                final_risk = min(final_risk, 24.0)

        # 3. Critical Confirmation Interlock:
        # True CRITICAL subsidence requires corroborated physical evidence:
        # - Displacement rate >= 0.50 mm/min with deformation/acoustic corroboration OR
        # - Severe cumulative tilt >= 2.5 degrees OR
        # - Severe absolute roof sag >= 8.0 mm OR
        # - Adjacent gallery nodes confirming active spatial anomaly
        if final_risk >= self.thresholds.get("critical_min", 75.0):
            has_disp_velocity = (
                displacement_rate_mm_per_min >= 0.50 and
                (displacement_mm >= 3.5 or tilt_magnitude >= 1.0 or vibration_rms >= 0.25)
            )
            has_tilt_deflection = tilt_magnitude >= 2.5
            has_absolute_sag = displacement_mm >= 8.0
            has_spatial_confirmation = spatial_abnormal_pct >= 0.25
            if not (has_disp_velocity or has_tilt_deflection or has_absolute_sag or has_spatial_confirmation):
                # Downgrade from Critical to High if single sensor lacks physical corroboration
                final_risk = 74.0

        final_risk = float(np.clip(final_risk, 0.0, 100.0))

        # Map to Risk Level
        if final_risk <= self.thresholds.get("normal_max", 24.9):
            risk_level = "NORMAL"
        elif final_risk <= self.thresholds.get("warning_max", 49.9):
            risk_level = "WARNING"
        elif final_risk <= self.thresholds.get("high_max", 74.9):
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        return {
            "risk_score": round(final_risk, 2),
            "risk_level": risk_level,
            "subscores": {
                "vibration_severity": round(vib_sub, 2),
                "tilt_severity": round(tilt_sub, 2),
                "displacement_severity": round(disp_sub, 2),
                "weather_severity": round(weather_sub, 2),
                "satellite_severity": round(sat_sub, 2),
                "anomaly_score": round(anomaly_sub, 2),
                "temporal_trend": round(temporal_sub, 2),
                "spatial_correlation": round(spatial_sub, 2)
            }
        }
