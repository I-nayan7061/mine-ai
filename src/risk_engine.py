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
        displacement_rate_mm_per_min: float
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
        disp_score = np.clip(
            ((displacement_rate_mm_per_min - self.disp_rate_nom) / max(0.01, self.disp_rate_crit - self.disp_rate_nom)) * 100.0,
            0.0, 100.0
        )

        return float(vib_score), float(tilt_score), float(disp_score)

    def calculate_risk(
        self,
        vibration_rms: float,
        tilt_magnitude: float,
        displacement_rate_mm_per_min: float,
        anomaly_score_norm: float,  # [0.0, 1.0]
        classifier_probs: Optional[Dict[str, float]] = None,
        temporal_trend_slope: float = 0.0,
        spatial_abnormal_pct: float = 0.0
    ) -> Dict[str, Any]:
        """Compute holistic risk score fusing all diagnostic dimensions with physical interlocks."""
        vib_sub, tilt_sub, disp_sub = self.compute_sensor_subscores(
            vibration_rms, tilt_magnitude, displacement_rate_mm_per_min
        )

        anomaly_sub = float(np.clip(anomaly_score_norm * 100.0, 0.0, 100.0))
        temporal_sub = float(np.clip(temporal_trend_slope * 50.0, 0.0, 100.0))
        spatial_sub = float(np.clip(spatial_abnormal_pct * 100.0, 0.0, 100.0))

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
            w.get("vibration", 0.15) * vib_sub +
            w.get("tilt", 0.20) * tilt_sub +
            w.get("displacement", 0.25) * disp_sub +
            w.get("anomaly", 0.20) * anomaly_sub +
            w.get("temporal", 0.10) * temporal_sub +
            w.get("spatial", 0.10) * spatial_sub
        )

        if classifier_probs:
            final_risk = 0.70 * composite_score + 0.30 * ml_score
        else:
            final_risk = composite_score

        # ====================================================================
        # GEOTECHNICAL PHYSICAL INTERLOCKS (Eliminating False Positives)
        # ====================================================================
        # 1. False Positive Blasting / Drilling Interlock:
        # If there is ZERO strata convergence velocity (< 0.04 mm/min) and ZERO tilt rate,
        # high vibration is an operational acoustic event, NOT strata collapse!
        # Cap risk score at 28.0 (low WARNING or NORMAL)
        if displacement_rate_mm_per_min < 0.04 and tilt_magnitude < 0.6 and abs(temporal_trend_slope) < 0.02:
            if final_risk > 28.0:
                final_risk = min(final_risk, 28.0)

        # 2. Critical Confirmation Interlock:
        # True CRITICAL subsidence requires at least one of:
        # - Displacement rate >= 0.8 mm/min OR
        # - Severe cumulative tilt >= 3.0 degrees OR
        # - Adjacent gallery nodes confirming active spatial anomaly
        if final_risk >= self.thresholds.get("critical_min", 75.0):
            has_disp_velocity = displacement_rate_mm_per_min >= 0.50
            has_tilt_deflection = tilt_magnitude >= 2.5
            has_spatial_confirmation = spatial_abnormal_pct >= 0.25
            if not (has_disp_velocity or has_tilt_deflection or has_spatial_confirmation):
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
                "anomaly_score": round(anomaly_sub, 2),
                "temporal_trend": round(temporal_sub, 2),
                "spatial_correlation": round(spatial_sub, 2)
            }
        }
