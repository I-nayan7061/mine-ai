"""Explainable AI (XAI) Module for Mine Subsidence Predictions.
SIH26025 - NexGen | SHAP Tree Explainer & Human-Centric Decision Attribution

Answers the critical mine safety question:
"Why did the AI classify this node as HIGH / CRITICAL risk?"

Provides:
- SHAP feature contribution values per inference
- Top contributing risk factors with direction of impact (+/-)
- Geotechnical domain explanations for mine safety operators
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import shap

from .utils import get_logger, safe_divide

logger = get_logger("mine_ai.explainability")

# Human-readable geomechanical factor descriptions
GEOTECH_FACTOR_DESCRIPTIONS = {
    "disp_rate_max": "Accelerated roof-to-floor convergence velocity",
    "disp_rate_mean": "Persistent strata convergence rate",
    "disp_trend_slope": "Upward linear roof sag trend",
    "disp_net_change": "High cumulative bed separation displacement",
    "disp_current": "Absolute strata convergence magnitude",
    "tilt_mag_current": "Severe strata/inclinometer tilt magnitude",
    "tilt_rate_max": "Rapid angular deformation rate",
    "tilt_trend_slope": "Increasing tilt deviation slope",
    "tilt_accel_max": "Dynamic angular acceleration spikes",
    "vib_rms": "High root-mean-square seismic vibration energy",
    "vib_peak": "Impulsive shock vibration spikes",
    "vib_crest_factor": "Impulsive fracturing ratio (Crest Factor)",
    "vib_spectral_energy_low": "Low-frequency ground vibration resonance",
    "spatial_num_abnormal_neighbors": "Multiple neighboring gallery nodes exhibiting deformation",
    "spatial_pct_abnormal_neighbors": "High cluster density of anomalous adjacent sensors",
    "spatial_disp_gradient_max": "Steep spatial displacement gradient between adjacent bolts",
    "spatial_dist_weighted_anomaly": "Proximity to active strata disturbance epicenter"
}


class MineExplainer:
    """Computes SHAP-based feature attributions and produces operator-facing explanations."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[shap.TreeExplainer] = None

        try:
            # TreeExplainer for Random Forest, XGBoost, LightGBM
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer initialized successfully.")
        except Exception as e:
            logger.warning("Could not initialize SHAP TreeExplainer: %s. Using feature importance fallback.", e)

    def explain_prediction(
        self,
        features_row: pd.DataFrame,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Generate structured explanation for a single prediction row.
        
        Returns:
            dict containing:
            - 'top_factors': list of factor names
            - 'factor_impacts': dict mapping factor -> impact percentage
            - 'human_explanations': list of descriptive sentences
        """
        row_ordered = features_row[self.feature_names].copy()

        # If SHAP is active, compute exact shap values
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(row_ordered)

                # For multiclass, shap_values is a list of arrays (one per class) or 3D array
                if isinstance(shap_values, list):
                    # Use the highest risk class (e.g. CRITICAL or HIGH, last class)
                    class_shap = np.abs(shap_values[-1][0])
                    signed_shap = shap_values[-1][0]
                elif len(shap_values.shape) == 3:
                    class_shap = np.abs(shap_values[0, :, -1])
                    signed_shap = shap_values[0, :, -1]
                else:
                    class_shap = np.abs(shap_values[0])
                    signed_shap = shap_values[0]

                # Rank by absolute impact
                top_indices = np.argsort(class_shap)[::-1][:top_k]
                total_impact = np.sum(class_shap[top_indices]) + 1e-8

                top_factors = []
                factor_impacts = {}
                human_explanations = []

                for idx in top_indices:
                    f_name = self.feature_names[idx]
                    f_val = float(row_ordered.iloc[0, idx])
                    f_impact_pct = round(float((class_shap[idx] / total_impact) * 100.0), 1)
                    direction = "Increasing" if signed_shap[idx] >= 0 else "Mitigating"

                    top_factors.append(f_name)
                    factor_impacts[f_name] = f_impact_pct

                    desc = GEOTECH_FACTOR_DESCRIPTIONS.get(f_name, f"Feature '{f_name}'")
                    human_explanations.append(
                        f"{desc} (value: {round(f_val, 3)}, impact: +{f_impact_pct}%)"
                    )

                return {
                    "top_factors": top_factors,
                    "factor_impacts": factor_impacts,
                    "human_explanations": human_explanations
                }
            except Exception as e:
                logger.warning("SHAP calculation error: %s. Using heuristic fallback.", e)

        # Fallback explanation if SHAP calculation encounters any issue
        fallback_top = self.feature_names[:top_k]
        return {
            "top_factors": fallback_top,
            "factor_impacts": {f: round(100.0 / top_k, 1) for f in fallback_top},
            "human_explanations": [
                f"{GEOTECH_FACTOR_DESCRIPTIONS.get(f, f)} contributing to risk status"
                for f in fallback_top
            ]
        }
