"""Production Live Inference Pipeline for Mine Strata Telemetry.
SIH26025 - NexGen | Real-Time Edge/Gateway & Cloud Ingestion Pipeline

Executes the complete end-to-end inference flow:
Raw Sensor Stream -> Data Validation -> Rolling Node Buffer -> Signal Preprocessing
-> Feature Extraction -> Feature Selection -> Anomaly Scoring (Isolation Forest)
-> ML Classification (XGBoost / Random Forest) -> Spatial-Temporal Fusion
-> Continuous Risk Score -> SHAP Factor Attribution -> Structured Alert Response
"""

import collections
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .anomaly_model import MineAnomalyDetector
from .classifier import CLASS_NAMES, MineRiskClassifier
from .config import app_config
from .data_loader import DataLoader
from .displacement_features import extract_displacement_features
from .explainability import MineExplainer
from .risk_engine import RiskEngine
from .spatial_features import MineSpatialGraph
from .tilt_features import extract_tilt_features
from .environmental_features import extract_environmental_features
from .geo_fetcher import get_moonidih_environmental_snapshot
from .utils import get_logger, load_artifact, load_json
from .vibration_features import extract_vibration_features

logger = get_logger("mine_ai.inference")


class NodeBufferManager:
    """Maintains a rolling temporal buffer of raw sensor telemetry per node."""

    def __init__(self, max_records_per_node: int = 300):
        self.max_records = max_records_per_node
        self.buffers: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.max_records)
        )

    def add_reading(self, reading: Dict[str, Any]) -> None:
        """Append validated reading to node's rolling queue."""
        node_id = reading["node_id"]
        ts = reading["timestamp"]
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        if getattr(ts, "tzinfo", None) is not None or getattr(ts, "tz", None) is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)

        entry = {
            "timestamp": ts,
            "node_id": node_id,
            "tilt_x": float(reading["tilt_x"]),
            "tilt_y": float(reading["tilt_y"]),
            "vibration": float(reading["vibration"]),
            "displacement_mm": float(reading["displacement_mm"])
        }
        self.buffers[node_id].append(entry)

    def get_dataframe(self, node_id: str) -> pd.DataFrame:
        """Return buffer contents as a chronologically sorted DataFrame."""
        entries = list(self.buffers.get(node_id, []))
        if not entries:
            return pd.DataFrame(columns=["timestamp", "node_id", "tilt_x", "tilt_y", "vibration", "displacement_mm"])
        df = pd.DataFrame(entries)
        return df.sort_values("timestamp").reset_index(drop=True)

    def clear(self, node_id: Optional[str] = None) -> None:
        if node_id:
            self.buffers[node_id].clear()
        else:
            self.buffers.clear()


class MineInferencePipeline:
    """End-to-End Real-Time Inference Engine."""

    def __init__(self, models_dir: Union[str, Path] = "mine-ai/models", config=None):
        self.config = config or app_config
        self.models_dir = self.config.resolve_path(models_dir)
        self.data_loader = DataLoader(config=self.config)
        self.buffer_manager = NodeBufferManager(
            max_records_per_node=self.config.get("api.buffer_capacity_per_node", 300)
        )
        self.spatial_graph = MineSpatialGraph(config=self.config)
        self.risk_engine = RiskEngine(config=self.config)

        # Active state table across nodes
        self.fleet_states: Dict[str, Dict[str, Any]] = {}

        # Load persisted models & feature schema
        self._load_models()

    def _load_models(self) -> None:
        """Load trained model artifacts from disk."""
        feat_path = self.models_dir / "feature_names.json"
        if feat_path.exists():
            feat_dict = load_json(feat_path)
            self.selected_features = feat_dict.get("selected_features", [])
        else:
            self.selected_features = []

        anom_path = self.models_dir / "isolation_forest.pkl"
        if anom_path.exists():
            self.anomaly_detector = MineAnomalyDetector.load(anom_path, config=self.config)
        else:
            self.anomaly_detector = None
            logger.warning("Anomaly detector artifact not found at %s", anom_path)

        xgb_path = self.models_dir / "xgboost_model.pkl"
        rf_path = self.models_dir / "random_forest.pkl"

        if xgb_path.exists():
            self.classifier = MineRiskClassifier.load(xgb_path, config=self.config)
            logger.info("Loaded primary classifier: XGBoost.")
        elif rf_path.exists():
            self.classifier = MineRiskClassifier.load(rf_path, config=self.config)
            logger.info("Loaded primary classifier: Random Forest.")
        else:
            self.classifier = None
            logger.warning("No classifier artifact found.")

        if self.classifier is not None and self.selected_features:
            scaler = getattr(self.classifier, "scaler", None)
            self.explainer = MineExplainer(self.classifier.model, self.selected_features, scaler=scaler)
        else:
            self.explainer = None

    def process_single_reading(self, raw_reading: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming live sensor reading and return instantaneous comprehensive risk diagnosis."""
        is_valid, validation_errors = self.data_loader.validate_single_reading(raw_reading)
        if not is_valid:
            logger.warning("Validation failure for payload from node %s: %s", raw_reading.get("node_id"), validation_errors)
            return {
                "success": False,
                "node_id": raw_reading.get("node_id", "UNKNOWN"),
                "timestamp": str(raw_reading.get("timestamp", datetime.now(timezone.utc).isoformat())),
                "error": "Validation failed",
                "validation_errors": validation_errors,
                "risk_level": "WARNING",
                "risk_score": 30.0
            }

        node_id = raw_reading["node_id"]
        self.buffer_manager.add_reading(raw_reading)
        df_buf = self.buffer_manager.get_dataframe(node_id)

        # Multi-Modal Environmental & Satellite Resolution for Indian Mining Footprint
        weather_in = raw_reading.get("weather")
        sat_in = raw_reading.get("satellite")
        if not weather_in or not sat_in:
            env_snap = get_moonidih_environmental_snapshot()
            if not weather_in:
                weather_in = env_snap.get("weather", {})
            if not sat_in:
                sat_in = env_snap.get("satellite", {})

        weather_summary = {
            "rain_cum_24h_mm": float(weather_in.get("rain_cum_24h_mm", 14.2)),
            "rain_cum_72h_mm": float(weather_in.get("rain_cum_72h_mm", 48.5)),
            "soil_moisture_deep": float(weather_in.get("soil_moisture_deep", 0.36))
        }
        satellite_summary = {
            "insar_velocity_mm_yr": float(sat_in.get("sentinel1_insar_velocity_mm_yr", -18.4)),
            "ndvi_anomaly": float(sat_in.get("sentinel2_ndvi_anomaly", -0.14)),
            "thermal_anomaly_k": float(sat_in.get("landsat_thermal_anomaly_k", 3.4))
        }

        # Warm-up phase: check for emergency instantaneous thresholds even if buffer is small
        raw_disp = float(raw_reading["displacement_mm"])
        raw_vib = float(raw_reading["vibration"])
        raw_tilt_mag = float(np.sqrt(float(raw_reading["tilt_x"]) ** 2 + float(raw_reading["tilt_y"]) ** 2))

        # Check for immediate critical strata failure during cold-start
        is_instant_critical = raw_disp >= 10.0 or raw_tilt_mag >= 5.0 or (raw_vib >= 2.5 and raw_disp >= 3.0)
        is_instant_high = (raw_disp >= 4.0 or raw_tilt_mag >= 2.5) and not is_instant_critical

        if len(df_buf) < 5 and not (is_instant_critical or is_instant_high):
            return {
                "success": True,
                "node_id": node_id,
                "timestamp": str(raw_reading["timestamp"]),
                "sensor_values": {
                    "vibration": round(float(raw_reading["vibration"]), 4),
                    "tilt_x": round(float(raw_reading["tilt_x"]), 4),
                    "tilt_y": round(float(raw_reading["tilt_y"]), 4),
                    "displacement_mm": round(float(raw_reading["displacement_mm"]), 4)
                },
                "status": "BUFFERING",
                "buffer_count": len(df_buf),
                "anomaly": False,
                "anomaly_score": 0.05,
                "risk_score": 10.0,
                "risk_level": "NORMAL",
                "class_probabilities": {"NORMAL": 0.95, "WARNING": 0.05, "HIGH": 0.0, "CRITICAL": 0.0},
                "subscores": {
                    "vibration_severity": 5.0,
                    "tilt_severity": 5.0,
                    "displacement_severity": 5.0,
                    "weather_severity": 15.0,
                    "satellite_severity": 15.0,
                    "anomaly_score": 5.0,
                    "temporal_trend": 0.0,
                    "spatial_correlation": 0.0
                },
                "weather_summary": weather_summary,
                "satellite_summary": satellite_summary,
                "top_contributing_features": ["buffer_warming"],
                "human_explanations": ["Warming up rolling window buffer (gathering initial samples)"],
                "neighbour_anomalies": 0,
                "confidence": 0.95
            }

        if len(df_buf) < 5 and (is_instant_critical or is_instant_high):
            emerg_score = 88.0 if is_instant_critical else 65.0
            emerg_level = "CRITICAL" if is_instant_critical else "HIGH"
            self.fleet_states[node_id] = {
                "displacement_mm": raw_disp,
                "tilt_magnitude": raw_tilt_mag,
                "vibration_rms": raw_vib,
                "is_abnormal": True,
                "risk_score": emerg_score
            }
            return {
                "success": True,
                "node_id": node_id,
                "timestamp": str(raw_reading["timestamp"]),
                "sensor_values": {
                    "vibration": round(float(raw_reading["vibration"]), 4),
                    "tilt_x": round(float(raw_reading["tilt_x"]), 4),
                    "tilt_y": round(float(raw_reading["tilt_y"]), 4),
                    "displacement_mm": round(float(raw_reading["displacement_mm"]), 4)
                },
                "status": "EMERGENCY_INTERLOCK_TRIGGERED",
                "buffer_count": len(df_buf),
                "anomaly": True,
                "anomaly_score": 0.95,
                "risk_score": emerg_score,
                "risk_level": emerg_level,
                "class_probabilities": {
                    "NORMAL": 0.0,
                    "WARNING": 0.05,
                    "HIGH": 0.25 if is_instant_critical else 0.70,
                    "CRITICAL": 0.70 if is_instant_critical else 0.25
                },
                "subscores": {
                    "vibration_severity": 80.0 if is_instant_critical else 60.0,
                    "tilt_severity": 85.0 if is_instant_critical else 65.0,
                    "displacement_severity": 90.0 if is_instant_critical else 70.0,
                    "weather_severity": 20.0,
                    "satellite_severity": 25.0,
                    "anomaly_score": 95.0,
                    "temporal_trend": 50.0,
                    "spatial_correlation": 0.0
                },
                "weather_summary": weather_summary,
                "satellite_summary": satellite_summary,
                "top_contributing_features": ["instantaneous_displacement_exceeded" if raw_disp >= 4.0 else "instantaneous_tilt_exceeded"],
                "human_explanations": [f"Emergency threshold exceeded on cold-start: displacement={raw_disp}mm, tilt={round(raw_tilt_mag,2)}°"],
                "neighbour_anomalies": 0,
                "confidence": 0.98
            }

        # 3. Compute Features
        vib_vals = df_buf["vibration"].values
        vib_feats = extract_vibration_features(vib_vals, sampling_rate_hz=1.0)

        t_sec = (df_buf["timestamp"] - df_buf["timestamp"].iloc[0]).dt.total_seconds().values
        tilt_feats = extract_tilt_features(df_buf["tilt_x"].values, df_buf["tilt_y"].values, time_seconds=t_sec)
        disp_feats = extract_displacement_features(df_buf["displacement_mm"].values, time_seconds=t_sec)

        # Update local fleet state for spatial calculations
        self.fleet_states[node_id] = {
            "displacement_mm": disp_feats["disp_current"],
            "tilt_magnitude": tilt_feats["tilt_mag_current"],
            "vibration_rms": vib_feats["vib_rms"],
            "is_abnormal": False,
            "risk_score": 10.0
        }

        spatial_feats = self.spatial_graph.compute_spatial_features(node_id, self.fleet_states)

        interaction_feats = {
            "geom_dynamic_strain_index": round(float(vib_feats["vib_rms"] + (disp_feats["disp_rate_max"] / 5.0) + (tilt_feats["tilt_rate_max"] / 10.0)), 4),
            "geom_tilt_disp_coupling": round(float(tilt_feats["tilt_rate_max"] * disp_feats["disp_rate_max"]), 4),
            "geom_stability_composite": round(float(tilt_feats["tilt_stability_index"] * disp_feats["disp_stability_index"]), 4)
        }

        # Multi-Modal Environmental & Satellite Enrichment for Indian Mining Footprint
        weather_in = raw_reading.get("weather")
        sat_in = raw_reading.get("satellite")
        if not weather_in or not sat_in:
            env_snap = get_moonidih_environmental_snapshot()
            if not weather_in:
                weather_in = env_snap.get("weather", {})
            if not sat_in:
                sat_in = env_snap.get("satellite", {})

        env_feats = extract_environmental_features(weather_in, sat_in, disp_feats)

        row = {}
        row.update(vib_feats)
        row.update(tilt_feats)
        row.update(disp_feats)
        row.update(spatial_feats)
        row.update(interaction_feats)
        row.update(env_feats)

        feature_df = pd.DataFrame([row])

        for f in self.selected_features:
            if f not in feature_df.columns:
                feature_df[f] = 0.0
        X_infer = feature_df[self.selected_features]

        # 4. Anomaly Detection
        if self.anomaly_detector is not None:
            is_anom_arr, anom_scores = self.anomaly_detector.predict(X_infer)
            is_anomaly = bool(is_anom_arr[0])
            anomaly_score = float(anom_scores[0])
        else:
            is_anomaly = False
            anomaly_score = 0.05

        # 5. Supervised Classification
        if self.classifier is not None:
            preds, probs = self.classifier.predict(X_infer)
            predicted_class = str(preds[0])
            class_probs = {CLASS_NAMES[i]: round(float(probs[0, i]), 4) for i in range(len(CLASS_NAMES))}
            confidence = round(float(np.max(probs[0])), 4)
        else:
            predicted_class = "NORMAL"
            class_probs = {"NORMAL": 0.90, "WARNING": 0.10, "HIGH": 0.0, "CRITICAL": 0.0}
            confidence = 0.90

        # 6. Geotechnical Rate Calculation:
        # Use sustained slope rate rather than 1-second white-noise spikes
        sustained_disp_rate = max(abs(disp_feats["disp_trend_slope"]), abs(disp_feats["disp_rate_mean"]))

        # Risk Engine Multi-Modal Fusion
        risk_result = self.risk_engine.calculate_risk(
            vibration_rms=vib_feats["vib_rms"],
            tilt_magnitude=tilt_feats["tilt_mag_current"],
            displacement_rate_mm_per_min=sustained_disp_rate,
            anomaly_score_norm=anomaly_score,
            classifier_probs=class_probs,
            temporal_trend_slope=disp_feats["disp_trend_slope"],
            spatial_abnormal_pct=spatial_feats.get("spatial_pct_abnormal_neighbors", 0.0),
            weather_features=env_feats,
            satellite_features=env_feats,
            displacement_mm=disp_feats["disp_current"]
        )

        final_risk_score = risk_result["risk_score"]
        final_risk_level = risk_result["risk_level"]

        self.fleet_states[node_id]["is_abnormal"] = final_risk_level in ["WARNING", "HIGH", "CRITICAL"]
        self.fleet_states[node_id]["risk_score"] = final_risk_score

        # 7. Explainability
        if self.explainer is not None and final_risk_level in ["WARNING", "HIGH", "CRITICAL"]:
            explanation = self.explainer.explain_prediction(X_infer, top_k=4)
        else:
            explanation = {
                "top_factors": ["strata_equilibrium", "vibration_nominal", "displacement_stable"],
                "factor_impacts": {"strata_equilibrium": 100.0},
                "human_explanations": ["Ground conditions are stable within baseline geotechnical tolerances."]
            }

        return {
            "success": True,
            "timestamp": str(raw_reading["timestamp"]),
            "node_id": node_id,
            "sensor_values": {
                "vibration": round(float(raw_reading["vibration"]), 4),
                "tilt_x": round(float(raw_reading["tilt_x"]), 4),
                "tilt_y": round(float(raw_reading["tilt_y"]), 4),
                "displacement_mm": round(float(raw_reading["displacement_mm"]), 4)
            },
            "weather_summary": {
                "rain_cum_24h_mm": env_feats.get("env_rain_cum_24h_mm", 0.0),
                "rain_cum_72h_mm": env_feats.get("env_rain_cum_72h_mm", 0.0),
                "soil_moisture_deep": env_feats.get("env_soil_moisture_deep", 0.35)
            },
            "satellite_summary": {
                "insar_velocity_mm_yr": env_feats.get("sat_insar_velocity_mm_yr", -18.4),
                "ndvi_anomaly": env_feats.get("sat_ndvi_anomaly", -0.14),
                "thermal_anomaly_k": env_feats.get("sat_thermal_anomaly_k", 3.4)
            },
            "anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "risk_score": final_risk_score,
            "risk_level": final_risk_level,
            "class_probabilities": class_probs,
            "subscores": risk_result["subscores"],
            "top_contributing_features": explanation["top_factors"],
            "human_explanations": explanation["human_explanations"],
            "neighbour_anomalies": int(spatial_feats.get("spatial_num_abnormal_neighbors", 0)),
            "confidence": confidence
        }
