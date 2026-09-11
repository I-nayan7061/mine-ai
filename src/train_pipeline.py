"""Leakage-Free Training and Model Evaluation Orchestrator.
SIH26025 - NexGen | Automated ML Pipeline Training

Execution sequence:
1. Ingest raw multi-cycle dataset.
2. Cycle-isolated rolling window generation (Zero window overlap across splits).
3. Physical-only spatial feature extraction (Zero target label leakage).
4. Feature selection fitted strictly on training data only.
5. StandardScaler fitted strictly on training data only.
6. Isolation Forest fitted on normal baseline training windows.
7. Multi-model supervised benchmark (LR, DT, RF, XGB, LGBM).
8. Comprehensive evaluation on unseen test cycle with High/Critical recall.
9. Persistence of models, standalone scaler, feature dictionary, and metadata.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .anomaly_model import MineAnomalyDetector
from .classifier import CLASS_NAMES, MineRiskClassifier
from .config import app_config
from .data_loader import DataLoader
from .displacement_features import extract_displacement_features
from .feature_selection import FeatureSelector
from .preprocessing import Preprocessor
from .spatial_features import MineSpatialGraph
from .tilt_features import extract_tilt_features
from .environmental_features import extract_environmental_features
from .utils import get_logger, save_artifact, save_json
from .vibration_features import extract_vibration_features

logger = get_logger("mine_ai.train")


def extract_features_from_isolated_windows(
    windows: List[Dict],
    spatial_graph: MineSpatialGraph
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Iterate through windows and build feature matrix, target labels, and timestamps.
    
    STRICT LEAKAGE-FREE GUARANTEE:
    Node states fed to spatial graph contain ONLY physical telemetry (displacement, tilt, vibration).
    NO risk_label, target class, or ground-truth risk score is EVER accessed during feature generation.
    """
    feature_rows = []
    labels = []
    timestamps = []

    # Sort windows chronologically
    windows_sorted = sorted(windows, key=lambda w: w["end_time"])

    # Maintain running PHYSICAL state per node for spatial features
    latest_node_states: Dict[str, Dict[str, float]] = {}

    for win in windows_sorted:
        data = win["data"]
        node_id = win["node_id"]
        end_t = win["end_time"]

        # 1. Vibration features
        vib_vals = data["vibration"].values
        vib_feats = extract_vibration_features(vib_vals, sampling_rate_hz=1.0)

        # 2. Tilt features
        t_sec = (data["timestamp"] - data["timestamp"].iloc[0]).dt.total_seconds().values
        tilt_feats = extract_tilt_features(data["tilt_x"].values, data["tilt_y"].values, time_seconds=t_sec)

        # 3. Displacement features
        disp_feats = extract_displacement_features(data["displacement_mm"].values, time_seconds=t_sec)

        # Update node state for spatial graph - STRICTLY PHYSICAL ONLY
        latest_node_states[node_id] = {
            "displacement_mm": disp_feats["disp_current"],
            "tilt_magnitude": tilt_feats["tilt_mag_current"],
            "vibration_rms": vib_feats["vib_rms"]
        }

        # 4. Spatial features (Purely physical distance-weighted neighbor metrics)
        spatial_feats = spatial_graph.compute_spatial_features(node_id, latest_node_states)

        # 5. Geomechanical Interaction features
        interaction_feats = {
            "geom_dynamic_strain_index": round(float(vib_feats["vib_rms"] + (disp_feats["disp_rate_max"] / 5.0) + (tilt_feats["tilt_rate_max"] / 10.0)), 4),
            "geom_tilt_disp_coupling": round(float(tilt_feats["tilt_rate_max"] * disp_feats["disp_rate_max"]), 4),
            "geom_stability_composite": round(float(tilt_feats["tilt_stability_index"] * disp_feats["disp_stability_index"]), 4)
        }

        # 6. Meteorological & Satellite Earth Observation Features
        w_dict = {}
        s_dict = {}
        if "weather_rain_cum_24h_mm" in data.columns:
            w_dict["rain_rate_mm_h"] = float(data["weather_rain_rate_mm_h"].iloc[-1])
            w_dict["rain_cum_24h_mm"] = float(data["weather_rain_cum_24h_mm"].iloc[-1])
            w_dict["rain_cum_72h_mm"] = float(data["weather_rain_cum_72h_mm"].iloc[-1])
            w_dict["soil_moisture_deep"] = float(data["weather_soil_moisture_deep"].iloc[-1])
            w_dict["ambient_temp_c"] = float(data["weather_ambient_temp_c"].iloc[-1])
        if "satellite_insar_velocity_mm_yr" in data.columns:
            s_dict["sentinel1_insar_velocity_mm_yr"] = float(data["satellite_insar_velocity_mm_yr"].iloc[-1])
            s_dict["sentinel2_ndvi_anomaly"] = float(data["satellite_ndvi_anomaly"].iloc[-1])
            s_dict["landsat_thermal_anomaly_k"] = float(data["satellite_thermal_anomaly_k"].iloc[-1])

        env_feats = extract_environmental_features(w_dict, s_dict, disp_feats)

        # Merge all candidate features
        row = {}
        row.update(vib_feats)
        row.update(tilt_feats)
        row.update(disp_feats)
        row.update(spatial_feats)
        row.update(interaction_feats)
        row.update(env_feats)

        feature_rows.append(row)
        labels.append(win["risk_label"])
        timestamps.append(end_t)

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels, name="risk_label")
    t_series = pd.Series(timestamps, name="timestamp")
    return X, y, t_series


# Alias for backward compatibility
extract_features_from_windows = extract_features_from_isolated_windows


def run_full_training_pipeline():
    """Execute complete end-to-end clean training and evaluation."""
    logger.info("=== STEP 1: Ingesting & Validating Multi-Cycle Dataset ===")
    dataset_path = Path("mine-ai/data/synthetic/synthetic_mine_subsidence_dataset.csv")
    if not dataset_path.exists():
        from simulator.scenario_generator import MineScenarioGenerator
        gen = MineScenarioGenerator(seed=42)
        gen.generate_full_benchmark_dataset("mine-ai/data/synthetic", num_cycles=4)

    loader = DataLoader()
    raw_df, quality_report = loader.load_csv(dataset_path)
    logger.info("Data quality report: %d records, is_valid=%s", quality_report.total_records, quality_report.is_valid)

    logger.info("=== STEP 2: Strict Cycle-Isolated Preprocessing & Windowing ===")
    preprocessor = Preprocessor()
    cleaned_df = preprocessor.clean_dataframe(raw_df)

    # Split raw data by cycle to eliminate ANY window boundary overlap
    train_raw = cleaned_df[cleaned_df["cycle_id"].isin(["Cycle_01", "Cycle_02"])].copy()
    val_raw = cleaned_df[cleaned_df["cycle_id"] == "Cycle_03"].copy()
    test_raw = cleaned_df[cleaned_df["cycle_id"] == "Cycle_04"].copy()

    logger.info("Cycle Raw Splits: Train=%d rows, Val=%d rows, Test=%d rows",
                len(train_raw), len(val_raw), len(test_raw))

    # Generate windows strictly isolated inside each cycle split
    train_windows = preprocessor.create_rolling_windows(train_raw)
    val_windows = preprocessor.create_rolling_windows(val_raw)
    test_windows = preprocessor.create_rolling_windows(test_raw)

    logger.info("Cycle Isolated Windows: Train=%d, Val=%d, Test=%d (Total=%d)",
                len(train_windows), len(val_windows), len(test_windows),
                len(train_windows) + len(val_windows) + len(test_windows))

    logger.info("=== STEP 3: Leakage-Free Feature Extraction ===")
    spatial_graph = MineSpatialGraph()

    X_train_raw, y_train, t_train = extract_features_from_isolated_windows(train_windows, spatial_graph)
    X_val_raw, y_val, t_val = extract_features_from_isolated_windows(val_windows, spatial_graph)
    X_test_raw, y_test, t_test = extract_features_from_isolated_windows(test_windows, spatial_graph)

    logger.info("Extracted %d candidate features across splits.", X_train_raw.shape[1])

    logger.info("=== STEP 4: Feature Selection (Fitted on Training Data ONLY) ===")
    selector = FeatureSelector()
    selected_cols = selector.fit(X_train_raw, y_train)

    X_train = selector.transform(X_train_raw)
    X_val = selector.transform(X_val_raw)
    X_test = selector.transform(X_test_raw)

    models_dir = Path("mine-ai/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path("mine-ai/data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    selector.save(models_dir / "feature_names.json")

    # Save processed feature CSVs
    train_df_out = X_train.copy()
    train_df_out["timestamp"] = t_train.values
    train_df_out["risk_label"] = y_train.values
    train_df_out.to_csv(processed_dir / "train_features.csv", index=False)

    val_df_out = X_val.copy()
    val_df_out["timestamp"] = t_val.values
    val_df_out["risk_label"] = y_val.values
    val_df_out.to_csv(processed_dir / "val_features.csv", index=False)

    test_df_out = X_test.copy()
    test_df_out["timestamp"] = t_test.values
    test_df_out["risk_label"] = y_test.values
    test_df_out.to_csv(processed_dir / "test_features.csv", index=False)

    logger.info("=== STEP 5: Fitting Standalone Scaler on Training Data ONLY ===")
    standalone_scaler = StandardScaler()
    standalone_scaler.fit(X_train)
    save_artifact(standalone_scaler, models_dir / "scaler.pkl")

    logger.info("=== STEP 6: Training Unsupervised Anomaly Model (Isolation Forest) ===")
    normal_mask = y_train == "NORMAL"
    X_train_normal = X_train[normal_mask] if normal_mask.sum() > 50 else X_train
    anomaly_detector = MineAnomalyDetector()
    anomaly_detector.fit(X_train_normal)
    anomaly_detector.save(models_dir / "isolation_forest.pkl")

    is_anom, anom_scores = anomaly_detector.predict(X_test)
    test_abnormal_true = (y_test != "NORMAL").values
    anom_accuracy = float(np.mean(is_anom == test_abnormal_true))
    logger.info("Isolation Forest Test Anomaly Concordance: %.2f%%", anom_accuracy * 100.0)

    logger.info("=== STEP 7: Multi-Model Benchmark & Supervised Training ===")
    candidate_models = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"]
    comparison_results = []
    trained_instances = {}

    for m_type in candidate_models:
        logger.info("Training candidate model: %s...", m_type)
        clf = MineRiskClassifier(model_type=m_type)
        clf.fit(X_train, y_train)

        # Measure model-only predict latency on test set
        X_test_scaled = clf.scaler.transform(X_test)
        start_lat = time.perf_counter()
        for _ in range(10):  # 10 passes for robust timing
            _ = clf.model.predict(X_test_scaled)
        model_only_latency_ms = ((time.perf_counter() - start_lat) / (10 * len(X_test))) * 1000.0

        eval_metrics = clf.evaluate(X_test, y_test)
        eval_metrics["model_only_latency_ms"] = round(model_only_latency_ms, 4)
        comparison_results.append(eval_metrics)
        trained_instances[m_type] = clf

        logger.info("Model %s: Acc=%.4f, High Recall=%.4f, Critical Recall=%.4f, Model Latency=%.4f ms",
                    m_type, eval_metrics["accuracy"], eval_metrics["high_recall"],
                    eval_metrics["critical_recall"], model_only_latency_ms)

    # Save trained model artifacts
    rf_model = trained_instances["random_forest"]
    rf_model.save(models_dir / "random_forest.pkl")

    xgb_model = trained_instances["xgboost"]
    xgb_model.save(models_dir / "xgboost_model.pkl")

    lgb_model = trained_instances["lightgbm"]
    lgb_model.save(models_dir / "lightgbm_model.pkl")

    logger.info("=== STEP 8: Measuring True End-to-End Inference Latency ===")
    from .inference import MineInferencePipeline
    pipe = MineInferencePipeline(models_dir=models_dir)

    # Measure end-to-end processing of 50 sequential readings
    now_t = pd.to_datetime("2026-09-10T12:00:00")
    dummy_payloads = [
        {
            "node_id": "N01",
            "timestamp": str(now_t + pd.Timedelta(seconds=i)),
            "tilt_x": 0.12,
            "tilt_y": 0.08,
            "vibration": 0.04,
            "displacement_mm": 1.05 + (i * 0.001)
        }
        for i in range(50)
    ]
    start_e2e = time.perf_counter()
    for p in dummy_payloads:
        pipe.process_single_reading(p)
    end_to_end_latency_ms = ((time.perf_counter() - start_e2e) / len(dummy_payloads)) * 1000.0
    logger.info("Measured True End-to-End Pipeline Latency: %.2f ms/reading (%.1f req/s)",
                end_to_end_latency_ms, 1000.0 / end_to_end_latency_ms)

    logger.info("=== STEP 9: Persisting Model Metadata & Evaluation Report ===")
    metadata = {
        "dataset": {
            "total_windows": len(train_windows) + len(val_windows) + len(test_windows),
            "train_windows": len(train_windows),
            "val_windows": len(val_windows),
            "test_windows": len(test_windows),
            "num_selected_features": len(selected_cols),
            "selected_features": selected_cols,
            "leakage_free_guarantee": True
        },
        "anomaly_detector": {
            "model_type": "isolation_forest",
            "contamination": anomaly_detector.contamination,
            "test_concordance_accuracy": round(anom_accuracy, 4)
        },
        "benchmark_comparison": comparison_results,
        "selected_primary_model": "random_forest",
        "selected_advanced_model": "xgboost",
        "latencies": {
            "model_only_latency_ms": comparison_results[3]["model_only_latency_ms"],  # XGBoost
            "end_to_end_pipeline_latency_ms": round(end_to_end_latency_ms, 2)
        }
    }
    save_json(metadata, models_dir / "model_metadata.json")
    logger.info("Clean Pipeline Training & Evaluation Successfully Completed! All artifacts saved in %s", models_dir)
    return metadata


if __name__ == "__main__":
    run_full_training_pipeline()
