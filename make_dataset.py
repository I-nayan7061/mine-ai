"""Dataset Creation & Feature Engineering Pipeline.
SIH26025 - NexGen | Mine Subsidence Early Warning System

This script generates:
1. Physics-grounded synthetic telemetry dataset (raw time-series stream across 5 ESP32 nodes)
2. Validated and preprocessed multi-domain rolling windows (60s window, 10s stride)
3. Extracted & selected 30 geotechnical features (Vibration, Tilt, Displacement, Spatial Graph)
4. Chronologically split training, validation, and test datasets for model training
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import app_config
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.spatial_features import MineSpatialGraph
from src.feature_selection import FeatureSelector
from src.classifier import chronological_split
from src.train_pipeline import extract_features_from_windows
from src.utils import get_logger, save_json
from simulator.scenario_generator import MineScenarioGenerator

logger = get_logger("mine_ai.make_dataset")


def generate_raw_dataset(
    output_dir: Path,
    num_cycles: int = 4,
    cycle_duration: int = 2000,
    seed: int = 42
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Generate multi-node physics-based synthetic sensor stream CSV and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing MineScenarioGenerator (seed=%d, cycles=%d, duration=%ds)...", seed, num_cycles, cycle_duration)
    
    gen = MineScenarioGenerator(seed=seed)
    
    dfs = []
    base_time = datetime(2026, 9, 10, 6, 0, 0)

    for c_id in range(1, num_cycles + 1):
        logger.info("Generating Cycle %d/%d (%d seconds across 5 nodes)...", c_id, num_cycles, cycle_duration)
        c_df = gen.generate_operational_cycle(c_id, start_time=base_time, duration_seconds=cycle_duration)
        dfs.append(c_df)
        base_time += timedelta(seconds=cycle_duration + 120)

    full_df = pd.concat(dfs, ignore_index=True)
    csv_file = output_dir / "synthetic_mine_subsidence_dataset.csv"
    meta_file = output_dir / "synthetic_mine_subsidence_metadata.json"

    full_df.to_csv(csv_file, index=False)

    metadata = {
        "project": "Mine Subsidence AI (SIH26025)",
        "data_type": "SYNTHETIC",
        "description": "Physics-grounded synthetic geotechnical sensor stream modeling underground coal mine strata deformation across multiple operational cycles.",
        "sensors": {
            "vibration": "Simulated MPU6050 accelerometer magnitude (g), nominal RMS ~0.05g ambient",
            "tilt": "Simulated MPU6050 dual-axis inclinometer (tilt_x, tilt_y in degrees)",
            "displacement": "Simulated VL53L1X Time-of-Flight convergence measurement (mm)"
        },
        "sampling_frequency_hz": 1.0,
        "units": {
            "tilt_x": "degrees",
            "tilt_y": "degrees",
            "vibration": "g (acceleration)",
            "displacement_mm": "millimeters"
        },
        "limitations": "Generated via physical geotechnical differential equations; must be recalibrated with in-situ field sensor measurements.",
        "intended_use": "Model training, anomaly detection benchmark, algorithm comparison, and unit/scenario verification.",
        "total_records": len(full_df),
        "num_cycles": num_cycles,
        "num_nodes": len(gen.nodes),
        "nodes": gen.nodes,
        "class_distribution": full_df["risk_label"].value_counts().to_dict(),
        "time_range": {
            "start": str(full_df["timestamp"].min()),
            "end": str(full_df["timestamp"].max())
        }
    }
    save_json(metadata, meta_file)
    logger.info("Raw dataset successfully saved: %s (%d records)", csv_file, len(full_df))
    return csv_file, meta_file, metadata


def generate_processed_feature_dataset(
    raw_csv_path: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """Process raw sensor streams into multi-domain rolling windows and extract selected features for training."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading & validating raw dataset from %s...", raw_csv_path)
    loader = DataLoader()
    raw_df, quality_report = loader.load_csv(raw_csv_path)
    if not quality_report.is_valid:
        logger.warning("Quality report flagged issues: bound_errors=%d", len(quality_report.bound_errors))

    logger.info("Preprocessing & rolling-window segmentation (60s window, 10s step)...")
    preprocessor = Preprocessor()
    cleaned_df = preprocessor.clean_dataframe(raw_df)
    windows = preprocessor.create_rolling_windows(cleaned_df)
    logger.info("Generated %d rolling windows.", len(windows))

    logger.info("Extracting 69 multi-domain geotechnical features across windows...")
    spatial_graph = MineSpatialGraph()
    X_full, y_full, t_series = extract_features_from_windows(windows, spatial_graph)
    logger.info("Extracted %d features for %d windows.", X_full.shape[1], X_full.shape[0])

    # Chronological Split (60% Train, 20% Val, 20% Test)
    data_for_split = X_full.copy()
    data_for_split["timestamp"] = t_series
    data_for_split["risk_label"] = y_full

    train_data, val_data, test_data = chronological_split(data_for_split, train_ratio=0.60, val_ratio=0.20)

    X_train_raw = train_data.drop(columns=["timestamp", "risk_label"])
    y_train = train_data["risk_label"]
    X_val_raw = val_data.drop(columns=["timestamp", "risk_label"])
    y_val = val_data["risk_label"]
    X_test_raw = test_data.drop(columns=["timestamp", "risk_label"])
    y_test = test_data["risk_label"]

    logger.info("Selecting top 30 non-collinear geotechnical features...")
    selector = FeatureSelector()
    selected_cols = selector.fit(X_train_raw, y_train)

    X_train = selector.transform(X_train_raw)
    X_val = selector.transform(X_val_raw)
    X_test = selector.transform(X_test_raw)
    X_all_selected = selector.transform(X_full)

    # Prepare export DataFrames
    train_df = X_train.copy()
    train_df["risk_label"] = y_train.values
    train_df["timestamp"] = train_data["timestamp"].values

    val_df = X_val.copy()
    val_df["risk_label"] = y_val.values
    val_df["timestamp"] = val_data["timestamp"].values

    test_df = X_test.copy()
    test_df["risk_label"] = y_test.values
    test_df["timestamp"] = test_data["timestamp"].values

    full_featured_df = X_all_selected.copy()
    full_featured_df["risk_label"] = y_full.values
    full_featured_df["timestamp"] = t_series.values

    # File paths
    train_csv = output_dir / "train_features.csv"
    val_csv = output_dir / "val_features.csv"
    test_csv = output_dir / "test_features.csv"
    full_csv = output_dir / "featured_training_dataset.csv"
    meta_json = output_dir / "processed_metadata.json"

    # Save CSVs
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    full_featured_df.to_csv(full_csv, index=False)

    proc_metadata = {
        "description": "Preprocessed rolling-window feature dataset for underground mine strata risk classification and anomaly detection.",
        "window_size_seconds": 60,
        "step_size_seconds": 10,
        "sampling_rate_hz": 1.0,
        "total_windows": len(full_featured_df),
        "feature_count": len(selected_cols),
        "selected_features": selected_cols,
        "target_classes": ["NORMAL", "WARNING", "HIGH", "CRITICAL"],
        "splits": {
            "train": {
                "records": len(train_df),
                "file": train_csv.name,
                "class_distribution": train_df["risk_label"].value_counts().to_dict()
            },
            "val": {
                "records": len(val_df),
                "file": val_csv.name,
                "class_distribution": val_df["risk_label"].value_counts().to_dict()
            },
            "test": {
                "records": len(test_df),
                "file": test_csv.name,
                "class_distribution": test_df["risk_label"].value_counts().to_dict()
            }
        },
        "created_at": datetime.utcnow().isoformat()
    }
    save_json(proc_metadata, meta_json)

    logger.info("Processed feature datasets exported successfully:")
    logger.info("  - Full Dataset: %s (%d rows, %d features)", full_csv, len(full_featured_df), len(selected_cols))
    logger.info("  - Train Set:    %s (%d rows)", train_csv, len(train_df))
    logger.info("  - Val Set:      %s (%d rows)", val_csv, len(val_df))
    logger.info("  - Test Set:     %s (%d rows)", test_csv, len(test_df))

    return proc_metadata


def print_dataset_summary(raw_meta: Dict[str, Any], proc_meta: Dict[str, Any]):
    """Print a clean visual summary of the generated datasets."""
    print("\n" + "=" * 78)
    print("      SIH26025 - NexGen | MINE SUBSIDENCE DATASET SUITE")
    print("=" * 78)
    print(f"Project:               {raw_meta.get('project', 'Mine Subsidence AI')}")
    print(f"Data Provenance:       {raw_meta.get('data_type', 'SYNTHETIC')} (Physics-Grounded Simulation)")
    print(f"Monitored Nodes:       {', '.join(raw_meta.get('nodes', []))} (5 ESP32 gallery nodes)")
    print(f"Operational Cycles:    {raw_meta.get('num_cycles', 4)} full shifts")
    print(f"Total Raw Telemetry:   {raw_meta.get('total_records', 0):,} records (1.0 Hz)")
    print(f"Raw Time Range:        {raw_meta.get('time_range', {}).get('start')} to {raw_meta.get('time_range', {}).get('end')}")
    
    print("\n--- Raw Telemetry Class Distribution ---")
    for cls, cnt in raw_meta.get("class_distribution", {}).items():
        pct = (cnt / raw_meta["total_records"]) * 100.0
        print(f"  * {cls:<10}: {cnt:>6,} samples ({pct:5.1f}%)")

    if proc_meta:
        print("\n--- Processed Feature Matrices (60s Windows, 10s Step) ---")
        print(f"Total Feature Windows: {proc_meta.get('total_windows', 0):,} windows")
        print(f"Selected Features:     {proc_meta.get('feature_count', 0)} indicators (from 69 multi-domain candidates)")
        splits = proc_meta.get("splits", {})
        print(f"  * Training Set:      {splits.get('train', {}).get('records', 0):,} samples (60% chronological)")
        print(f"  * Validation Set:    {splits.get('val', {}).get('records', 0):,} samples (20% chronological)")
        print(f"  * Test Set:          {splits.get('test', {}).get('records', 0):,} samples (20% chronological)")

        print("\n--- Key Geotechnical Feature Indicators ---")
        feats = proc_meta.get("selected_features", [])
        for i in range(0, min(len(feats), 18), 3):
            sub = feats[i:i+3]
            print("  " + " | ".join(f"{f:<25}" for f in sub))
        if len(feats) > 18:
            print(f"  ... and {len(feats) - 18} more features.")

    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create synthetic mine subsidence dataset and feature matrices.")
    parser.add_argument("--cycles", type=int, default=4, help="Number of operational cycles (default: 4)")
    parser.add_argument("--duration", type=int, default=2000, help="Duration in seconds per cycle (default: 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--raw-dir", type=str, default=None, help="Directory for raw telemetry CSV (default: data/synthetic)")
    parser.add_argument("--processed-dir", type=str, default=None, help="Directory for processed features (default: data/processed)")
    parser.add_argument("--no-process", action="store_true", help="Skip feature extraction and preprocessing")
    parser.add_argument("--train", action="store_true", help="Automatically trigger model training pipeline after dataset creation")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else app_config.resolve_path("data/synthetic")
    processed_dir = Path(args.processed_dir) if args.processed_dir else app_config.resolve_path("data/processed")

    logger.info("Starting dataset generation pipeline...")
    raw_csv, raw_meta_file, raw_meta = generate_raw_dataset(
        output_dir=raw_dir,
        num_cycles=args.cycles,
        cycle_duration=args.duration,
        seed=args.seed
    )

    proc_meta = None
    if not args.no_process:
        proc_meta = generate_processed_feature_dataset(
            raw_csv_path=raw_csv,
            output_dir=processed_dir
        )

    print_dataset_summary(raw_meta, proc_meta)

    if args.train:
        logger.info("Triggering full model training pipeline on new dataset...")
        from src.train_pipeline import run_full_training_pipeline
        run_full_training_pipeline()


if __name__ == "__main__":
    main()
