"""Data Loader and Quality Validation Module.
SIH26025 - NexGen | Mine Subsidence Monitoring & Early Warning System

Supports:
- Loading CSV datasets (synthetic, external, or live recorded)
- Validating raw sensor inputs against physical bounds
- Checking stuck sensors, unrealistic jumps, missing/duplicate timestamps
- Generating structured Data Quality Reports
"""

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .config import app_config
from .utils import get_logger, safe_divide

logger = get_logger("mine_ai.data_loader")


class DataQualityReport:
    """Encapsulates data quality assessment metrics."""

    def __init__(self):
        self.total_records: int = 0
        self.num_nodes: int = 0
        self.node_ids: List[str] = []
        self.time_range: Dict[str, Optional[str]] = {"start": None, "end": None}
        self.nominal_interval_seconds: float = 1.0
        self.duplicate_records_count: int = 0
        self.missing_values_per_col: Dict[str, int] = {}
        self.invalid_bounds_count: int = 0
        self.stuck_sensors_detected: List[Dict[str, Any]] = []
        self.unrealistic_jumps_detected: List[Dict[str, Any]] = []
        self.node_completeness: Dict[str, float] = {}
        self.sampling_gaps_count: int = 0
        self.is_valid: bool = True
        self.warnings: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "num_nodes": self.num_nodes,
            "node_ids": self.node_ids,
            "time_range": self.time_range,
            "nominal_interval_seconds": self.nominal_interval_seconds,
            "duplicate_records_count": self.duplicate_records_count,
            "missing_values_per_col": self.missing_values_per_col,
            "invalid_bounds_count": self.invalid_bounds_count,
            "stuck_sensors_detected": self.stuck_sensors_detected,
            "unrealistic_jumps_detected": self.unrealistic_jumps_detected,
            "node_completeness": self.node_completeness,
            "sampling_gaps_count": self.sampling_gaps_count,
            "is_valid": self.is_valid,
            "warnings": self.warnings
        }


class DataLoader:
    """Robust data ingestion and schema validation for mine sensor data."""

    def __init__(self, config=None):
        self.config = config or app_config
        self.expected_cols = self.config.get("data.expected_columns", [
            "timestamp", "node_id", "tilt_x", "tilt_y", "vibration", "displacement_mm"
        ])
        self.sensor_bounds = self.config.get("data.sensor_bounds", {
            "tilt_x": [-90.0, 90.0],
            "tilt_y": [-90.0, 90.0],
            "vibration": [0.0, 10.0],
            "displacement_mm": [-100.0, 2000.0]
        })

    def validate_single_reading(self, reading: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single incoming live sensor payload (e.g. from ESP32 via HTTP/MQTT)."""
        issues = []
        # Check required fields
        for col in self.expected_cols:
            if col not in reading or reading[col] is None:
                issues.append(f"Missing required field: {col}")

        if issues:
            return False, issues

        # Check numeric types and bounds
        for sensor, (low, high) in self.sensor_bounds.items():
            val = reading.get(sensor)
            if val is not None:
                try:
                    f_val = float(val)
                    if math.isnan(f_val) or math.isinf(f_val):
                        issues.append(f"Sensor {sensor} has NaN/Inf value: {val}")
                    elif f_val < low or f_val > high:
                        issues.append(f"Sensor {sensor} out of realistic bounds [{low}, {high}]: {f_val}")
                except (ValueError, TypeError):
                    issues.append(f"Sensor {sensor} has non-numeric value: {val}")

        return len(issues) == 0, issues

    def load_csv(self, filepath: Union[str, Path]) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Load and validate a CSV dataset, returning the validated dataframe and quality report."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        logger.info("Loading dataset from %s...", path)
        df = pd.read_csv(path)
        report = self.validate_dataframe(df)
        return df, report

    def validate_dataframe(self, df: pd.DataFrame) -> DataQualityReport:
        """Perform deep data validation and generate a DataQualityReport."""
        report = DataQualityReport()
        report.total_records = len(df)

        # 1. Check required columns
        missing_cols = [c for c in self.expected_cols if c not in df.columns]
        if missing_cols:
            report.is_valid = False
            report.warnings.append(f"Missing mandatory columns: {missing_cols}")
            return report

        # 2. Timestamp formatting
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception as e:
            report.is_valid = False
            report.warnings.append(f"Timestamp parsing failed: {e}")
            return report

        # 3. Nodes and Time Range
        nodes = df["node_id"].dropna().unique().tolist()
        report.num_nodes = len(nodes)
        report.node_ids = nodes
        report.time_range = {
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max())
        }

        # 4. Duplicates
        dup_mask = df.duplicated(subset=["timestamp", "node_id"], keep=False)
        report.duplicate_records_count = int(dup_mask.sum())
        if report.duplicate_records_count > 0:
            report.warnings.append(f"Detected {report.duplicate_records_count} duplicate records for same timestamp & node.")

        # 5. Missing values
        for col in self.expected_cols:
            report.missing_values_per_col[col] = int(df[col].isna().sum())

        # 6. Sensor Bounds & Stuck Sensor Detection
        invalid_bounds = 0
        for node in nodes:
            node_df = df[df["node_id"] == node].sort_values("timestamp")
            report.node_completeness[node] = round(float(len(node_df) / max(1, len(df) / report.num_nodes)), 3)

            for sensor, (low, high) in self.sensor_bounds.items():
                if sensor in node_df.columns:
                    s_vals = node_df[sensor].dropna()
                    oob = (s_vals < low) | (s_vals > high)
                    invalid_bounds += int(oob.sum())

                    # Check stuck sensor: constant reading for 100+ consecutive steps
                    if len(s_vals) > 100:
                        diffs = s_vals.diff().abs()
                        # If diff is exactly 0 for 100 steps
                        rolling_zero = (diffs == 0.0).rolling(100).sum()
                        if (rolling_zero >= 99).any():
                            report.stuck_sensors_detected.append({
                                "node_id": node,
                                "sensor": sensor,
                                "description": "Constant unchanging value detected over 100+ consecutive steps"
                            })

            # Check unrealistic jumps in displacement (> 100 mm in 1 sec)
            if "displacement_mm" in node_df.columns:
                disp_diff = node_df["displacement_mm"].diff().abs()
                huge_jumps = disp_diff > 50.0
                if huge_jumps.any():
                    jump_indices = node_df[huge_jumps].index.tolist()
                    report.unrealistic_jumps_detected.append({
                        "node_id": node,
                        "sensor": "displacement_mm",
                        "jump_count": len(jump_indices),
                        "max_jump": float(disp_diff.max())
                    })

        report.invalid_bounds_count = invalid_bounds
        logger.info("Data Validation Complete: %d records, %d nodes, %d duplicates, %d bound errors.",
                    report.total_records, report.num_nodes, report.duplicate_records_count, report.invalid_bounds_count)
        return report
