"""Data Preprocessing and Time Windowing Module (Leakage-Free & Cycle-Isolated).
SIH26025 - NexGen | Underground Mine Strata Monitoring

Handles:
- Chronological ordering per node and operational cycle
- Duplicate removal
- Missing value interpolation and bound clipping
- Rolling time-series window segmentation strictly isolated within each cycle
  (Guarantees zero window overlap across train/val/test cycle boundaries)
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np
import pandas as pd

from .config import app_config
from .signal_processing import apply_rolling_median_filter
from .utils import get_logger

logger = get_logger("mine_ai.preprocessing")


class Preprocessor:
    """Preprocesses raw mine sensor telemetry and constructs rolling temporal windows."""

    def __init__(self, config=None):
        self.config = config or app_config
        self.window_size_sec = self.config.window_size_seconds
        self.step_size_sec = self.config.step_size_seconds
        self.min_samples = self.config.get("preprocessing.min_samples_per_window", 10)
        self.sensor_bounds = self.config.get("data.sensor_bounds", {
            "tilt_x": [-90.0, 90.0],
            "tilt_y": [-90.0, 90.0],
            "vibration": [0.0, 10.0],
            "displacement_mm": [-100.0, 2000.0]
        })

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort, deduplicate, clip bounds, and impute short missing gaps."""
        cleaned = df.copy()
        cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"])

        # Sort by node and timestamp
        sort_cols = ["cycle_id", "node_id", "timestamp"] if "cycle_id" in cleaned.columns else ["node_id", "timestamp"]
        cleaned = cleaned.sort_values(sort_cols).reset_index(drop=True)

        # Deduplicate
        dedup_cols = ["cycle_id", "node_id", "timestamp"] if "cycle_id" in cleaned.columns else ["node_id", "timestamp"]
        initial_len = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=dedup_cols, keep="last").reset_index(drop=True)
        if len(cleaned) < initial_len:
            logger.info("Removed %d duplicate timestamps across nodes.", initial_len - len(cleaned))

        # Process per node / cycle group
        node_dfs = []
        group_cols = ["cycle_id", "node_id"] if "cycle_id" in cleaned.columns else ["node_id"]
        for _, group in cleaned.groupby(group_cols, as_index=False):
            grp = group.copy()
            for sensor, (low, high) in self.sensor_bounds.items():
                if sensor in grp.columns:
                    grp[sensor] = grp[sensor].clip(lower=low, upper=high)
                    grp[sensor] = grp[sensor].interpolate(method="linear", limit=5).bfill().ffill()
            node_dfs.append(grp)

        final_df = pd.concat(node_dfs, ignore_index=True)
        final_df = final_df.sort_values(["timestamp", "node_id"]).reset_index(drop=True)
        return final_df

    def create_rolling_windows(
        self,
        df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Slice the time-series dataframe into rolling windows per cycle and node.
        
        Strict Isolation Guarantee:
        Windows are generated independently within each (cycle_id, node_id) partition.
        No rolling window ever spans across operational shift / cycle boundaries.
        """
        windows = []
        df_copy = df.copy()
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

        group_cols = ["cycle_id", "node_id"] if "cycle_id" in df_copy.columns else ["node_id"]

        for grp_keys, group in df_copy.groupby(group_cols):
            if isinstance(grp_keys, tuple) and len(grp_keys) >= 2:
                cycle_id, node_id = grp_keys[0], grp_keys[1]
            elif isinstance(grp_keys, tuple) and len(grp_keys) == 1:
                cycle_id = "Cycle_01"
                node_id = grp_keys[0]
            else:
                cycle_id = "Cycle_01"
                node_id = grp_keys

            group = group.sort_values("timestamp").reset_index(drop=True)
            if len(group) < self.min_samples:
                continue

            start_t = group["timestamp"].min()
            end_t = group["timestamp"].max()

            curr_window_start = start_t
            while curr_window_start + timedelta(seconds=self.window_size_sec) <= end_t:
                curr_window_end = curr_window_start + timedelta(seconds=self.window_size_sec)

                mask = (group["timestamp"] >= curr_window_start) & (group["timestamp"] < curr_window_end)
                win_data = group[mask]

                if len(win_data) >= self.min_samples:
                    label = win_data["risk_label"].mode()[0] if "risk_label" in win_data.columns else "UNKNOWN"
                    scenario = win_data["scenario"].iloc[-1] if "scenario" in win_data.columns else "unknown"

                    windows.append({
                        "cycle_id": cycle_id,
                        "node_id": node_id,
                        "start_time": curr_window_start,
                        "end_time": curr_window_end,
                        "data": win_data.reset_index(drop=True),
                        "risk_label": label,
                        "scenario": scenario
                    })

                curr_window_start += timedelta(seconds=self.step_size_sec)

        logger.info("Generated %d strictly cycle-isolated rolling windows (size=%ds, step=%ds).",
                    len(windows), self.window_size_sec, self.step_size_sec)
        return windows
