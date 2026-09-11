"""Geotechnical Mine Subsidence Scenario Simulator.
Generates multi-shift time-series sensor streams representing:
- Normal baseline operations
- Gradual roof sag (Warning)
- Accelerated strata convergence (High Risk)
- Impending roof fall / localized collapse (Critical)
- Operational transient disturbances (Passing haulage / Blasting vibrations)

SIH26025 - NexGen | Category: Hardware & Disaster Management
DATA_TYPE = SYNTHETIC (Explicitly marked for research & prototype calibration)
"""

import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.utils import get_logger, save_json

logger = get_logger("mine_ai.simulator")

DATA_TYPE = "SYNTHETIC"
DATASET_METADATA = {
    "project": "Mine Subsidence AI (SIH26025)",
    "data_type": DATA_TYPE,
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
    "intended_use": "Model training, anomaly detection benchmark, algorithm comparison, and unit/scenario verification."
}


class MineScenarioGenerator:
    """Generates synthetic multi-node mine monitoring time-series data across operational shifts."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.nodes = ["N01", "N02", "N03", "N04", "N05"]
        # Node coordinates in gallery (meters)
        self.node_positions = {
            "N01": (0.0, 0.0),    # Intake
            "N02": (25.0, 0.0),   # Junction
            "N03": (50.0, 0.0),   # Longwall face (epicenter of disturbance)
            "N04": (50.0, 30.0),  # Return crosscut
            "N05": (25.0, 30.0)   # Goaf boundary
        }

    def generate_operational_cycle(
        self,
        cycle_id: int,
        start_time: datetime,
        duration_seconds: int = 2400
    ) -> pd.DataFrame:
        """Generate a complete operational shift/cycle spanning:
        
        0% - 35%: Normal stable strata
        35% - 60%: Gradual bed separation & convergence (Warning)
        60% - 80%: Accelerated strata deformation (High Risk)
        80% - 95%: Severe fracture / roof collapse dynamics (Critical)
        95% - 100%: Transient post-collapse stabilization / blast noise
        """
        records = []
        t = np.arange(duration_seconds)
        norm_t = t / float(duration_seconds)

        # Epicenter node is N03 (active excavation face)
        for node_id in self.nodes:
            pos = self.node_positions[node_id]
            dist_to_epicenter = math.hypot(pos[0] - 50.0, pos[1] - 0.0)
            spatial_attenuation = math.exp(-dist_to_epicenter / 35.0)

            # Random baseline offsets per cycle and node
            base_disp_val = 1.0 + self.rng.uniform(-0.1, 0.1)
            base_tilt_x_val = 0.10 + self.rng.uniform(-0.03, 0.03)
            base_tilt_y_val = 0.08 + self.rng.uniform(-0.03, 0.03)

            disp = np.full(duration_seconds, base_disp_val) + self.rng.normal(0, 0.015, size=duration_seconds)
            tilt_x = np.full(duration_seconds, base_tilt_x_val) + self.rng.normal(0, 0.015, size=duration_seconds)
            tilt_y = np.full(duration_seconds, base_tilt_y_val) + self.rng.normal(0, 0.015, size=duration_seconds)
            vib = np.abs(self.rng.normal(0.035, 0.012, size=duration_seconds))

            labels = []

            for i in range(duration_seconds):
                tau = norm_t[i]
                if tau < 0.35:
                    # Phase 1: NORMAL
                    labels.append("NORMAL")
                elif tau < 0.60:
                    # Phase 2: WARNING (Slow creep)
                    creep_factor = (tau - 0.35) / 0.25
                    disp[i] += (0.8 * creep_factor) * spatial_attenuation
                    tilt_x[i] += (0.6 * creep_factor) * spatial_attenuation
                    tilt_y[i] += (0.4 * creep_factor) * spatial_attenuation
                    if self.rng.random() > 0.95:
                        vib[i] += self.rng.uniform(0.1, 0.25) * spatial_attenuation
                    labels.append("WARNING" if spatial_attenuation > 0.35 else "NORMAL")
                elif tau < 0.80:
                    # Phase 3: HIGH (Accelerated convergence)
                    high_factor = (tau - 0.60) / 0.20
                    disp[i] += (0.8 + 3.2 * (high_factor ** 1.5)) * spatial_attenuation
                    tilt_x[i] += (0.6 + 2.2 * (high_factor ** 1.3)) * spatial_attenuation
                    tilt_y[i] += (0.4 + 1.5 * (high_factor ** 1.3)) * spatial_attenuation
                    vib[i] += (self.rng.exponential(0.08) + 0.15 * high_factor) * spatial_attenuation
                    labels.append("HIGH" if spatial_attenuation > 0.40 else "WARNING")
                elif tau < 0.95:
                    # Phase 4: CRITICAL (Dynamic tensile fracturing & rapid roof sag)
                    crit_factor = (tau - 0.80) / 0.15
                    collapse_jump = 1.0 / (1.0 + np.exp(-12.0 * (crit_factor - 0.5)))
                    disp[i] += (4.0 + 12.0 * collapse_jump) * spatial_attenuation
                    tilt_x[i] += (2.8 + 4.5 * collapse_jump) * spatial_attenuation
                    tilt_y[i] += (1.9 + 3.2 * collapse_jump) * spatial_attenuation
                    vib_shock = np.abs(self.rng.normal(0, 0.55)) * (collapse_jump + 0.2)
                    vib[i] += vib_shock * spatial_attenuation
                    labels.append("CRITICAL" if spatial_attenuation > 0.35 else "HIGH")
                else:
                    # Phase 5: Stabilization with occasional haulage blast vibration
                    disp[i] += 16.0 * spatial_attenuation
                    tilt_x[i] += 7.3 * spatial_attenuation
                    tilt_y[i] += 5.1 * spatial_attenuation
                    vib[i] += np.abs(self.rng.normal(0.05, 0.02))
                    labels.append("CRITICAL" if spatial_attenuation > 0.35 else "HIGH")

            # Assemble rows
            for i in range(duration_seconds):
                curr_t = start_time + timedelta(seconds=int(t[i]))
                records.append({
                    "timestamp": curr_t.isoformat(),
                    "node_id": node_id,
                    "tilt_x": round(float(tilt_x[i]), 4),
                    "tilt_y": round(float(tilt_y[i]), 4),
                    "vibration": round(float(max(0.0, vib[i])), 4),
                    "displacement_mm": round(float(disp[i]), 4),
                    "cycle_id": f"Cycle_{cycle_id:02d}",
                    "risk_label": labels[i],
                    "data_type": DATA_TYPE
                })

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["timestamp", "node_id"]).reset_index(drop=True)
        return df

    def generate_full_benchmark_dataset(self, output_dir: str | Path, num_cycles: int = 4) -> Tuple[Path, Dict[str, Any]]:
        """Generate multi-cycle dataset allowing chronological train/val/test splits with all classes."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        dfs = []
        base_time = datetime(2026, 9, 10, 6, 0, 0)
        cycle_duration = 2000  # 2000 seconds per cycle across 5 nodes = 10,000 samples per cycle

        for c_id in range(1, num_cycles + 1):
            logger.info("Generating Operational Cycle %d/%d (%d seconds)...", c_id, num_cycles, cycle_duration)
            c_df = self.generate_operational_cycle(c_id, start_time=base_time, duration_seconds=cycle_duration)
            dfs.append(c_df)
            base_time += timedelta(seconds=cycle_duration + 120)

        full_df = pd.concat(dfs, ignore_index=True)
        csv_file = out_path / "synthetic_mine_subsidence_dataset.csv"
        meta_file = out_path / "synthetic_mine_subsidence_metadata.json"

        full_df.to_csv(csv_file, index=False)

        metadata = dict(DATASET_METADATA)
        metadata.update({
            "total_records": len(full_df),
            "num_cycles": num_cycles,
            "num_nodes": len(self.nodes),
            "nodes": self.nodes,
            "class_distribution": full_df["risk_label"].value_counts().to_dict(),
            "time_range": {
                "start": str(full_df["timestamp"].min()),
                "end": str(full_df["timestamp"].max())
            }
        })
        save_json(metadata, meta_file)
        logger.info("Saved synthetic benchmark dataset (%d rows) to %s", len(full_df), csv_file)
        return csv_file, metadata


if __name__ == "__main__":
    gen = MineScenarioGenerator(seed=42)
    csv_f, meta = gen.generate_full_benchmark_dataset("mine-ai/data/synthetic", num_cycles=4)
    print(f"Generated {meta['total_records']} records across {meta['num_cycles']} cycles.")
    print(f"Class distribution: {meta['class_distribution']}")
