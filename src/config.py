"""Configuration loader and schema definitions for the Mine Subsidence AI System.
SIH26025 - NexGen
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from .utils import get_logger

logger = get_logger("mine_ai.config")

# Find base project directory
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.yaml"


class Config:
    """Singleton-style or instantiable config wrapper for YAML settings."""

    def __init__(self, config_path: Optional[str | Path] = None):
        if config_path is None:
            self.config_path = DEFAULT_CONFIG_PATH
        else:
            self.config_path = Path(config_path)

        if not self.config_path.exists():
            # Fallback check relative to cwd
            alt_path = Path.cwd() / "mine-ai" / "config.yaml"
            if alt_path.exists():
                self.config_path = alt_path
            else:
                logger.warning("Config file not found at %s. Using default fallback settings.", self.config_path)
                self.data = self._default_fallback()
                return

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f) or {}

    def _default_fallback(self) -> Dict[str, Any]:
        return {
            "system": {"random_seed": 42, "problem_id": "SIH26025"},
            "preprocessing": {"window_size_seconds": 60, "step_size_seconds": 10},
            "risk_engine": {
                "weights": {
                    "vibration": 0.15, "tilt": 0.20, "displacement": 0.25,
                    "anomaly": 0.20, "temporal": 0.10, "spatial": 0.10
                },
                "thresholds": {
                    "normal_max": 24.9, "warning_max": 49.9,
                    "high_max": 74.9, "critical_min": 75.0
                }
            }
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """Access nested config values via dot-notation, e.g. 'risk_engine.weights.vibration'."""
        keys = key_path.split(".")
        current = self.data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    @property
    def window_size_seconds(self) -> int:
        return self.get("preprocessing.window_size_seconds", 60)

    @property
    def step_size_seconds(self) -> int:
        return self.get("preprocessing.step_size_seconds", 10)

    @property
    def nominal_sampling_rate(self) -> float:
        return float(self.get("data.nominal_sampling_rate_hz", 1.0))

    @property
    def risk_weights(self) -> Dict[str, float]:
        return self.get("risk_engine.weights", {
            "vibration": 0.15, "tilt": 0.20, "displacement": 0.25,
            "anomaly": 0.20, "temporal": 0.10, "spatial": 0.10
        })

    @property
    def risk_thresholds(self) -> Dict[str, float]:
        return self.get("risk_engine.thresholds", {
            "normal_max": 24.9, "warning_max": 49.9,
            "high_max": 74.9, "critical_min": 75.0
        })

    @property
    def spatial_graph(self) -> Dict[str, Any]:
        return self.get("spatial_graph", {})

    def resolve_path(self, path_or_key: str | Path) -> Path:
        """Resolve a path relative to project root or from config paths setting."""
        raw_str = str(path_or_key)
        val = self.get(raw_str, None)
        target = Path(val if val else raw_str)
        if target.is_absolute():
            return target
        parts = target.parts
        if parts and parts[0] == "mine-ai":
            stripped = Path(*parts[1:])
        else:
            stripped = target

        cand1 = PROJECT_DIR / stripped
        if cand1.exists():
            return cand1
        cand2 = Path.cwd() / target
        if cand2.exists():
            return cand2
        cand3 = Path.cwd() / stripped
        if cand3.exists():
            return cand3
        return PROJECT_DIR / stripped


# Global singleton instance
app_config = Config()
