"""Utility functions for logging, model serialization, and safe calculations.
SIH26025 - NexGen Mine Subsidence AI System
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np


def get_logger(name: str = "mine_ai", level: int = logging.INFO) -> logging.Logger:
    """Create or return a configured console logger with standard formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger("mine_ai")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers without division by zero or NaN explosion."""
    if denominator == 0.0 or np.isnan(denominator) or np.isinf(denominator):
        return default
    result = numerator / denominator
    return default if (np.isnan(result) or np.isinf(result)) else float(result)


def save_artifact(obj: Any, filepath: str | Path) -> None:
    """Save an object to disk using joblib, ensuring parent directory exists."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, str(path))
    logger.info("Saved artifact to %s", path)


def load_artifact(filepath: str | Path) -> Any:
    """Load an artifact from disk using joblib."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {path}")
    return joblib.load(str(path))


def save_json(data: Dict[str, Any], filepath: str | Path, indent: int = 2) -> None:
    """Save dictionary to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.info("Saved JSON to %s", path)


def load_json(filepath: str | Path) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
