"""Displacement and Roof Sag Feature Extraction Module.
SIH26025 - NexGen | Time-of-Flight Strata Convergence & Subsidence Analysis

Calculates convergence metrics, deformation velocities (rates), accelerations,
linear trend slopes, and geotechnical stability indices with noise-robust filtering.
"""

from typing import Dict, Optional
import numpy as np
from scipy import stats
from .signal_processing import apply_rolling_median_filter
from .utils import safe_divide


def extract_displacement_features(
    disp_data: np.ndarray,
    time_seconds: Optional[np.ndarray] = None,
    prefix: str = "disp_"
) -> Dict[str, float]:
    """Extract displacement features from convergence / roof sag measurements."""
    n = len(disp_data)
    if n == 0:
        default_keys = [
            "current", "mean", "median", "std", "min", "max", "range",
            "net_change", "rate_mean", "rate_max", "accel_max", "trend_slope",
            "trend_r2", "sudden_jump_max", "stability_index"
        ]
        return {f"{prefix}{k}": 0.0 for k in default_keys}

    clean_disp = np.nan_to_num(disp_data, nan=0.0)

    # Apply 5-point median filter to suppress high-frequency optical noise
    # while strictly preserving genuine step-like ground displacement edges
    if n >= 5:
        smoothed_disp = apply_rolling_median_filter(clean_disp, window_size=5)
    else:
        smoothed_disp = clean_disp.copy()

    current_val = float(clean_disp[-1])
    mean_val = float(np.mean(clean_disp))
    median_val = float(np.median(clean_disp))
    std_val = float(np.std(smoothed_disp, ddof=1 if n > 1 else 0))
    min_val = float(np.min(clean_disp))
    max_val = float(np.max(clean_disp))
    range_val = float(max_val - min_val)
    net_change = float(smoothed_disp[-1] - smoothed_disp[0])

    # Time-series rate and acceleration calculation
    if time_seconds is None or len(time_seconds) != n:
        t_vec = np.arange(n, dtype=float)
    else:
        t_vec = np.array(time_seconds, dtype=float)

    if n > 3:
        diff_t = np.diff(t_vec)
        diff_t = np.where(diff_t <= 0.0, 1.0, diff_t)
        diff_d = np.diff(smoothed_disp)
        # Rates in mm/min (1 min = 60s)
        rates = (diff_d / diff_t) * 60.0

        # Noise deadband: rates below 0.02 mm/min are sensor jitter
        rates_deadband = np.where(np.abs(rates) < 0.02, 0.0, rates)

        rate_mean = float(np.mean(rates_deadband))
        rate_max = float(np.max(np.abs(rates_deadband)))
        sudden_jump = float(np.max(np.abs(diff_d)))

        if len(rates) > 1:
            accels = (np.diff(rates_deadband) / diff_t[:-1]) * 60.0  # mm/min^2
            accel_max = float(np.max(np.abs(accels)))
        else:
            accel_max = 0.0
    else:
        rate_mean = 0.0
        rate_max = 0.0
        accel_max = 0.0
        sudden_jump = 0.0

    # Trend Slope (mm/min) and R^2 on smoothed values
    if n > 2 and (t_vec[-1] > t_vec[0]):
        slope, intercept, r_val, p_val, std_err = stats.linregress(t_vec, smoothed_disp)
        trend_slope = float(slope) * 60.0  # mm/min
        # Deadband on slope
        if abs(trend_slope) < 0.01:
            trend_slope = 0.0
        trend_r2 = float(r_val ** 2) if not np.isnan(r_val) else 0.0
    else:
        trend_slope = 0.0
        trend_r2 = 0.0

    # Stability Index: [0, 1]
    stability = 1.0 / (1.0 + std_val + abs(trend_slope) + (rate_max / 2.0))

    return {
        f"{prefix}current": round(current_val, 4),
        f"{prefix}mean": round(mean_val, 4),
        f"{prefix}median": round(median_val, 4),
        f"{prefix}std": round(std_val, 4),
        f"{prefix}min": round(min_val, 4),
        f"{prefix}max": round(max_val, 4),
        f"{prefix}range": round(range_val, 4),
        f"{prefix}net_change": round(net_change, 4),
        f"{prefix}rate_mean": round(rate_mean, 5),
        f"{prefix}rate_max": round(rate_max, 5),
        f"{prefix}accel_max": round(accel_max, 5),
        f"{prefix}trend_slope": round(trend_slope, 5),
        f"{prefix}trend_r2": round(trend_r2, 4),
        f"{prefix}sudden_jump_max": round(sudden_jump, 4),
        f"{prefix}stability_index": round(float(stability), 4)
    }
