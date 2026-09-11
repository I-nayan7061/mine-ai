"""Tilt and Inclinometer Feature Extraction Module.
SIH26025 - NexGen | Dual-Axis Ground/Roof Strata Tilt Analysis

Calculates tilt magnitude, angular velocities, angular accelerations,
trend slopes, goodness-of-fit R², and strata stability metrics with noise deadband.
"""

from typing import Dict, Optional
import numpy as np
from scipy import stats
from .signal_processing import apply_rolling_median_filter
from .utils import safe_divide


def extract_tilt_features(
    tilt_x: np.ndarray,
    tilt_y: np.ndarray,
    time_seconds: Optional[np.ndarray] = None,
    prefix: str = "tilt_"
) -> Dict[str, float]:
    """Extract tilt and inclination features from dual-axis sensor records."""
    n = len(tilt_x)
    if n == 0 or len(tilt_y) == 0:
        default_keys = [
            "x_current", "y_current", "mag_current", "mag_mean", "mag_std",
            "mag_min", "mag_max", "mag_range", "rate_mean", "rate_max",
            "accel_max", "trend_slope", "trend_r2", "sudden_change_max",
            "net_angular_change", "stability_index"
        ]
        return {f"{prefix}{k}": 0.0 for k in default_keys}

    tx = np.nan_to_num(tilt_x, nan=0.0)
    ty = np.nan_to_num(tilt_y, nan=0.0)

    # 1. Tilt Magnitude: sqrt(tilt_x^2 + tilt_y^2)
    mag = np.sqrt(tx ** 2 + ty ** 2)

    # Median smoothing for rate derivative stability
    if n >= 5:
        smoothed_mag = apply_rolling_median_filter(mag, window_size=5)
    else:
        smoothed_mag = mag.copy()

    current_x = float(tx[-1])
    current_y = float(ty[-1])
    current_mag = float(mag[-1])

    mag_mean = float(np.mean(mag))
    mag_std = float(np.std(smoothed_mag, ddof=1 if n > 1 else 0))
    mag_min = float(np.min(mag))
    mag_max = float(np.max(mag))
    mag_range = float(mag_max - mag_min)
    net_change = float(smoothed_mag[-1] - smoothed_mag[0])
    # Directional vector delta accounting for reciprocal rocking & shear reversals
    net_vector_change = float(np.hypot(tx[-1] - tx[0], ty[-1] - ty[0]))

    # 2. Time-series derivatives (rate and acceleration)
    if time_seconds is None or len(time_seconds) != n:
        dt = 1.0
        t_vec = np.arange(n, dtype=float)
    else:
        t_vec = np.array(time_seconds, dtype=float)

    if n > 3:
        diff_t = np.diff(t_vec)
        diff_t = np.where(diff_t <= 0.0, 1.0, diff_t)
        diff_mag = np.diff(smoothed_mag)
        rates = (diff_mag / diff_t) * 60.0  # deg/min

        # Noise deadband on tilt rates: < 0.03 deg/min is IMU temperature drift
        rates_deadband = np.where(np.abs(rates) < 0.03, 0.0, rates)

        rate_mean = float(np.mean(rates_deadband))
        rate_max = float(np.max(np.abs(rates_deadband)))
        sudden_change = float(np.max(np.abs(diff_mag)))

        if len(rates) > 1:
            accels = (np.diff(rates_deadband) / diff_t[:-1]) * 60.0  # deg/min^2
            accel_max = float(np.max(np.abs(accels)))
        else:
            accel_max = 0.0
    else:
        rate_mean = 0.0
        rate_max = 0.0
        sudden_change = 0.0
        accel_max = 0.0

    # 3. Linear Trend Slope and R^2
    if n > 2 and (t_vec[-1] > t_vec[0]):
        slope, intercept, r_val, p_val, std_err = stats.linregress(t_vec, smoothed_mag)
        trend_slope = float(slope) * 60.0  # deg/min
        if abs(trend_slope) < 0.02:
            trend_slope = 0.0
        trend_r2 = float(r_val ** 2) if not np.isnan(r_val) else 0.0
    else:
        trend_slope = 0.0
        trend_r2 = 0.0

    # 4. Strata Stability Index: [0, 1]
    stability = 1.0 / (1.0 + mag_std + abs(trend_slope) + (rate_max / 10.0))

    return {
        f"{prefix}x_current": round(current_x, 4),
        f"{prefix}y_current": round(current_y, 4),
        f"{prefix}mag_current": round(current_mag, 4),
        f"{prefix}mag_mean": round(mag_mean, 4),
        f"{prefix}mag_std": round(mag_std, 4),
        f"{prefix}mag_min": round(mag_min, 4),
        f"{prefix}mag_max": round(mag_max, 4),
        f"{prefix}mag_range": round(mag_range, 4),
        f"{prefix}rate_mean": round(rate_mean, 5),
        f"{prefix}rate_max": round(rate_max, 5),
        f"{prefix}accel_max": round(accel_max, 5),
        f"{prefix}trend_slope": round(trend_slope, 5),
        f"{prefix}trend_r2": round(trend_r2, 4),
        f"{prefix}sudden_change_max": round(sudden_change, 4),
        f"{prefix}net_angular_change": round(net_change, 4),
        f"{prefix}net_vector_change": round(net_vector_change, 4),
        f"{prefix}stability_index": round(float(stability), 4)
    }
