"""Vibration Feature Extraction Module.
SIH26025 - NexGen | Underground Mine Subsidence Early Warning System

Calculates comprehensive time-domain, peak-ratio, and spectral frequency-domain features
from vibration and accelerometer streams.
"""

from typing import Dict, Optional
import numpy as np
from scipy import stats

from .signal_processing import compute_spectral_features
from .utils import safe_divide


def extract_vibration_features(
    vib_data: np.ndarray,
    sampling_rate_hz: float = 1.0,
    prefix: str = "vib_"
) -> Dict[str, float]:
    """Extract complete vibration feature set from an array of vibration measurements."""
    features: Dict[str, float] = {}
    n = len(vib_data)

    if n == 0 or np.all(np.isnan(vib_data)):
        # Return zeros for empty input
        default_keys = [
            "mean", "std", "variance", "min", "max", "range", "rms", "peak",
            "peak_to_peak", "mav", "energy", "power", "skewness", "kurtosis",
            "cov", "crest_factor", "impulse_factor", "shape_factor", "clearance_factor",
            "dominant_frequency", "second_dominant_frequency", "spectral_centroid",
            "spectral_energy_low", "spectral_energy_mid", "spectral_energy_high", "spectral_entropy"
        ]
        return {f"{prefix}{k}": 0.0 for k in default_keys}

    clean_vib = np.nan_to_num(vib_data, nan=0.0, posinf=0.0, neginf=0.0)
    abs_vib = np.abs(clean_vib)

    # 1. Basic Statistical Time-Domain Features
    mean_val = float(np.mean(clean_vib))
    std_val = float(np.std(clean_vib, ddof=1 if n > 1 else 0))
    var_val = float(std_val ** 2)
    min_val = float(np.min(clean_vib))
    max_val = float(np.max(clean_vib))
    peak_val = float(np.max(abs_vib))
    p2p_val = float(max_val - min_val)
    mav_val = float(np.mean(abs_vib))  # Mean Absolute Value

    # Root Mean Square (RMS)
    rms_val = float(np.sqrt(np.mean(clean_vib ** 2)))
    # Energy and Power
    energy_val = float(np.sum(clean_vib ** 2))
    power_val = float(np.mean(clean_vib ** 2))

    # Skewness & Kurtosis
    skew_val = float(stats.skew(clean_vib)) if (std_val > 1e-8 and n > 2) else 0.0
    kurt_val = float(stats.kurtosis(clean_vib)) if (std_val > 1e-8 and n > 3) else 0.0
    cov_val = float(safe_divide(std_val, abs(mean_val) + 1e-8, default=0.0))

    # 2. Peak-Related Diagnostic Ratios
    # Crest Factor = Peak / RMS
    crest_factor = float(safe_divide(peak_val, rms_val, default=1.0))
    # Impulse Factor = Peak / MAV
    impulse_factor = float(safe_divide(peak_val, mav_val, default=1.0))
    # Shape Factor = RMS / MAV
    shape_factor = float(safe_divide(rms_val, mav_val, default=1.0))
    # Clearance Factor = Peak / (Mean(sqrt(|x|)))^2
    mean_sqrt = np.mean(np.sqrt(abs_vib))
    clearance_factor = float(safe_divide(peak_val, (mean_sqrt ** 2), default=1.0))

    features.update({
        f"{prefix}mean": round(mean_val, 5),
        f"{prefix}std": round(std_val, 5),
        f"{prefix}variance": round(var_val, 5),
        f"{prefix}min": round(min_val, 5),
        f"{prefix}max": round(max_val, 5),
        f"{prefix}range": round(p2p_val, 5),
        f"{prefix}rms": round(rms_val, 5),
        f"{prefix}peak": round(peak_val, 5),
        f"{prefix}peak_to_peak": round(p2p_val, 5),
        f"{prefix}mav": round(mav_val, 5),
        f"{prefix}energy": round(energy_val, 5),
        f"{prefix}power": round(power_val, 5),
        f"{prefix}skewness": round(skew_val, 5),
        f"{prefix}kurtosis": round(kurt_val, 5),
        f"{prefix}cov": round(cov_val, 5),
        f"{prefix}crest_factor": round(crest_factor, 5),
        f"{prefix}impulse_factor": round(impulse_factor, 5),
        f"{prefix}shape_factor": round(shape_factor, 5),
        f"{prefix}clearance_factor": round(clearance_factor, 5),
    })

    # 3. Frequency-Domain / Spectral Features (via FFT)
    spectral_feats = compute_spectral_features(clean_vib, sampling_rate_hz=sampling_rate_hz)
    for k, v in spectral_feats.items():
        features[f"{prefix}{k}"] = round(v, 5)

    return features
