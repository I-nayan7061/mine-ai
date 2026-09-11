"""Signal Processing Module for Mine Subsidence Sensor Signals.
SIH26025 - NexGen | Vibration, Tilt, and Strata Disturbance Signal Analysis

Implements:
- Butterworth low-pass & band-pass filters
- Rolling median filter (preserves sharp shock edges)
- Fast Fourier Transform (FFT) analysis
- Spectral centroid, spectral energy bands, and spectral entropy
"""

import math
from typing import Dict, Optional, Tuple
import numpy as np
from scipy import signal
from .utils import safe_divide


def apply_butterworth_filter(
    data: np.ndarray,
    cutoff_hz: float = 0.4,
    sampling_rate_hz: float = 1.0,
    order: int = 2,
    btype: str = "low"
) -> np.ndarray:
    """Apply zero-phase Butterworth filter. Falls back safely if signal is too short."""
    if len(data) < order * 3:
        return data.copy()

    nyquist = 0.5 * sampling_rate_hz
    normal_cutoff = min(0.99, max(0.01, cutoff_hz / nyquist))
    try:
        b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
        filtered = signal.filtfilt(b, a, data)
        return filtered
    except Exception:
        # Fallback to moving average if filter instability occurs
        return apply_rolling_median_filter(data, window_size=3)


def apply_rolling_median_filter(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Non-linear median filter that attenuates high-frequency impulsive spikes

    while strictly preserving genuine step-like ground displacement edges.
    """
    if len(data) < window_size or window_size <= 1:
        return data.copy()
    if window_size % 2 == 0:
        window_size += 1
    return signal.medfilt(data, kernel_size=window_size)


def compute_fft_spectrum(
    signal_data: np.ndarray,
    sampling_rate_hz: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one-sided Fast Fourier Transform (FFT) amplitude spectrum."""
    n = len(signal_data)
    if n < 4:
        return np.array([0.0]), np.array([0.0])

    # Detrend to remove DC component
    detrended = signal_data - np.mean(signal_data)
    # Apply Hanning window to mitigate spectral leakage
    window = np.hanning(n)
    windowed = detrended * window

    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
    amplitudes = np.abs(fft_vals) * (2.0 / n)

    return freqs, amplitudes


def compute_spectral_features(
    signal_data: np.ndarray,
    sampling_rate_hz: float = 1.0
) -> Dict[str, float]:
    """Compute key frequency-domain spectral indicators from signal data."""
    freqs, amps = compute_fft_spectrum(signal_data, sampling_rate_hz)
    total_power = np.sum(amps ** 2)

    if total_power <= 1e-12 or len(amps) < 2:
        return {
            "dominant_frequency": 0.0,
            "second_dominant_frequency": 0.0,
            "spectral_centroid": 0.0,
            "spectral_energy_low": 0.0,
            "spectral_energy_mid": 0.0,
            "spectral_energy_high": 0.0,
            "spectral_entropy": 0.0
        }

    # Dominant frequencies
    sorted_indices = np.argsort(amps)[::-1]
    dominant_freq = float(freqs[sorted_indices[0]])
    second_dominant_freq = float(freqs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0

    # Spectral centroid
    spectral_centroid = float(safe_divide(np.sum(freqs * amps), np.sum(amps), default=0.0))

    # Energy bands (Nyquist = 0.5 * sampling_rate_hz)
    nyq = 0.5 * sampling_rate_hz
    low_band_mask = freqs <= (0.2 * nyq)
    mid_band_mask = (freqs > (0.2 * nyq)) & (freqs <= (0.6 * nyq))
    high_band_mask = freqs > (0.6 * nyq)

    energy_low = float(safe_divide(np.sum(amps[low_band_mask] ** 2), total_power))
    energy_mid = float(safe_divide(np.sum(amps[mid_band_mask] ** 2), total_power))
    energy_high = float(safe_divide(np.sum(amps[high_band_mask] ** 2), total_power))

    # Spectral entropy with safe clipping
    norm_amps = amps / np.sum(amps)
    safe_amps = np.clip(norm_amps, 1e-12, 1.0)
    entropy_terms = np.where(norm_amps > 1e-12, norm_amps * np.log2(safe_amps), 0.0)
    spectral_entropy = float(-np.sum(entropy_terms) / np.log2(len(amps))) if len(amps) > 1 else 0.0

    return {
        "dominant_frequency": dominant_freq,
        "second_dominant_frequency": second_dominant_freq,
        "spectral_centroid": spectral_centroid,
        "spectral_energy_low": energy_low,
        "spectral_energy_mid": energy_mid,
        "spectral_energy_high": energy_high,
        "spectral_entropy": spectral_entropy
    }
