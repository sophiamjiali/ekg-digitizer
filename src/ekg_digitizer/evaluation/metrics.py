# ==============================================================================
# File Name : metrics.py
# Purpose   : Competition SNR metric, MAE, and signal alignment
# ==============================================================================
"""Metrics for comparing digitized EKG signals against reference signals.

Follows the official metric of the Kaggle "PhysioNet - Digitization of ECG
Images" competition. See ``docs/metrics.md`` for definitions.
"""

from __future__ import annotations

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
MAX_TIME_SHIFT_S = 0.2
PERFECT_SCORE = 1e6  # cap on the linear signal/noise ratio of one image
MIN_SCORE_DB = -384


def align(x_ref: np.ndarray, x_est: np.ndarray, fs: float) -> np.ndarray:
    """Align ``x_est`` to ``x_ref`` with a time shift and a vertical offset.

    The time shift is the lag within ``+/- MAX_TIME_SHIFT_S`` that maximizes the
    cross-correlation of the mean-centred signals (smallest lag on ties). The
    vertical offset is the least-squares offset over the overlapping samples.
    Samples shifted out of range become NaN.
    """
    x_ref = np.asarray(x_ref, dtype=float)
    x_est = np.asarray(x_est, dtype=float)
    n_ref, n_est = x_ref.size, x_est.size

    correlation = np.correlate(x_ref - x_ref.mean(), x_est - x_est.mean(), mode="full")
    lags = np.arange(-(n_est - 1), n_ref)
    max_shift = int(fs * MAX_TIME_SHIFT_S)
    valid = np.abs(lags) <= max_shift
    best = np.flatnonzero(valid & (correlation == correlation[valid].max()))
    lag = lags[best[np.argmin(np.abs(lags[best]))]]

    aligned = np.concatenate(
        (
            np.full(max(lag, 0), np.nan),
            x_est[max(-lag, 0) : min(n_ref - lag, n_est)],
            np.full(max(n_ref - n_est - lag, 0), np.nan),
        )
    )
    overlap = np.isfinite(aligned)
    if overlap.any():
        aligned -= np.mean(aligned[overlap] - x_ref[overlap])
    return aligned


def power(x_ref: np.ndarray, x_est: np.ndarray) -> tuple[float, float]:
    """Return ``(signal_power, noise_power)`` as sums of squares.

    Non-finite values in ``x_est`` (e.g. from alignment) count as 0.
    """
    x_est = np.where(np.isfinite(x_est), x_est, 0.0)
    return float(np.sum(x_ref**2)), float(np.sum((x_ref - x_est) ** 2))


def snr_ratio(signal: float, noise: float) -> float:
    """Linear signal/noise ratio, capped at ``PERFECT_SCORE``."""
    if noise == 0:
        return PERFECT_SCORE
    if signal == 0:
        return 0.0
    return min(signal / noise, PERFECT_SCORE)


def to_db(ratios: list[float]) -> float:
    """Average linear ratios across images, then convert to dB."""
    with np.errstate(divide="ignore"):
        return max(float(10 * np.log10(np.mean(ratios))), MIN_SCORE_DB)


def mae(x_ref: np.ndarray, x_est: np.ndarray) -> float:
    """Mean absolute error in mV. Non-finite values in ``x_est`` count as 0."""
    x_est = np.where(np.isfinite(x_est), x_est, 0.0)
    return float(np.mean(np.abs(x_ref - x_est)))
