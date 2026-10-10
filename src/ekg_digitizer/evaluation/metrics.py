# ==============================================================================
# File Name : metrics.py
# Purpose   : Signal reconstruction metrics (SNR, MAE) and alignment
# ==============================================================================
"""Metrics for comparing digitized EKG signals against reference signals.

All metrics operate on 1-D arrays for a single lead. Non-finite values
(NaN, +/-inf) mark missing samples, following the PhysioNet Challenge 2024
evaluation code. See ``docs/metrics.md`` for definitions.
"""

from __future__ import annotations

import numpy as np

# Defaults from the PhysioNet Challenge 2024 evaluation code.
MAX_SHIFT_S = 0.5
MAX_OFFSET_MV = 1.0


def _pad_to_same_length(x_ref: np.ndarray, x_est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad the shorter signal with NaNs so both signals have the same length."""
    x_ref = np.asarray(x_ref, dtype=float).ravel()
    x_est = np.asarray(x_est, dtype=float).ravel()
    n = max(x_ref.size, x_est.size)
    x_ref = np.pad(x_ref, (0, n - x_ref.size), constant_values=np.nan)
    x_est = np.pad(x_est, (0, n - x_est.size), constant_values=np.nan)
    return x_ref, x_est


def _shift(x: np.ndarray, lag: int) -> np.ndarray:
    """Shift ``x`` right by ``lag`` samples (left if negative), filling with NaN."""
    out = np.full_like(x, np.nan)
    if lag >= 0:
        out[lag:] = x[: x.size - lag]
    else:
        out[:lag] = x[-lag:]
    return out


def align(x_ref: np.ndarray, x_est: np.ndarray, fs: float) -> np.ndarray:
    """Align ``x_est`` to ``x_ref`` with a time shift and a vertical offset.

    The time shift is the lag within ``+/- MAX_SHIFT_S`` that maximizes the
    Pearson correlation over the overlapping samples. The vertical offset is
    the mean residual after shifting. If the offset exceeds ``MAX_OFFSET_MV``,
    the estimate is returned unaligned.
    """
    x_ref, x_est = _pad_to_same_length(x_ref, x_est)

    max_lag = int(round(MAX_SHIFT_S * fs))
    best_lag, best_corr = None, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        shifted = _shift(x_est, lag)
        idx = np.isfinite(x_ref) & np.isfinite(shifted)
        if idx.sum() < 2:
            continue
        a = x_ref[idx] - x_ref[idx].mean()
        b = shifted[idx] - shifted[idx].mean()
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            continue
        corr = float(np.dot(a, b)) / norm
        if corr > best_corr:
            best_lag, best_corr = lag, corr

    if best_lag is None:
        return x_est

    shifted = _shift(x_est, best_lag)
    idx = np.isfinite(x_ref) & np.isfinite(shifted)
    offset = float(np.mean(x_ref[idx] - shifted[idx]))
    if abs(offset) > MAX_OFFSET_MV:
        return x_est
    return shifted + offset


def snr(x_ref: np.ndarray, x_est: np.ndarray) -> float:
    """Signal-to-noise ratio in dB for one lead.

    ``10 * log10(sum(x_ref**2) / sum((x_ref - x_est)**2))`` over samples finite
    in both signals, scaled by the fraction of reference samples the estimate
    covers. Returns 0 when the estimate covers no samples, and NaN for a
    perfect reconstruction or an all-zero reference.
    """
    x_ref, x_est = _pad_to_same_length(x_ref, x_est)
    idx_ref = np.isfinite(x_ref)
    idx = idx_ref & np.isfinite(x_est)
    if idx_ref.any() and not idx.any():
        return 0.0

    p_signal = np.sum(x_ref[idx] ** 2)
    p_noise = np.sum((x_ref[idx] - x_est[idx]) ** 2)
    if not (p_signal > 0 and p_noise > 0):
        return float("nan")
    return float(10 * np.log10(p_signal / p_noise) * idx.sum() / idx_ref.sum())


def mae(x_ref: np.ndarray, x_est: np.ndarray) -> float:
    """Mean absolute error in mV over samples finite in both signals."""
    x_ref, x_est = _pad_to_same_length(x_ref, x_est)
    idx = np.isfinite(x_ref) & np.isfinite(x_est)
    return float(np.mean(np.abs(x_ref[idx] - x_est[idx]))) if idx.any() else float("nan")
