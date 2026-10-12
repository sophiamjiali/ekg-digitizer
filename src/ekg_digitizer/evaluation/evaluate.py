# ==============================================================================
# File Name : evaluate.py
# Purpose   : Scores predicted signals against references across a data split
# ==============================================================================
"""Evaluate predicted EKG signals against reference signals.

A record is a mapping from lead name (e.g. ``"II"``) to a 1-D signal in mV.
On disk, each record is a CSV file with a header row of lead names and one
column per lead; references and predictions share file names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ekg_digitizer.evaluation.metrics import LEADS, align, mae, power, snr_ratio, to_db

Record = dict[str, np.ndarray]


def _fit_length(x_est: np.ndarray, n: int) -> np.ndarray:
    """Truncate or zero-pad a prediction to ``n`` samples."""
    return np.pad(x_est[:n], (0, max(n - x_est.size, 0)))


def score_record(reference: Record, prediction: Record) -> tuple[float, float]:
    """Return ``(linear SNR ratio, MAE)`` for one record.

    Signal and noise power are summed over all leads before taking the ratio.
    The sampling frequency is inferred from lead II, which spans 10 s.
    Leads missing from ``prediction`` are scored as all zeros.
    """
    fs = reference["II"].size / 10
    signal = noise = 0.0
    maes = []
    for lead in (lead for lead in LEADS if lead in reference):
        x_ref = reference[lead]
        x_est = _fit_length(prediction.get(lead, np.zeros(0)), x_ref.size)
        aligned = align(x_ref, x_est, fs)
        p_signal, p_noise = power(x_ref, aligned)
        signal += p_signal
        noise += p_noise
        maes.append(mae(x_ref, aligned))
    return snr_ratio(signal, noise), float(np.mean(maes))


def evaluate(references: dict[str, Record], predictions: dict[str, Record]) -> dict[str, float]:
    """Return SNR (dB) and MAE (mV) over a split.

    SNR follows the competition: per-record linear ratios are averaged, then
    converted to dB. MAE is averaged over leads, then records. Missing
    records in ``predictions`` are scored as all zeros.
    """
    scores = [score_record(ref, predictions.get(record_id, {})) for record_id, ref in references.items()]
    return {
        "snr": to_db([ratio for ratio, _ in scores]),
        "mae": float(np.mean([m for _, m in scores])),
        "num_records": len(scores),
    }


def load_record(path: Path) -> Record:
    """Load a record CSV with a header row of lead names. Empty cells are dropped."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, deletechars="")
    return {name: (x := np.atleast_1d(data[name]))[np.isfinite(x)] for name in data.dtype.names} # type: ignore


def load_split(directory: Path) -> dict[str, Record]:
    """Load every ``*.csv`` record in ``directory``, keyed by file stem."""
    return {path.stem: load_record(path) for path in sorted(Path(directory).glob("*.csv"))}
