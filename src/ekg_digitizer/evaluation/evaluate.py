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

from ekg_digitizer.evaluation.metrics import align, mae, snr

Record = dict[str, np.ndarray]


def _nanmean(values: list[float]) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def evaluate(references: dict[str, Record], predictions: dict[str, Record], fs: float) -> dict[str, float]:
    """Return mean SNR (dB) and MAE (mV) over all leads of all records.

    Each lead's score is averaged per record, then records are averaged.
    Missing records or leads in ``predictions`` are scored as empty.
    """
    record_snrs, record_maes = [], []
    for record_id, reference in references.items():
        prediction = predictions.get(record_id, {})
        lead_snrs, lead_maes = [], []
        for lead, x_ref in reference.items():
            x_est = align(x_ref, prediction.get(lead, np.full_like(x_ref, np.nan)), fs)
            lead_snrs.append(snr(x_ref, x_est))
            lead_maes.append(mae(x_ref, x_est))
        record_snrs.append(_nanmean(lead_snrs))
        record_maes.append(_nanmean(lead_maes))

    return {"snr": _nanmean(record_snrs), "mae": _nanmean(record_maes), "num_records": len(references)}


def load_record(path: Path) -> Record:
    """Load a record CSV with a header row of lead names."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, deletechars="")
    return {name: np.atleast_1d(data[name]) for name in data.dtype.names}


def load_split(directory: Path) -> dict[str, Record]:
    """Load every ``*.csv`` record in ``directory``, keyed by file stem."""
    return {path.stem: load_record(path) for path in sorted(Path(directory).glob("*.csv"))}
