import numpy as np
import pytest

from ekg_digitizer.evaluation import align, evaluate, load_split, mae, snr

FS = 500


@pytest.fixture
def ecg_like():
    t = np.arange(0, 2.5, 1 / FS)
    return np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 7 * t)


def test_snr_matches_definition(ecg_like):
    noise = 0.1 * np.random.default_rng(0).standard_normal(ecg_like.size)
    expected = 10 * np.log10(np.sum(ecg_like**2) / np.sum(noise**2))
    assert snr(ecg_like, ecg_like + noise) == pytest.approx(expected)


def test_snr_is_zero_for_empty_prediction(ecg_like):
    assert snr(ecg_like, np.full_like(ecg_like, np.nan)) == 0.0


def test_snr_penalizes_missing_samples(ecg_like):
    est = ecg_like + 0.1
    partial = est.copy()
    partial[: ecg_like.size // 2] = np.nan
    assert snr(ecg_like, partial) < snr(ecg_like, est)


def test_mae(ecg_like):
    assert mae(ecg_like, ecg_like + 0.2) == pytest.approx(0.2)


def test_align_removes_shift_and_offset(ecg_like):
    est = np.roll(ecg_like, 25) + 0.4
    assert snr(ecg_like, align(ecg_like, est, FS)) > snr(ecg_like, est) + 20


def test_align_ignores_large_offset(ecg_like):
    est = ecg_like + 5.0
    np.testing.assert_array_equal(align(ecg_like, est, FS), est)


def test_evaluate_on_split(tmp_path, ecg_like):
    for name in ("ref", "pred"):
        (tmp_path / name).mkdir()
    for record in ("r1", "r2"):
        np.savetxt(tmp_path / "ref" / f"{record}.csv", np.column_stack([ecg_like, -ecg_like]), delimiter=",", header="I,II", comments="")
    noise = 0.1 * np.random.default_rng(0).standard_normal((ecg_like.size, 2))
    np.savetxt(tmp_path / "pred" / "r1.csv", np.column_stack([ecg_like, -ecg_like]) + noise, delimiter=",", header="I,II", comments="")

    references, predictions = load_split(tmp_path / "ref"), load_split(tmp_path / "pred")
    r1_only = evaluate({"r1": references["r1"]}, predictions, fs=FS)
    scores = evaluate(references, predictions, fs=FS)
    assert scores["num_records"] == 2
    assert 10 < r1_only["snr"] < 30
    assert scores["snr"] == pytest.approx(r1_only["snr"] / 2)  # missing r2 scores 0 dB
