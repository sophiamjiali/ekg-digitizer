import numpy as np
import pytest

from ekg_digitizer.evaluation import align, evaluate, load_split, mae, score_record

# Example record and expected scores from the official competition metric's doctests.
OFFICIAL_IDS = (
    [f"{i}_{lead}" for lead in ["I", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"] for i in range(3)]
    + [f"{i}_II" for i in range(12)]
)
OFFICIAL_VALUES = [
    0.1, 0.3, 0.4, 0.6, 0.6, 0.4, 0.2, 0.3, 0.4, 0.5, 0.2, 0.7, 0.2, 0.3, 0.4, 0.8, 0.6, 0.7,
    0.2, 0.3, -0.1, 0.5, 0.6, 0.7, 0.2, 0.9, 0.4, 0.5, 0.6, 0.7, 0.1, 0.3, 0.4,
    0.6, 0.6, 0.4, 0.2, 0.3, 0.4, 0.5, 0.2, 0.7, 0.2, 0.3, 0.4,
]  # fmt: skip


def to_record(values):
    record = {}
    for row_id, value in zip(OFFICIAL_IDS, values):
        record.setdefault(row_id.split("_")[1], []).append(value)
    return {lead: np.array(v) for lead, v in record.items()}


@pytest.mark.parametrize(
    "edits, expected",
    [({}, 60.0), ({0: 0.9}, 13.6291), ({0: 0.9, 4: 0.3}, 13.0576)],
)
def test_snr_matches_official_examples(edits, expected):
    reference = {"r": to_record(OFFICIAL_VALUES)}
    values = list(OFFICIAL_VALUES)
    for i, v in edits.items():
        values[i] = v
    assert evaluate(reference, {"r": to_record(values)})["snr"] == pytest.approx(expected, abs=1e-4)


def test_zero_reference_scores_minimum():
    zeros = to_record([0.0] * len(OFFICIAL_VALUES))
    assert evaluate({"r": zeros}, {"r": to_record(OFFICIAL_VALUES)})["snr"] == -384


@pytest.mark.parametrize(
    "ref, est, fs, expected",
    [
        ([0, 1, 2, 1, 0], [10, 11, 12, 11, 10], 100, [0, 1, 2, 1, 0]),  # vertical offset removed
        ([0, 0, 1, 2, 1, 0, 0], [1, 2, 1, 0, 0, 0, 0], 100, [np.nan, np.nan, 1, 2, 1, 0, 0]),  # time shift corrected
        ([0, 0, 0, 0, 1, 2, 1], [1, 2, 1, 0, 0, 0, 0], 50, [np.nan] * 4 + [1, 2, 1]),  # shift limited to 0.2 * fs
    ],
)
def test_align_matches_official_examples(ref, est, fs, expected):
    np.testing.assert_allclose(align(np.array(ref, float), np.array(est, float), fs), expected, atol=1e-9)


def test_missing_leads_and_records_score_as_zeros():
    reference = to_record(OFFICIAL_VALUES)
    zeros = {lead: np.zeros_like(x) for lead, x in reference.items()}
    assert score_record(reference, {}) == score_record(reference, zeros)
    assert evaluate({"r": reference}, {})["num_records"] == 1


def test_mae():
    assert mae(np.array([1.0, 2.0]), np.array([1.5, np.nan])) == pytest.approx(1.25)


def test_evaluate_on_split(tmp_path):
    t = np.arange(0, 10, 1 / 100)
    ecg = np.sin(2 * np.pi * 1.2 * t)
    for name in ("ref", "pred"):
        (tmp_path / name).mkdir()
    noise = 0.1 * np.random.default_rng(0).standard_normal(t.size)
    np.savetxt(tmp_path / "ref" / "r1.csv", np.column_stack([ecg, ecg]), delimiter=",", header="I,II", comments="")
    np.savetxt(tmp_path / "pred" / "r1.csv", np.column_stack([ecg + noise, np.roll(ecg, 5) + 0.3]), delimiter=",", header="I,II", comments="")

    scores = evaluate(load_split(tmp_path / "ref"), load_split(tmp_path / "pred"))
    assert scores["num_records"] == 1
    assert 15 < scores["snr"] < 30
