"""Regression tests for train/serve parity (issue #36).

Two checks:
  1. Preprocessing on the fixed regression images (runs whenever data/ exists).
  2. End-to-end SNR/MAE of the trained checkpoint on those images, compared with
     a recorded baseline (runs only once the inference pipeline from #33 exists,
     and when the checkpoint, data, and baseline file are all present).

Both skip cleanly in CI, where data/ and best_model.pth are not available.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from ekg_digitizer.config import load_config
from ekg_digitizer.evaluation.evaluate import evaluate, load_record
from ekg_digitizer.models.dataset import EKGDataset
from ekg_digitizer.preprocessing.model_input import prepare_model_input

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "train"
CHECKPOINT = REPO / "best_model.pth"
IDS_FILE = REPO / "tests" / "fixtures" / "regression_ids.txt"
BASELINE_FILE = REPO / "tests" / "fixtures" / "regression_baseline.json"

# Tolerances against the recorded baseline
SNR_TOLERANCE_DB = 0.1
MAE_TOLERANCE_MV = 0.001


def load_regression_ids() -> list[str]:
    """Read the fixed record IDs, skipping comment lines."""
    return [
        line.strip()
        for line in IDS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


requires_data = pytest.mark.skipif(
    not DATA_DIR.exists(), reason="data/train not present (not in git)"
)


@requires_data
def test_preprocessing_on_regression_images():
    """Each regression image loads and preprocesses to the model's input shape."""
    config = load_config(str(REPO / "configs" / "config.yaml"))
    width = config.image_properties.resize_width
    height = config.image_properties.resize_height

    dataset = EKGDataset(
        images=pd.DataFrame({"id": load_regression_ids()}),
        data_dir=str(DATA_DIR),
        config=config,
        transform=None,
        is_train=True,
    )

    for idx in range(len(dataset)):
        base_id = dataset.images.iloc[idx]["id"]
        tensor = prepare_model_input(dataset.load_image(base_id), width, height)
        assert tuple(tensor.shape) == (3, height, width), base_id
        assert bool(tensor.isfinite().all()), base_id


def _inference_available() -> bool:
    """True once #33 has added the predict function this test depends on."""
    try:
        from ekg_digitizer.inference import predict_record  # noqa: F401
    except ImportError:
        return False
    return True


@requires_data
@pytest.mark.skipif(not CHECKPOINT.exists(), reason="best_model.pth not present (not in git)")
@pytest.mark.skipif(not BASELINE_FILE.exists(), reason="baseline not recorded yet")
@pytest.mark.skipif(not _inference_available(), reason="inference pipeline (#33) not implemented yet")
def test_snr_matches_recorded_baseline():
    """Predictions from the checkpoint score within tolerance of the baseline.

    Contract with #33 (adjust here if #33 differs):
        predict_record(checkpoint_path, image) -> {lead: 1-D np.ndarray}
    where ``image`` is the RGB uint8 array from EKGDataset.load_image().
    """
    from ekg_digitizer.inference import predict_record

    config = load_config(str(REPO / "configs" / "config.yaml"))
    dataset = EKGDataset(
        images=pd.DataFrame({"id": load_regression_ids()}),
        data_dir=str(DATA_DIR),
        config=config,
        transform=None,
        is_train=True,
    )

    references = {}
    predictions = {}
    for base_id in dataset.images["id"]:
        references[base_id] = load_record(DATA_DIR / str(base_id) / f"{base_id}.csv")
        predictions[base_id] = predict_record(str(CHECKPOINT), dataset.load_image(base_id))

    scores = evaluate(references, predictions)
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))

    assert abs(scores["snr"] - baseline["snr"]) <= SNR_TOLERANCE_DB, scores
    assert abs(scores["mae"] - baseline["mae"]) <= MAE_TOLERANCE_MV, scores
