import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from ekg_digitizer.config import load_config
from ekg_digitizer.evaluation.evaluate import evaluate, load_record
from ekg_digitizer.evaluation.metrics import LEADS
from ekg_digitizer.inference import (
    index_images,
    load_model,
    predict_record,
    resample_record,
    tta_views,
    write_record,
)
from ekg_digitizer.models.model import EnhancedEKGNet

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def config():
    config = load_config(str(REPO / "configs" / "config.yaml"))
    # Small input keeps the forward pass fast on CPU; the network is size-agnostic
    config.image_properties.resize_width = 128
    config.image_properties.resize_height = 96
    return config


@pytest.fixture(scope="module")
def checkpoint(config, tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "best_model.pth"
    torch.save({"epoch": 0, "model_state_dict": EnhancedEKGNet(config).state_dict()}, path)
    return path


@pytest.fixture
def image():
    return np.random.default_rng(0).integers(0, 256, size=(300, 400, 3), dtype=np.uint8)


def test_predict_record_shapes(config, checkpoint, image):
    record = predict_record(checkpoint, image, config=config, device="cpu", use_tta=False)

    assert list(record) == LEADS
    assert record["II"].shape == (5000,)  # 10 s at 500 Hz
    assert all(record[lead].shape == (1250,) for lead in LEADS if lead != "II")  # 2.5 s
    assert all(np.isfinite(signal).all() for signal in record.values())


def test_predict_record_is_deterministic(config, checkpoint, image):
    a = predict_record(checkpoint, image, config=config, device="cpu", use_tta=False)
    b = predict_record(checkpoint, image, config=config, device="cpu", use_tta=False)
    assert all(np.array_equal(a[lead], b[lead]) for lead in LEADS)


def test_weights_only_file_matches_checkpoint(config, checkpoint, tmp_path, image):
    weights = tmp_path / "model_weights.pth"
    torch.save(torch.load(checkpoint, weights_only=True)["model_state_dict"], weights)

    full = load_model(checkpoint, config)
    slim = load_model(weights, config)
    x = torch.randn(1, 3, 96, 128)
    with torch.inference_mode():
        assert torch.equal(full(x)["II"], slim(x)["II"])


def test_tta_keeps_shapes_and_identity_first(config, checkpoint, image):
    views = tta_views(image, 5)
    assert len(views) == 5 and np.array_equal(views[0], image)
    assert len(tta_views(image, 1)) == 1 and len(tta_views(image, 99)) == 7

    tta_config = dataclasses.replace(config, augmentations=dataclasses.replace(config.augmentations, tta_count=3))
    plain = predict_record(checkpoint, image, config=tta_config, device="cpu", use_tta=False)
    tta = predict_record(checkpoint, image, config=tta_config, device="cpu", use_tta=True)
    assert tta["II"].shape == plain["II"].shape
    assert not np.array_equal(tta["II"], plain["II"])


def test_rejects_non_rgb_uint8(config, checkpoint):
    with pytest.raises(ValueError):
        predict_record(checkpoint, np.zeros((64, 64), dtype=np.uint8), config=config, device="cpu")


@pytest.mark.parametrize("fs", [250, 500, 1000])
def test_resample_record_lengths(fs):
    record = {"II": np.zeros(5000, dtype=np.float32), "I": np.zeros(1250, dtype=np.float32)}
    out = resample_record(record, fs, 500)
    assert out["II"].size == 10 * fs and out["I"].size == int(2.5 * fs)


def test_written_record_scores_through_evaluate(tmp_path):
    rng = np.random.default_rng(0)
    record = {lead: rng.normal(size=5000 if lead == "II" else 1250).astype(np.float32) for lead in LEADS}
    write_record(record, tmp_path / "r1.csv")

    loaded = load_record(tmp_path / "r1.csv")
    assert list(loaded) == LEADS
    assert loaded["II"].size == 5000 and loaded["V6"].size == 1250  # empty cells dropped

    scores = evaluate({"r1": loaded}, {"r1": load_record(tmp_path / "r1.csv")})
    assert scores["num_records"] == 1 and scores["mae"] < 1e-9


def test_index_images_layouts(tmp_path):
    (tmp_path / "A").mkdir()
    for name in ["A/A-0003.png", "A/A-0001.png", "A/A-0012.png", "B.png", "B-0001.png"]:
        (tmp_path / name).write_bytes(b"")

    found = index_images(tmp_path)
    assert found["A"].name == "A-0001.png"  # first in SEGMENTS order
    assert found["B"].name == "B.png"  # a plain <id>.png wins