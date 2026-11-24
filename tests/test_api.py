from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from ekg_digitizer.api.main import create_app
from ekg_digitizer.api.service import Digitizer
from ekg_digitizer.api.settings import Settings
from ekg_digitizer.config import load_config
from ekg_digitizer.evaluation.metrics import LEADS
from ekg_digitizer.models.model import EnhancedEKGNet

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def digitizer():
    config = load_config(str(REPO / "configs" / "config.yaml"))
    config.image_properties.resize_width = 128  # small input keeps CPU tests fast
    config.image_properties.resize_height = 96
    config.augmentations.use_tta = False
    torch.manual_seed(0)
    model = EnhancedEKGNet(config).eval()
    return Digitizer(model, config, torch.device("cpu"), version="v-test")


def make_client(digitizer, **overrides):
    base = Settings.from_env()
    settings = Settings(**{**base.__dict__, **overrides})
    return TestClient(create_app(settings, digitizer))


@pytest.fixture
def client(digitizer):
    with make_client(digitizer) as c:
        yield c


def png_bytes(shape=(120, 160, 3)):
    img = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def upload(client, data, name="ekg.png", mime="image/png", **params):
    return client.post("/api/v1/digitize", files={"file": (name, data, mime)}, params=params)


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/ready").json()
    assert body["status"] == "ready" and body["model_version"] == "v-test"


def test_ready_is_503_before_model_loads(digitizer):
    app = create_app(Settings.from_env(), digitizer)  # lifespan not started
    assert TestClient(app).get("/ready").status_code == 503


def test_digitize_json(client):
    res = upload(client, png_bytes())
    assert res.status_code == 200
    body = res.json()
    assert body["model_version"] == "v-test" and body["fs"] == 500 and body["units"] == "mV"
    assert list(body["leads"]) == LEADS
    assert body["leads"]["II"]["n_samples"] == 5000 and len(body["leads"]["II"]["values"]) == 5000
    assert body["leads"]["I"]["n_samples"] == 1250 and body["leads"]["I"]["duration_s"] == 2.5


def test_digitize_accepts_jpeg(client):
    _, buf = cv2.imencode(".jpg", np.full((100, 100, 3), 200, dtype=np.uint8))
    assert upload(client, buf.tobytes(), "a.jpg", "image/jpeg").status_code == 200


def test_resampling(client):
    body = upload(client, png_bytes(), fs=250).json()
    assert body["fs"] == 250 and body["leads"]["II"]["n_samples"] == 2500


def test_digitize_csv(client):
    res = upload(client, png_bytes(), format="csv")
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/csv")
    lines = res.text.strip().splitlines()
    assert lines[0].split(",") == LEADS and len(lines) == 5001
    assert res.headers["x-model-version"] == "v-test"


def test_rejects_non_image(client):
    res = upload(client, b"not an image", "x.png")  # a spoofed name/content-type must not matter
    assert res.status_code == 415


def test_rejects_corrupt_image(client):
    res = upload(client, png_bytes()[:60])  # valid PNG signature, truncated body
    assert res.status_code == 400


def test_rejects_empty_file(client):
    assert upload(client, b"").status_code == 400


def test_rejects_oversized_upload(digitizer):
    with make_client(digitizer, max_upload_bytes=1024) as c:
        assert upload(c, png_bytes((200, 200, 3))).status_code == 413


def test_rejects_too_many_pixels(digitizer):
    with make_client(digitizer, max_image_pixels=100) as c:
        assert upload(c, png_bytes()).status_code == 400


def test_rejects_bad_fs_and_missing_file(client):
    assert upload(client, png_bytes(), fs=5).status_code == 422
    assert client.post("/api/v1/digitize").status_code == 422


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200 and "EKG Digitizer" in res.text