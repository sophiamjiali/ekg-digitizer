import cv2
import numpy as np
import pandas as pd

from src.ekg_digitizer.models.dataset import EKGDataset


class ImageProperties:
    img_height = 100
    img_width = 100
    img_channels = 3
    resize_width = 64
    resize_height = 64


class SignalProperties:
    base_fs = 500
    lead_ii_duration = 10.0
    other_leads_duration = 2.5


class TestConfig:
    image_properties = ImageProperties()
    signal_properties = SignalProperties()


def make_dataset():
    return EKGDataset(
        images=pd.DataFrame({"id": []}),
        data_dir=".",
        config=TestConfig(),
        is_train=False
    )


def test_convert_grayscale_to_rgb(tmp_path):
    dataset = make_dataset()

    image = np.full((10, 10), 128, dtype=np.uint8)
    image_path = tmp_path / "gray.png"
    cv2.imwrite(str(image_path), image)

    converted = dataset._convert_image(str(image_path))

    assert converted.shape == (10, 10, 3)
    assert np.all(converted[:, :, 0] == 128)
    assert np.all(converted[:, :, 1] == 128)
    assert np.all(converted[:, :, 2] == 128)


def test_convert_bgr_to_rgb(tmp_path):
    dataset = make_dataset()

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :] = [255, 0, 0]  # BGR blue

    image_path = tmp_path / "bgr.png"
    cv2.imwrite(str(image_path), image)

    converted = dataset._convert_image(str(image_path))

    assert converted.shape == (10, 10, 3)
    assert np.all(converted[:, :, 0] == 0)
    assert np.all(converted[:, :, 1] == 0)
    assert np.all(converted[:, :, 2] == 255)


def test_convert_rgba_to_rgb(tmp_path):
    dataset = make_dataset()

    image = np.zeros((10, 10, 4), dtype=np.uint8)
    image[:, :] = [255, 0, 0, 255]  # BGRA blue

    image_path = tmp_path / "rgba.png"
    cv2.imwrite(str(image_path), image)

    converted = dataset._convert_image(str(image_path))

    assert converted.shape == (10, 10, 3)
    assert np.all(converted[:, :, 0] == 0)
    assert np.all(converted[:, :, 1] == 0)
    assert np.all(converted[:, :, 2] == 255)
