import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from ekg_digitizer.preprocessing.model_input import prepare_model_input


def test_matches_training_eval_transform():
    """prepare_model_input must produce the same tensor as the training
    pipeline's resize + normalize steps, so inference sees what the model saw."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
    width, height = 128, 96

    # Same steps as scripts/train.py, without augmentation
    reference = A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])(image=image)["image"]

    result = prepare_model_input(image, width, height)

    assert result.shape == (3, height, width)
    assert result.dtype == torch.float32
    assert torch.allclose(result, reference, atol=1e-5)
