# ==============================================================================
# File Name : model_input.py
# Purpose   : Single image preprocessing path shared by training and inference
# ==============================================================================

import cv2
import numpy as np
import torch

# ImageNet statistics, the same values used by the training augmentation
# pipeline in scripts/train.py. The pretrained backbone expects them, so
# inference must apply them too.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def prepare_model_input(image: np.ndarray, width: int, height: int) -> torch.Tensor:
    """Resize an RGB uint8 image and return a normalized CHW float tensor.

    Mirrors the training transform (A.Resize -> A.Normalize -> ToTensorV2):
    the image is scaled to [0, 1], normalized with ImageNet mean/std, and
    moved from HWC to CHW layout for PyTorch.
    """
    # Resize first so the normalization runs on the final pixel grid.
    # INTER_LINEAR is albumentations' default for A.Resize.
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    scaled = resized.astype(np.float32) / 255.0
    normalized = (scaled - IMAGENET_MEAN) / IMAGENET_STD

    # .copy() gives torch a contiguous array; transpose alone returns a view
    return torch.from_numpy(normalized.transpose(2, 0, 1).copy())
