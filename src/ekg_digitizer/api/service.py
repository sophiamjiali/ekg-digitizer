# ==============================================================================
# File Name : service.py
# Purpose   : Image validation and inference wrapper, independent of FastAPI
# ==============================================================================
"""Turns uploaded bytes into a validated RGB image and runs the model on it."""

import threading
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn

from ekg_digitizer.config import Config
from ekg_digitizer.evaluation.evaluate import Record
from ekg_digitizer.inference import predict_image, resample_record

# Formats the endpoint accepts, detected from magic bytes (never from the filename
# or the client-supplied Content-Type, both of which are attacker-controlled).
_SIGNATURES = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
}


class InvalidImageError(ValueError):
    """The upload is not a usable image (maps to HTTP 400/415)."""


class UnsupportedImageTypeError(InvalidImageError):
    """The upload is not a PNG or JPEG (maps to HTTP 415)."""


class ModelOutputError(RuntimeError):
    """The model produced non-finite values (maps to HTTP 500)."""


def decode_image(data: bytes, max_pixels: int) -> np.ndarray:
    """Decode PNG/JPEG bytes into an RGB uint8 array, rejecting oversized images."""
    if not data:
        raise InvalidImageError("Empty file")
    if not any(data.startswith(sig) for sig in _SIGNATURES):
        raise UnsupportedImageTypeError("Only PNG and JPEG images are supported")

    # Cheap header-only check would be ideal; imdecode needs the full decode, so the
    # upload size cap (enforced by the caller) is the first line of defence and the
    # pixel cap below guards the downstream tensor allocations.
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR_RGB)
    if img is None:
        raise InvalidImageError("Could not decode image; the file may be corrupt")
    if img.shape[0] * img.shape[1] > max_pixels:
        raise InvalidImageError(f"Image exceeds the {max_pixels:,} pixel limit")
    return img


@dataclass
class Digitizer:
    """A loaded model plus everything needed to run it safely from many requests."""

    model   : nn.Module
    config  : Config
    device  : torch.device
    version : str
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        self._slots = threading.BoundedSemaphore(self.max_concurrency)

    @property
    def base_fs(self) -> int:
        return self.config.signal_properties.base_fs

    def digitize(self, image: np.ndarray, fs: int | None = None, use_tta: bool | None = None) -> Record:
        """Run inference; blocks (call from a worker thread) while all slots are busy."""
        with self._slots:
            record = predict_image(self.model, image, self.config, self.device, use_tta)
        if fs is not None:
            record = resample_record(record, fs, self.base_fs)
        if not all(np.isfinite(signal).all() for signal in record.values()):
            raise ModelOutputError("Model produced non-finite values")
        return record