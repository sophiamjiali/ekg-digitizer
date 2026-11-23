# ==============================================================================
# File Name : inference.py
# Purpose   : Helper functions for entry point inference script
# ==============================================================================

from __future__ import annotations

import re
import cv2
import torch

import pandas as pd
import numpy as numpy

from torch import nn
from math import gcd
from pathlib import Path
from scipy.signal import resample_poly

from ekg_digitizer.config import Config, load_config
from ekg_digitizer.evaluation.evaluate import Record
from ekg_digitizer.evaluation.metrics import LEADS
from ekg_digitizer.models.dataset import SEGMENTS
from ekg_digitizer.models.model import EnhancedEKGNet
from ekg_digitizer.preprocessing.model_input import prepare_model_input

# Constants
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_IMAGE_STEM    = re.compile(r"^(?P<id>.+?)(?:-(?P<segment>\d{4}))?$")
_MODEL_CACHE: dict[tuple[str, int, str], nn.Module] = {}


# =====| Model Loading |========================================================

def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint: str | Path,
               config    : Config,
               device    : str | torch.device = "cpu") -> nn.Module:
    """
    Build the network and load `model_state_dict` from a training checkpoint. 
    
    A weights-only file (`model_weights.pth`, a bare state dictionary) is 
    loaded too. The model is returned in evaluation mode, so dropout is toggled 
    off and batch normalization uses its running statistics.
    """

    # Load the training model checkpoint as weights only (no model architecture)
    state = torch.load(checkpoint, map_location = "cpu", weights_only = True)
    state = state.get('model_state_dict', state)
    state = {key.removeprefix('module.'): value for key, value in state.items()}

    # Initialize the model with the loaded weights
    model = EnhancedEKGNet(config)
    model.load_state_dict(state)

    return model.to(device).eval()


# =====| Prediction |===========================================================

def tta_views(image: np.ndarray, count: int) -> list[np.ndarray]:
    """
    Return `image` plus up to `count - 1` mild, deterministic photometric 
    variants.
    
    The variants stay inside the contract, brightness, and blur ranges of the 
    training augmentations. No geometric changes: a shift or scale would move 
    the waveform in time, and averaging misaligned predictions would blur it. 
    The original image is always first, so `count = 1` means no TTA.
    """

    variants = [
        lambda x: x,
        lambda x: cv2.convertScaleAbs(x, alpha=1.1, beta=0),
        lambda x: cv2.convertScaleAbs(x, alpha=0.9, beta=0),
        lambda x: cv2.GaussianBlur(x, (3, 3), 0),
        lambda x: cv2.convertScaleAbs(x, alpha=1.0, beta=12),
        lambda x: cv2.convertScaleAbs(x, alpha=1.0, beta=-12),
        lambda x: cv2.medianBlur(x, 3),
    ]
    return [make(image) for make in variants[: max(count, 1)]]


@torch.inference_mode()
def predict_image(model  : nn.Module,
                  image  : np.ndarray,
                  config : Config,
                  device : str | torch.device = "cpu",
                  use_tta: bool | None = None) -> Record: 
    """
    Predict every lead from one RGB uint8 image, averaging over TTA views.

    `use_tta = None` follows `config.augmentations.use_tta`. Signals are in mV 
    at `config.signal_properties.base_fs`.
    """

    # Type-check for an RGB uint8 image
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB uint8 image (H,W,3), got "
                         f"{image.dtype} {image.shape}")

    # Consolidate using TTA against provided parameters and GT configurations
    use_tta = config.augmentations.use_tta if use_tta is None else use_tta
    views   = tta_views(image, config.augmentations.tta_count if use_tta else 1)

    # Prepare the inference batch using the configuration width and height
    width  = config.image_properties.resize_width
    height = config.image_properties.resize_height
    batch  = torch.stack([prepare_model_input(view, width, height) 
                         for view in views]).to(device)

    outputs = model(batch)
    return {lead: signal.mean(dim = 0).float().cpu().numpy() 
            for lead, signal in outputs.items()}


def _cached_model(checkpoint: str  | Path,
                  config    : Config,
                  device    : torch.device) -> nn.Module:
    """
    Load once per (file, modification time, device); keeps a single model in 
    memory.
    """

    path = Path(checkpoint)
    key  = (str(path.resolve()), path.stat().st_mtime_ns, str(device))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = load_model(path, config, device)

    return _MODEL_CACHE[key]


def predict_record(checkpoint: str | Path,
                   image     : np.ndarray,
                   config    : Config | None = None,
                   device    : str | torch.device | None = None,
                   use_tta   : bool | None = None) -> Record:
    """
    Digitize one image: `(checkpoint path, RGB uint8 image) -> {lead: signal in 
    mV}`. The model is cached between calls, so looping over records does not 
    reload it. `config` defaults to `configs/config.yaml` and must match the 
    checkpoint.
    """

    config = config or load_config(str(DEFAULT_CONFIG))
    device = torch.device(device) if device else default_device()

    return predict_image(
        model   = _cached_model(checkpoint, config, device),
        image   = image,
        config  = config,
        device  = device,
        use_tta = use_tta
    )

# =====| Images and Records on Disk |===========================================

def read_image(path: str | Path) -> np.ndarray:
    """Read an image file as RGB uint8, like `EKGDataset._convert_image`."""

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None  : raise ValueError(f"Unable to load image: {path}")
    if img.ndim == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB if img.shape[2] == 4 
                        else CV2.COLOR_BGR2RGB)


def index_images(image_dir: str | Path) -> dict[str, Path]:
    """
    Map record id to the image to digitize, scanning `image_dir` recursively.

    Handles both `<id>.png` and the training layout `<id>/<id>-<segment>.png`. 
    A plain `<id>.png` wins; otherwise, the first segment in `SEGMENTS` order, 
    the same choice `EKGDataset.load_image` makes.
    """

    best: dict[str, tuple[int, Path]] = {}
    for path in sorted(Path(image_dir).rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES: continue

        match              = _IMAGE_STEM.match(path.stem)
        record_id, segment = match['id'], match['segment']

        if segment is None: rank = -1
        else: rank = SEGMENTS.index(segment) if segment in SEGMENTS 
                     else len(SEGMENTS)

        if record_id not in best or rank < best[record_id][0]:
            best[record_id] = (rank, path)

        return {record_id: path for record_id, (_, path) in best.items()}


def resample_record(record: Record, fs: int, base_fs: int) -> Record:
    """Resample a record from `base_fs` to `fs`; inverse of `EKGDataset`."""

    if fs == base_fs: return record

    divisor  = gcd(fs, base_fs)
    up, down = fs // divisor, base_fs // divisor

    return {lead: resample_poly(signal, up = up, down = down)
                  .astype(np.float32) for lead, signal in record.items()}


def write_record(record: Record, path: str | Path) -> None:
    """
    Write a record CSV: header of lead names, one column per lead. Shorter 
    leads leave their trailing cells empty, which `load_record` drops.
    """

    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    columns = {lead: pd.Series(record[lead]) for lead in LEADS 
                     if lead in record}

    pd.DataFrame(columns).to_csv(path, index = False, float_format = "%.6f")

  # [END]