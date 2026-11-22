# ==============================================================================
# File Name : weights.py
# Purpose   : Exports and loads weights-only (state_dict) model files for serving
# ==============================================================================
"""Weights-only model files.

The training checkpoint (``best_model.pth``) also holds the optimizer state,
which is only needed to resume training. Serving needs just the model's
``state_dict``, so a slim copy is exported for faster image builds and cold starts.
"""

from pathlib import Path

import torch
from torch import nn

WEIGHTS_FILE = "model_weights.pth"


def export_weights(checkpoint: str | Path, out: str | Path) -> Path:
    """Save only ``model_state_dict`` from a full training checkpoint to ``out``."""
    # weights_only=True refuses arbitrary pickled objects, so loading is safe
    full = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.save(full["model_state_dict"], out)
    return Path(out)


def load_weights(model: nn.Module, path: str | Path) -> nn.Module:
    """Load a weights-only file into ``model`` and switch it to eval mode for serving."""
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.eval()
