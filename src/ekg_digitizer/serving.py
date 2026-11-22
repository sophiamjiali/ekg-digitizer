# ==============================================================================
# File Name : serving.py
# Purpose   : Loads the production model from the W&B Model Registry at startup
# ==============================================================================
"""Load whichever registry version holds the ``production`` alias.

Call ``load_production_model`` once when the service starts. Promoting another
version (``scripts/register_model.py promote ... --alias production``) changes
the served model on the next restart, with no code change.

Environment variables:
    WANDB_API_KEY       : W&B API key (required)
    EKG_REGISTRY_PATH   : registry to load from (default: wandb-registry-model/ekg-digitizer)
    EKG_MODEL_CACHE_DIR : where downloaded weights are kept (default: ~/.cache/ekg-digitizer)
"""

import os
from dataclasses import dataclass
from pathlib import Path

import wandb
from torch import nn

from ekg_digitizer.models.weights import WEIGHTS_FILE, load_weights
from ekg_digitizer.registry import REGISTRY_PATH

SERVED_ALIAS = "production"


@dataclass(frozen=True)
class ServedModel:
    """The loaded model plus which registry version it came from."""

    model: nn.Module
    version: str  # e.g. "v3"; expose this so callers can see what's being served
    weights_path: Path


def _api() -> wandb.Api:
    """Connect to W&B with the key from WANDB_API_KEY, failing clearly if it's missing."""
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY is not set")
    return wandb.Api(api_key=api_key)


def fetch_production_weights(api: wandb.Api | None = None) -> tuple[str, Path]:
    """Download the production version's weights file, reusing a cached copy if present.

    Returns ``(version, path)``. The alias is looked up on every call, so a new
    promotion is picked up; only the download is skipped when cached.
    """
    api = api or _api()
    registry_path = os.environ.get("EKG_REGISTRY_PATH", REGISTRY_PATH)
    cache_root = Path(os.environ.get("EKG_MODEL_CACHE_DIR", Path.home() / ".cache" / "ekg-digitizer"))

    artifact = api.artifact(f"{registry_path}:{SERVED_ALIAS}")
    version = artifact.version

    # Cache per version: versions are immutable, so a cached file never goes stale
    version_dir = cache_root / version
    weights_path = version_dir / WEIGHTS_FILE
    if not weights_path.exists():
        try:
            entry = artifact.get_entry(WEIGHTS_FILE)
        except KeyError as err:
            # Models logged before the weights-only export (#34) only have the full checkpoint
            raise RuntimeError(f"{registry_path}:{version} has no {WEIGHTS_FILE}; re-log it with log_model") from err
        entry.download(root=str(version_dir))  # only the slim file, not the full checkpoint

    return version, weights_path


def load_production_model(model: nn.Module, api: wandb.Api | None = None) -> ServedModel:
    """Load the production weights into ``model`` (built from the config) for serving."""
    version, weights_path = fetch_production_weights(api)
    return ServedModel(model=load_weights(model, weights_path), version=version, weights_path=weights_path)
