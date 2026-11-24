# ==============================================================================
# File Name : settings.py
# Purpose   : Service settings read from environment variables
# ==============================================================================
"""Runtime settings for the API.

Environment variables (all optional):
    EKG_CONFIG          : model config YAML (default: configs/config.yaml in the repo)
    EKG_CHECKPOINT      : local checkpoint or weights file; skips the W&B registry (dev/offline)
    EKG_MAX_UPLOAD_MB   : largest accepted upload in MB (default: 10)
    EKG_MAX_IMAGE_PIXELS: largest accepted decoded image, in pixels (default: 40,000,000)
    EKG_MAX_CONCURRENCY : simultaneous inferences per process (default: 1)

Registry loading (the default) additionally needs WANDB_API_KEY; see
``ekg_digitizer.serving``.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from ekg_digitizer.inference import DEFAULT_CONFIG


@dataclass(frozen=True)
class Settings:
    config_path    : Path
    checkpoint     : Path | None
    max_upload_bytes: int
    max_image_pixels: int
    max_concurrency : int

    @classmethod
    def from_env(cls) -> "Settings":
        checkpoint = os.environ.get("EKG_CHECKPOINT")
        return cls(
            config_path      = Path(os.environ.get("EKG_CONFIG", DEFAULT_CONFIG)),
            checkpoint       = Path(checkpoint) if checkpoint else None,
            max_upload_bytes = int(float(os.environ.get("EKG_MAX_UPLOAD_MB", "10")) * 1024 * 1024),
            max_image_pixels = int(os.environ.get("EKG_MAX_IMAGE_PIXELS", "40000000")),
            max_concurrency  = max(1, int(os.environ.get("EKG_MAX_CONCURRENCY", "1"))),
        )

# [END]