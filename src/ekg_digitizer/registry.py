# ==============================================================================
# File Name : registry.py
# Purpose   : Logs, registers, and promotes models in the W&B Model Registry
# ==============================================================================
"""W&B Model Registry helpers.

Models are versioned in the ``ekg-digitizer`` collection of the W&B "Model"
registry. The ``staging`` and ``production`` aliases mark which version is
under review and which one is approved for serving. An alias points to only
one version at a time, so promoting a version moves the alias to it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

REGISTRY = "wandb-registry-model"
COLLECTION = "ekg-digitizer"
MODEL_TYPE = "model"
ALIASES = ("staging", "production")


def collection_path(collection: str = COLLECTION) -> str:
    return f"{REGISTRY}/{collection}"


def _check_alias(alias: str | None) -> None:
    if alias is not None and alias not in ALIASES:
        raise ValueError(f"Alias must be one of {ALIASES}, got {alias!r}")


def log_model(run, path: str | Path, name: str = COLLECTION, metadata: dict[str, Any] | None = None):
    """Log a checkpoint file or directory as a model artifact of ``run``.

    Training scripts call this so their runs can be registered later.
    """
    import wandb

    path = Path(path)
    artifact = wandb.Artifact(name, type=MODEL_TYPE, metadata=metadata or {})
    artifact.add_dir(str(path)) if path.is_dir() else artifact.add_file(str(path))
    return run.log_artifact(artifact)


def find_model_artifact(run):
    """Return the most recent model artifact logged by a W&B API ``run``."""
    models = [a for a in run.logged_artifacts() if a.type == MODEL_TYPE]
    if not models:
        raise ValueError(f"Run {run.path} has no logged artifact of type '{MODEL_TYPE}'")
    return max(models, key=lambda a: a.created_at)


def register(run, alias: str | None = None, collection: str = COLLECTION):
    """Register the model from a training run in the registry.

    The run's config and summary metrics are stored on the model's metadata,
    so every registered version records how it was trained and how it scored.
    """
    _check_alias(alias)
    artifact = find_model_artifact(run)
    metrics = {k: v for k, v in run.summary_metrics.items() if isinstance(v, (int, float)) and not k.startswith("_")}
    artifact.metadata.update({"run": "/".join(run.path), "config": dict(run.config), "metrics": metrics})
    artifact.save()
    return artifact.link(collection_path(collection), aliases=[alias] if alias else None)


def promote(api, version: str, alias: str, collection: str = COLLECTION):
    """Point ``alias`` at ``version`` (e.g. ``"v3"``) of the registered model."""
    if alias is None:
        raise ValueError("An alias is required to promote a model")
    _check_alias(alias)
    artifact = api.artifact(f"{collection_path(collection)}:{version}")
    if alias not in artifact.aliases:
        artifact.aliases.append(alias)
        artifact.save()
    return artifact
