# ==============================================================================
# File Name : registry.py
# Purpose   : Saves, registers, and promotes models in the W&B Model Registry
# ==============================================================================
"""W&B Model Registry helpers.

Registered models are numbered versions (v0, v1, ...). Aliases mark status:
    staging    : under review
    production : approved for use
Each alias is on only one version at a time.
"""

import wandb

REGISTRY_PATH = "wandb-registry-model/ekg-digitizer"
ALIASES = ("staging", "production")


def _check_alias(alias: str) -> None:
    if alias not in ALIASES:
        raise ValueError(f"Alias must be one of {ALIASES}, got {alias!r}")


def log_model(run, checkpoint: str) -> wandb.Artifact:
    """Save a checkpoint (e.g. best_model.pth) to a training run."""
    artifact = wandb.Artifact("ekg-digitizer", type="model")
    artifact.add_file(checkpoint)
    return run.log_artifact(artifact)


def register(run, alias: str | None = None) -> wandb.Artifact:
    """Add a training run's newest model to the registry, with its config and metrics."""
    if alias:
        _check_alias(alias)

    models = [a for a in run.logged_artifacts() if a.type == "model"]
    if not models:
        raise ValueError(f"Run {run.id} has no saved model")
    model = max(models, key=lambda a: a.created_at)

    # Keep numeric metrics only; keys starting with "_" are W&B's own (e.g. _step)
    metrics = {
        name: value
        for name, value in run.summary_metrics.items()
        if isinstance(value, (int, float)) and not name.startswith("_")
    }
    model.metadata.update(config=dict(run.config), metrics=metrics)
    model.save()

    return model.link(REGISTRY_PATH, aliases=[alias] if alias else None)


def promote(api, version: str, alias: str) -> wandb.Artifact:
    """Move an alias to a registered version (e.g. "v3")."""
    _check_alias(alias)

    model = api.artifact(f"{REGISTRY_PATH}:{version}")
    if alias not in model.aliases:
        model.aliases.append(alias)
        model.save()
    return model
