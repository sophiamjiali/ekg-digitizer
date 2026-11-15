from types import SimpleNamespace

import pytest

from ekg_digitizer import registry


class FakeModel:
    """Stands in for a W&B artifact so tests don't need an account."""

    def __init__(self, type="model", created_at="2026-01-01", aliases=None):
        self.type = type
        self.created_at = created_at
        self.aliases = aliases or []
        self.metadata = {}
        self.saved = 0
        self.linked = None

    def save(self):
        self.saved += 1

    def link(self, target_path, aliases=None):
        self.linked = (target_path, aliases)
        return self


def fake_run(*artifacts):
    return SimpleNamespace(
        id="abc123",
        config={"lr": 1e-4},
        summary_metrics={"val/snr": 12.5, "loss": 0.8, "_step": 10, "note": "text"},
        logged_artifacts=lambda: list(artifacts),
    )


def test_register_adds_newest_model_with_config_and_metrics():
    old, new = FakeModel(created_at="2026-01-01"), FakeModel(created_at="2026-02-01")
    registry.register(fake_run(FakeModel(type="dataset"), new, old), alias="staging")

    assert new.linked == (registry.REGISTRY_PATH, ["staging"])
    assert new.metadata == {"config": {"lr": 1e-4}, "metrics": {"val/snr": 12.5, "loss": 0.8}}
    assert old.linked is None


def test_register_without_alias():
    model = FakeModel()
    registry.register(fake_run(model))
    assert model.linked == (registry.REGISTRY_PATH, None)


def test_register_requires_a_saved_model():
    with pytest.raises(ValueError, match="no saved model"):
        registry.register(fake_run(FakeModel(type="dataset")))


@pytest.mark.parametrize("alias", ["prod", "latest"])
def test_unknown_alias_is_rejected(alias):
    with pytest.raises(ValueError, match="Alias must be one of"):
        registry.register(fake_run(FakeModel()), alias=alias)


def test_promote_adds_alias_once():
    model = FakeModel(aliases=["latest"])
    api = SimpleNamespace(artifact=lambda path: model if path == f"{registry.REGISTRY_PATH}:v3" else None)

    registry.promote(api, "v3", "production")
    registry.promote(api, "v3", "production")

    assert model.aliases == ["latest", "production"]
    assert model.saved == 1
