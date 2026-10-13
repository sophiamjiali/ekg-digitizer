from types import SimpleNamespace

import pytest

from ekg_digitizer import registry


class FakeArtifact:
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
        return SimpleNamespace(version="v0")


def fake_run(*artifacts):
    return SimpleNamespace(
        path=["team", "ekg", "abc123"],
        config={"lr": 1e-4},
        summary_metrics={"val/snr": 22.6, "val/mae": 0.04, "_step": 10, "note": "text"},
        logged_artifacts=lambda: list(artifacts),
    )


def test_register_links_latest_model_with_config_and_metrics():
    old, new = FakeArtifact(created_at="2026-01-01"), FakeArtifact(created_at="2026-02-01")
    registry.register(fake_run(FakeArtifact(type="dataset"), new, old), alias="staging")

    assert new.linked == ("wandb-registry-model/ekg-digitizer", ["staging"])
    assert new.metadata == {
        "run": "team/ekg/abc123",
        "config": {"lr": 1e-4},
        "metrics": {"val/snr": 22.6, "val/mae": 0.04},
    }
    assert new.saved == 1
    assert old.linked is None


def test_register_without_alias():
    model = FakeArtifact()
    registry.register(fake_run(model))
    assert model.linked == ("wandb-registry-model/ekg-digitizer", None)


def test_register_requires_a_model_artifact():
    with pytest.raises(ValueError, match="no logged artifact"):
        registry.register(fake_run(FakeArtifact(type="dataset")))


@pytest.mark.parametrize("alias", ["prod", "latest"])
def test_unknown_alias_is_rejected(alias):
    with pytest.raises(ValueError, match="Alias must be one of"):
        registry.register(fake_run(FakeArtifact()), alias=alias)


def test_promote_adds_alias_to_version():
    artifact = FakeArtifact(aliases=["latest"])
    api = SimpleNamespace(artifact=lambda name: artifact if name == "wandb-registry-model/ekg-digitizer:v3" else None)

    registry.promote(api, "v3", "production")
    assert artifact.aliases == ["latest", "production"]
    assert artifact.saved == 1

    registry.promote(api, "v3", "production")  # already promoted: no change
    assert artifact.saved == 1
