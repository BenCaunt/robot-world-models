from __future__ import annotations

import json

from robot_world_models.catalog import (
    build_catalog,
    render_catalog,
    repository_root,
    validate_repository,
)


def test_repository_manifests_validate_and_cross_reference() -> None:
    loaded = validate_repository()

    assert {manifest.kind for _, manifest in loaded} == {
        "dataset",
        "robot",
        "joint-mapping",
        "recipe",
    }
    assert {manifest.id for _, manifest in loaded} == {
        "qb1t-so101-teleop-cubes",
        "nashmo-so101",
        "so-arm101",
        "nashmo-so101-to-so-arm101",
        "so101-dinov2-transformer-h5-poc",
        "so101-dinov2-transformer-poc",
        "so101-dinov2-visual-8x8-poc",
        "so101-dinov2-visual-h5-poc",
        "so101-dinov2-visual-poc",
        "so101-state-dynamics-poc",
    }


def test_catalog_is_deterministic() -> None:
    first = render_catalog()
    second = render_catalog()

    assert first == second
    payload = json.loads(first)
    assert payload["schemaVersion"] == "robot-world-models.catalog.v1"
    assert payload["contentSha256"]
    assert len(payload["entries"]) == 10


def test_generated_files_are_current() -> None:
    assert build_catalog(check=True) == []
    assert (repository_root() / "catalog" / "catalog.json").exists()
