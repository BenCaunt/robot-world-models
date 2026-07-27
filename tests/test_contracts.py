from __future__ import annotations

import copy

import pytest
import yaml
from pydantic import ValidationError

from robot_world_models.catalog import repository_root
from robot_world_models.contracts import MANIFEST_ADAPTER


def test_provisional_joint_mapping_cannot_enable_animation() -> None:
    path = (
        repository_root()
        / "catalog"
        / "mappings"
        / "nashmo-so101-to-so-arm101.yaml"
    )
    payload = yaml.safe_load(path.read_text())
    invalid = copy.deepcopy(payload)
    invalid["mapping"]["status"] = "provisional"

    with pytest.raises(ValidationError, match="validated joint mapping"):
        MANIFEST_ADAPTER.validate_python(invalid)


def test_so101_mapping_has_validated_partial_animation() -> None:
    path = (
        repository_root()
        / "catalog"
        / "mappings"
        / "nashmo-so101-to-so-arm101.yaml"
    )
    manifest = MANIFEST_ADAPTER.validate_python(yaml.safe_load(path.read_text()))

    assert manifest.mapping.status == "validated"
    assert manifest.mapping.coverage == "partial"
    assert manifest.mapping.animate_in_rerun is True
    assert manifest.mapping.unmapped_features == ["main_gripper"]
    assert set(manifest.mapping.entries) == {
        "main_shoulder_pan",
        "main_shoulder_lift",
        "main_elbow_flex",
        "main_wrist_flex",
        "main_wrist_roll",
    }


def test_recipe_uses_episode_split() -> None:
    path = repository_root() / "catalog" / "recipes" / "so101-state-dynamics-poc.yaml"
    payload = yaml.safe_load(path.read_text())
    manifest = MANIFEST_ADAPTER.validate_python(payload)

    assert manifest.kind == "recipe"
    assert manifest.mixture.split.unit == "episode"
    assert manifest.evaluation.rerun_required is True
