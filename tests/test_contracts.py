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


def test_visual_recipe_pins_encoder_and_requires_rgb() -> None:
    path = repository_root() / "catalog" / "recipes" / "so101-dinov2-visual-poc.yaml"
    payload = yaml.safe_load(path.read_text())
    manifest = MANIFEST_ADAPTER.validate_python(payload)

    assert manifest.model.vision.camera == "observation.images.laptop"
    assert len(manifest.model.vision.encoder.revision) == 40
    assert manifest.model.vision.encoder.warmhub_resolution == "registry-gap"
    assert "sensor/rgb" in manifest.modalities.required


def test_five_step_visual_recipe_declares_unrolled_objective() -> None:
    path = (
        repository_root()
        / "catalog"
        / "recipes"
        / "so101-dinov2-visual-h5-poc.yaml"
    )
    manifest = MANIFEST_ADAPTER.validate_python(yaml.safe_load(path.read_text()))

    assert manifest.intent.horizon_steps == 5
    assert manifest.model.vision.encoder.patch_pool_grid == 4
    assert manifest.model.vision.training_rollout_horizon == 5
    assert manifest.model.vision.rollout_loss_discount == 0.8


def test_visual_recipe_rejects_mismatched_training_horizon() -> None:
    path = (
        repository_root()
        / "catalog"
        / "recipes"
        / "so101-dinov2-visual-h5-poc.yaml"
    )
    payload = yaml.safe_load(path.read_text())
    payload["intent"]["horizon_steps"] = 1

    with pytest.raises(
        ValidationError,
        match="intent horizon_steps must equal training_rollout_horizon",
    ):
        MANIFEST_ADAPTER.validate_python(payload)


def test_visual_transformer_recipe_declares_spatial_attention() -> None:
    path = (
        repository_root()
        / "catalog"
        / "recipes"
        / "so101-dinov2-transformer-h5-poc.yaml"
    )
    manifest = MANIFEST_ADAPTER.validate_python(yaml.safe_load(path.read_text()))

    assert manifest.model.family == "action-conditioned-spatiotemporal-transformer"
    assert manifest.model.vision.attention_heads == 8
    assert manifest.model.vision.predictor_hidden_dimension == 256
    assert manifest.model.vision.predictor_hidden_layers == 4


def test_visual_transformer_rejects_invalid_attention_width() -> None:
    path = (
        repository_root()
        / "catalog"
        / "recipes"
        / "so101-dinov2-transformer-poc.yaml"
    )
    payload = yaml.safe_load(path.read_text())
    payload["model"]["vision"]["attention_heads"] = 7

    with pytest.raises(
        ValidationError,
        match="predictor_hidden_dimension must be divisible by attention_heads",
    ):
        MANIFEST_ADAPTER.validate_python(payload)


def test_visual_transformer_requires_attention_heads() -> None:
    path = (
        repository_root()
        / "catalog"
        / "recipes"
        / "so101-dinov2-transformer-poc.yaml"
    )
    payload = yaml.safe_load(path.read_text())
    del payload["model"]["vision"]["attention_heads"]

    with pytest.raises(
        ValidationError,
        match="visual transformer requires attention_heads",
    ):
        MANIFEST_ADAPTER.validate_python(payload)


def test_visual_transformer_requires_encoder_and_decoder_layers() -> None:
    path = (
        repository_root()
        / "catalog"
        / "recipes"
        / "so101-dinov2-transformer-poc.yaml"
    )
    payload = yaml.safe_load(path.read_text())
    payload["model"]["vision"]["predictor_hidden_layers"] = 1

    with pytest.raises(
        ValidationError,
        match="visual transformer requires at least two predictor_hidden_layers",
    ):
        MANIFEST_ADAPTER.validate_python(payload)
