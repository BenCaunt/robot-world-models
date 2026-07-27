from __future__ import annotations

import numpy as np
import pytest

from robot_world_models.training import Normalization, TrainingSpikeError
from robot_world_models.visual_data import CachedVisualEpisode, cache_visual_episode
from robot_world_models.visual_training import (
    _batch_arrays,
    _make_model,
    _write_rollout_preview,
    visual_window_refs,
)


def test_visual_windows_never_cross_episode_boundaries() -> None:
    episodes = [
        CachedVisualEpisode(
            episode_id=str(index),
            features=np.zeros((5, 16, 8), dtype=np.float32),
            frames=np.zeros((5, 64, 64, 3), dtype=np.uint8),
            states=np.zeros((5, 2), dtype=np.float32),
            actions=np.zeros((5, 2), dtype=np.float32),
        )
        for index in range(2)
    ]

    refs = visual_window_refs(episodes, context_frames=3)

    assert [(ref.episode, ref.target) for ref in refs] == [
        (0, 3),
        (0, 4),
        (1, 3),
        (1, 4),
    ]


def test_five_step_windows_and_targets_stay_inside_each_episode() -> None:
    episode = CachedVisualEpisode(
        episode_id="0",
        features=np.arange(9 * 4 * 2, dtype=np.float32).reshape(9, 4, 2),
        frames=np.zeros((9, 8, 8, 3), dtype=np.uint8),
        states=np.arange(18, dtype=np.float32).reshape(9, 2),
        actions=np.arange(18, dtype=np.float32).reshape(9, 2),
    )
    normalization = Normalization(
        state_mean=np.zeros(2, dtype=np.float32),
        state_std=np.ones(2, dtype=np.float32),
        action_mean=np.zeros(2, dtype=np.float32),
        action_std=np.ones(2, dtype=np.float32),
    )

    refs = visual_window_refs([episode], context_frames=3, rollout_horizon=5)
    arrays = _batch_arrays(
        [episode],
        refs,
        normalization,
        context_frames=3,
        rollout_horizon=5,
    )
    contexts, states, actions, target_features, target_states, target_frames = arrays

    assert [(ref.episode, ref.target) for ref in refs] == [(0, 3), (0, 4)]
    assert contexts.shape == (2, 3, 4, 2)
    assert states.shape == (2, 2)
    assert actions.shape == (2, 5, 2)
    assert target_features.shape == (2, 5, 4, 2)
    assert target_states.shape == (2, 5, 2)
    assert target_frames.shape == (2, 8, 8, 3)
    np.testing.assert_array_equal(actions[0], episode.actions[2:7])
    np.testing.assert_array_equal(target_states[0], episode.states[3:8])


def test_visual_model_shapes_and_decoder_range() -> None:
    torch = pytest.importorskip("torch")
    from robot_world_models.models.visual_latent import VisualLatentDynamics

    model = VisualLatentDynamics(
        state_dimension=2,
        action_dimension=2,
        latent_dimension=8,
        context_frames=3,
        patch_grid=4,
        output_size=64,
        hidden_dimension=16,
        hidden_layers=2,
    )
    context = torch.nn.functional.normalize(torch.randn(2, 3, 16, 8), dim=-1)

    features, state = model(context, torch.zeros(2, 2), torch.zeros(2, 2))
    images = model.decode(features)

    assert features.shape == (2, 16, 8)
    assert state.shape == (2, 2)
    assert images.shape == (2, 3, 64, 64)
    assert images.min() >= 0
    assert images.max() <= 1
    assert not any(
        isinstance(module, torch.nn.ConvTranspose2d)
        for module in model.decoder.modules()
    )

    higher_resolution_model = VisualLatentDynamics(
        state_dimension=2,
        action_dimension=2,
        latent_dimension=8,
        context_frames=3,
        patch_grid=8,
        output_size=64,
        hidden_dimension=16,
        hidden_layers=2,
    )
    higher_resolution_features = torch.nn.functional.normalize(
        torch.randn(1, 3, 64, 8),
        dim=-1,
    )
    higher_resolution_prediction, _ = higher_resolution_model(
        higher_resolution_features,
        torch.zeros(1, 2),
        torch.zeros(1, 2),
    )

    assert higher_resolution_model.decode(higher_resolution_prediction).shape == (
        1,
        3,
        64,
        64,
    )


def test_visual_transformer_mixes_spatial_tokens() -> None:
    torch = pytest.importorskip("torch")
    from robot_world_models.models.visual_latent import VisualLatentDynamics
    from robot_world_models.models.visual_transformer import (
        VisualSpatiotemporalTransformer,
    )

    torch.manual_seed(7)
    common = {
        "state_dimension": 2,
        "action_dimension": 2,
        "latent_dimension": 8,
        "context_frames": 3,
        "patch_grid": 2,
        "output_size": 8,
        "hidden_dimension": 32,
    }
    tokenwise = VisualLatentDynamics(**common, hidden_layers=2).eval()
    transformer = VisualSpatiotemporalTransformer(
        **common,
        hidden_layers=4,
        attention_heads=4,
    ).eval()
    torch.nn.init.normal_(transformer.feature_head.weight, std=0.02)
    context = torch.nn.functional.normalize(torch.randn(1, 3, 4, 8), dim=-1)
    perturbed = context.clone()
    perturbed[:, :, 0] = torch.nn.functional.normalize(
        perturbed[:, :, 0] + 2,
        dim=-1,
    )
    state = torch.zeros(1, 2)
    action = torch.zeros(1, 2)

    tokenwise_before, _ = tokenwise(context, state, action)
    tokenwise_after, _ = tokenwise(perturbed, state, action)
    transformer_before, _ = transformer(context, state, action)
    transformer_after, _ = transformer(perturbed, state, action)

    torch.testing.assert_close(tokenwise_before[:, 1], tokenwise_after[:, 1])
    assert not torch.allclose(transformer_before[:, 1], transformer_after[:, 1])
    assert transformer.decode(transformer_before).shape == (1, 3, 8, 8)


def test_visual_transformer_starts_at_persistence() -> None:
    torch = pytest.importorskip("torch")
    from robot_world_models.models.visual_transformer import (
        VisualSpatiotemporalTransformer,
    )

    model = VisualSpatiotemporalTransformer(
        state_dimension=2,
        action_dimension=2,
        latent_dimension=8,
        context_frames=3,
        patch_grid=2,
        output_size=8,
        hidden_dimension=32,
        hidden_layers=4,
        attention_heads=4,
    ).eval()
    context = torch.nn.functional.normalize(torch.randn(2, 3, 4, 8), dim=-1)
    state = torch.randn(2, 2)

    features, predicted_state = model(context, state, torch.randn(2, 2))

    torch.testing.assert_close(features, context[:, -1])
    torch.testing.assert_close(predicted_state, state)


def test_visual_model_factory_uses_recipe_implementation() -> None:
    pytest.importorskip("torch")
    import yaml

    from robot_world_models.catalog import repository_root
    from robot_world_models.contracts import MANIFEST_ADAPTER
    from robot_world_models.models.visual_transformer import (
        VisualSpatiotemporalTransformer,
    )

    path = (
        repository_root()
        / "catalog"
        / "recipes"
        / "so101-dinov2-transformer-poc.yaml"
    )
    recipe = MANIFEST_ADAPTER.validate_python(yaml.safe_load(path.read_text()))

    assert isinstance(_make_model(recipe), VisualSpatiotemporalTransformer)


def test_visual_model_factory_rejects_unknown_implementation() -> None:
    pytest.importorskip("torch")
    import yaml

    from robot_world_models.catalog import repository_root
    from robot_world_models.contracts import MANIFEST_ADAPTER

    path = (
        repository_root()
        / "catalog"
        / "recipes"
        / "so101-dinov2-transformer-poc.yaml"
    )
    recipe = MANIFEST_ADAPTER.validate_python(yaml.safe_load(path.read_text()))
    unknown_model = recipe.model.model_copy(update={"implementation": "example:Unknown"})
    unknown_recipe = recipe.model_copy(update={"model": unknown_model})

    with pytest.raises(TrainingSpikeError, match="unsupported visual model implementation"):
        _make_model(unknown_recipe)


def test_feature_cache_uses_encoder_aligned_rgb_targets(tmp_path, monkeypatch) -> None:
    source_frames = [
        np.full((8, 12, 3), value, dtype=np.uint8)
        for value in (10, 20)
    ]
    encoder_targets = np.stack(
        [
            np.full((4, 4, 3), value, dtype=np.uint8)
            for value in (101, 202)
        ]
    )

    class FakeEncoder:
        def cache_contract(self, *, output_size):
            return {"fake": True, "outputSize": output_size}

        def encode_with_targets(self, frames, *, output_size):
            assert frames == source_frames
            assert output_size == 4
            return np.zeros((2, 4, 8), dtype=np.float16), encoder_targets

    monkeypatch.setattr(
        "robot_world_models.visual_data.decode_video",
        lambda _, **__: iter(source_frames),
    )
    cache_path = tmp_path / "episode.npz"
    receipt = cache_visual_episode(
        episode_id="0",
        video_path=tmp_path / "ignored.mp4",
        states=np.zeros((2, 2), dtype=np.float32),
        actions=np.zeros((2, 2), dtype=np.float32),
        encoder=FakeEncoder(),
        encoder_batch_size=2,
        output_size=4,
        cache_path=cache_path,
    )

    with np.load(cache_path) as cached:
        assert int(cached["cache_version"]) == 4
        np.testing.assert_array_equal(cached["frames"], encoder_targets)
    assert receipt["rgbTargetTransform"].startswith("exact DINOv2")

    reused = cache_visual_episode(
        episode_id="0",
        video_path=tmp_path / "ignored.mp4",
        states=np.zeros((2, 2), dtype=np.float32),
        actions=np.zeros((2, 2), dtype=np.float32),
        encoder=FakeEncoder(),
        encoder_batch_size=2,
        output_size=4,
        cache_path=cache_path,
    )

    assert reused["reused"] is True


def test_rollout_preview_is_written(tmp_path) -> None:
    actual = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    predicted = np.full((3, 8, 8, 3), 255, dtype=np.uint8)

    path = _write_rollout_preview(
        output_path=tmp_path / "preview.png",
        actual_frames=actual,
        predicted_frames=predicted,
    )

    assert path.exists()
    assert path.stat().st_size > 0
