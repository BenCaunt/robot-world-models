from __future__ import annotations

import numpy as np
import pytest

from robot_world_models.visual_data import CachedVisualEpisode, cache_visual_episode
from robot_world_models.visual_training import _write_rollout_preview, visual_window_refs


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
        lambda _: iter(source_frames),
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
        assert int(cached["cache_version"]) == 3
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
