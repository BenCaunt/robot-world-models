from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class VisualDataError(RuntimeError):
    """Raised when video and numeric observations cannot be aligned exactly."""


@dataclass(frozen=True)
class CachedVisualEpisode:
    episode_id: str
    features: np.ndarray
    frames: np.ndarray
    states: np.ndarray
    actions: np.ndarray


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_video(
    path: Path,
    *,
    start_seconds: float = 0.0,
    frame_count: int | None = None,
) -> Iterator[np.ndarray]:
    try:
        import av
    except ImportError as error:
        raise VisualDataError(
            "visual data support requires: uv sync --extra train --extra lerobot --extra vision"
        ) from error

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if start_seconds > 0:
            target_pts = int(start_seconds / float(stream.time_base))
            container.seek(target_pts, stream=stream, any_frame=False, backward=True)
        emitted = 0
        tolerance = 0.5 / float(stream.average_rate)
        for frame in container.decode(stream):
            timestamp = (
                float(frame.pts * stream.time_base) if frame.pts is not None else None
            )
            if start_seconds > 0 and timestamp is None:
                raise VisualDataError(
                    f"segmented video frame has no timestamp: {path}"
                )
            if timestamp is not None and timestamp + tolerance < start_seconds:
                continue
            yield frame.to_ndarray(format="rgb24")
            emitted += 1
            if frame_count is not None and emitted == frame_count:
                return


class DinoV2FeatureEncoder:
    """Frozen, revision-pinned DINOv2 encoder with spatial patch pooling."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str,
        patch_pool_grid: int,
        expected_input_size: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise VisualDataError(
                "DINOv2 support requires: uv sync --extra train --extra vision"
            ) from error
        self._torch = torch
        self.device = device
        self.patch_pool_grid = patch_pool_grid
        self.model_id = model_id
        self.revision = revision
        self.expected_input_size = expected_input_size
        self.processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision).to(device).eval()
        self.model.requires_grad_(False)
        self.hidden_size = int(self.model.config.hidden_size)
        self.patch_size = int(self.model.config.patch_size)

    def _process(self, frames: Sequence[np.ndarray]):
        pixel_values = self.processor(
            images=list(frames),
            return_tensors="pt",
        )["pixel_values"]
        if tuple(pixel_values.shape[-2:]) != (
            self.expected_input_size,
            self.expected_input_size,
        ):
            raise VisualDataError(
                "encoder processor output differs from the recipe input_size: "
                f"{tuple(pixel_values.shape[-2:])}"
            )
        return pixel_values

    def cache_contract(self, *, output_size: int) -> dict[str, object]:
        return {
            "modelId": self.model_id,
            "revision": self.revision,
            "inputSize": self.expected_input_size,
            "patchPoolGrid": self.patch_pool_grid,
            "outputSize": output_size,
            "rgbTarget": "exact-encoder-view-denormalized-bilinear-resize",
        }

    def _encode_pixels(self, pixel_values) -> np.ndarray:
        torch = self._torch
        from torch.nn import functional as functional

        pixel_values = pixel_values.to(self.device)
        with torch.inference_mode():
            tokens = self.model(pixel_values=pixel_values).last_hidden_state[:, 1:]
            patch_count = tokens.shape[1]
            patch_grid = int(patch_count**0.5)
            if patch_grid**2 != patch_count:
                raise VisualDataError(f"DINOv2 returned a non-square patch grid: {patch_count}")
            spatial = tokens.transpose(1, 2).reshape(
                tokens.shape[0],
                tokens.shape[2],
                patch_grid,
                patch_grid,
            )
            pooled = functional.adaptive_avg_pool2d(
                spatial,
                (self.patch_pool_grid, self.patch_pool_grid),
            )
            pooled = pooled.flatten(start_dim=2).transpose(1, 2)
            pooled = functional.normalize(pooled, dim=-1)
        return pooled.cpu().numpy().astype(np.float16)

    def encode(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        return self._encode_pixels(self._process(frames))

    def encode_with_targets(
        self,
        frames: Sequence[np.ndarray],
        *,
        output_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode and derive RGB targets from the exact normalized encoder view."""
        torch = self._torch
        from torch.nn import functional as functional

        pixel_values = self._process(frames)
        features = self._encode_pixels(pixel_values)
        mean = torch.tensor(self.processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std).view(1, 3, 1, 1)
        encoder_view = (pixel_values * std + mean).clamp(0, 1)
        targets = functional.interpolate(
            encoder_view,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )
        targets = targets.mul(255).round().byte().permute(0, 2, 3, 1).numpy()
        return features, targets


def cache_visual_episode(
    *,
    episode_id: str,
    video_path: Path,
    video_start_seconds: float = 0.0,
    video_frame_count: int | None = None,
    states: np.ndarray,
    actions: np.ndarray,
    encoder: DinoV2FeatureEncoder,
    encoder_batch_size: int,
    output_size: int,
    cache_path: Path,
) -> dict[str, object]:
    cache_contract = {
        **encoder.cache_contract(output_size=output_size),
        "videoStartSeconds": video_start_seconds,
        "videoFrameCount": video_frame_count,
    }
    cache_contract_json = json.dumps(cache_contract, sort_keys=True, separators=(",", ":"))
    if cache_path.exists():
        with np.load(cache_path) as cached:
            frame_count = int(cached["features"].shape[0])
            cache_version = int(cached["cache_version"]) if "cache_version" in cached.files else 1
            cached_contract = (
                str(cached["cache_contract_json"])
                if "cache_contract_json" in cached.files
                else None
            )
            if (
                cache_version == 4
                and cached_contract == cache_contract_json
                and frame_count != len(states)
            ):
                raise VisualDataError(
                    f"stale feature cache for episode {episode_id}: "
                    f"{frame_count} frames != {len(states)} states"
                )
        if cache_version == 4 and cached_contract == cache_contract_json:
            return {
                "episodeId": episode_id,
                "path": str(cache_path),
                "frames": frame_count,
                "sha256": file_sha256(cache_path),
                "cacheVersion": cache_version,
                "reused": True,
            }

    feature_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    pending: list[np.ndarray] = []
    decoded_count = 0
    for frame in decode_video(
        video_path,
        start_seconds=video_start_seconds,
        frame_count=video_frame_count,
    ):
        decoded_count += 1
        pending.append(frame)
        if len(pending) == encoder_batch_size:
            features, targets = encoder.encode_with_targets(
                pending,
                output_size=output_size,
            )
            feature_batches.append(features)
            target_batches.append(targets)
            pending.clear()
    if pending:
        features, targets = encoder.encode_with_targets(
            pending,
            output_size=output_size,
        )
        feature_batches.append(features)
        target_batches.append(targets)
    if decoded_count != len(states) or len(actions) != len(states):
        raise VisualDataError(
            f"episode {episode_id} is not frame-aligned: "
            f"video={decoded_count}, state={len(states)}, action={len(actions)}"
        )
    features = np.concatenate(feature_batches)
    frames = np.concatenate(target_batches)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        cache_version=np.asarray(4),
        cache_contract_json=np.asarray(cache_contract_json),
        features=features,
        frames=frames,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
    )
    return {
        "episodeId": episode_id,
        "path": str(cache_path),
        "frames": decoded_count,
        "featuresShape": list(features.shape),
        "framesShape": list(frames.shape),
        "cacheVersion": 4,
        "cacheContract": cache_contract,
        "rgbTargetTransform": "exact DINOv2 processor view, denormalized then resized",
        "bytes": cache_path.stat().st_size,
        "sha256": file_sha256(cache_path),
        "reused": False,
    }


def load_cached_visual_episode(episode_id: str, cache_path: Path) -> CachedVisualEpisode:
    with np.load(cache_path) as cached:
        return CachedVisualEpisode(
            episode_id=episode_id,
            features=np.asarray(cached["features"], dtype=np.float32),
            frames=np.asarray(cached["frames"], dtype=np.uint8),
            states=np.asarray(cached["states"], dtype=np.float32),
            actions=np.asarray(cached["actions"], dtype=np.float32),
        )
