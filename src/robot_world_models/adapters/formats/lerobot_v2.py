from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from robot_world_models.adapters.base import CanonicalEpisode, SourceReceipt


class LeRobotV2Error(RuntimeError):
    """Raised when a LeRobot v2.1 dataset violates the canonical episode contract."""


class LeRobotV2Adapter:
    """Read state/action episodes directly from LeRobot v2.1 Parquet files."""

    adapter_id = "lerobot-v2"

    def __init__(
        self,
        *,
        dataset_wref: str,
        robot_wref: str,
        episode_indices: Sequence[int] | None = None,
    ) -> None:
        self.dataset_wref = dataset_wref
        self.robot_wref = robot_wref
        self.episode_indices = list(episode_indices) if episode_indices is not None else None

    @staticmethod
    def _info(source: SourceReceipt) -> dict[str, Any]:
        path = source.destination / "meta" / "info.json"
        if not path.exists():
            raise LeRobotV2Error(f"missing metadata: {path}")
        info = json.loads(path.read_text())
        if info.get("codebase_version") != "v2.1":
            raise LeRobotV2Error(
                "lerobot-v2 only accepts codebase_version v2.1; "
                f"found {info.get('codebase_version')!r}"
            )
        return info

    def inspect(self, source: SourceReceipt) -> dict[str, Any]:
        info = self._info(source)
        features = info.get("features", {})
        for required in ("observation.state", "action", "episode_index", "frame_index"):
            if required not in features:
                raise LeRobotV2Error(f"missing required feature: {required}")
        return {
            "codebaseVersion": info["codebase_version"],
            "fps": info["fps"],
            "totalEpisodes": info["total_episodes"],
            "totalFrames": info["total_frames"],
            "state": features["observation.state"],
            "action": features["action"],
            "cameraKeys": sorted(
                key for key, value in features.items() if value.get("dtype") == "video"
            ),
        }

    @staticmethod
    def materialization_patterns(
        upstream_files: Sequence[str],
        episode_indices: Sequence[int],
        cameras: Sequence[str] = (),
    ) -> list[str]:
        selected_names = {
            f"episode_{episode_index:06d}.parquet" for episode_index in episode_indices
        }
        episode_paths = sorted(
            path
            for path in upstream_files
            if path.startswith("data/")
            and path.rsplit("/", maxsplit=1)[-1] in selected_names
        )
        if len(episode_paths) != len(selected_names):
            missing = sorted(
                selected_names
                - {path.rsplit("/", maxsplit=1)[-1] for path in episode_paths}
            )
            raise LeRobotV2Error(f"preflight did not find selected episodes: {missing}")
        video_paths: list[str] = []
        for camera in cameras:
            selected_video_paths = sorted(
                path
                for path in upstream_files
                if path.startswith("videos/")
                and f"/{camera}/" in path
                and path.rsplit("/", maxsplit=1)[-1].replace(".mp4", ".parquet")
                in selected_names
            )
            if len(selected_video_paths) != len(selected_names):
                found = {
                    path.rsplit("/", maxsplit=1)[-1].replace(".mp4", ".parquet")
                    for path in selected_video_paths
                }
                missing = sorted(selected_names - found)
                raise LeRobotV2Error(
                    f"preflight did not find camera {camera!r} episodes: {missing}"
                )
            video_paths.extend(selected_video_paths)
        return ["meta/*", *episode_paths, *video_paths]

    @classmethod
    def video_path(
        cls,
        source: SourceReceipt,
        *,
        camera: str,
        episode_index: int,
    ):
        info = cls._info(source)
        feature = info.get("features", {}).get(camera)
        if feature is None or feature.get("dtype") != "video":
            raise LeRobotV2Error(f"camera is not declared as a video feature: {camera}")
        template = info["video_path"]
        relative = template.format(
            episode_chunk=episode_index // int(info["chunks_size"]),
            video_key=camera,
            episode_index=episode_index,
        )
        path = source.destination / relative
        if not path.exists():
            raise LeRobotV2Error(f"missing episode video: {relative}")
        return path

    def episodes(self, source: SourceReceipt) -> Iterator[CanonicalEpisode]:
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise LeRobotV2Error(
                "LeRobot v2 support requires: uv sync --extra lerobot"
            ) from error

        info = self._info(source)
        total_episodes = int(info["total_episodes"])
        indices = (
            self.episode_indices
            if self.episode_indices is not None
            else list(range(total_episodes))
        )
        template = info["data_path"]
        chunk_size = int(info["chunks_size"])
        state_dimension = int(info["features"]["observation.state"]["shape"][0])
        action_dimension = int(info["features"]["action"]["shape"][0])
        for episode_index in indices:
            if not 0 <= episode_index < total_episodes:
                raise LeRobotV2Error(f"episode index out of range: {episode_index}")
            relative = template.format(
                episode_chunk=episode_index // chunk_size,
                episode_index=episode_index,
            )
            path = source.destination / relative
            if not path.exists():
                raise LeRobotV2Error(f"missing episode data: {relative}")
            table = parquet.read_table(
                path,
                columns=[
                    "observation.state",
                    "action",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                ],
            )
            episode_values = (
                table.column("episode_index")
                .combine_chunks()
                .to_numpy(zero_copy_only=False)
            )
            if not np.all(episode_values == episode_index):
                raise LeRobotV2Error(f"mixed episode indices in {relative}")
            frame_indices = (
                table.column("frame_index")
                .combine_chunks()
                .to_numpy(zero_copy_only=False)
            )
            if not np.array_equal(frame_indices, np.arange(table.num_rows)):
                raise LeRobotV2Error(f"non-contiguous frame indices in {relative}")
            timestamps = (
                table.column("timestamp")
                .combine_chunks()
                .to_numpy(zero_copy_only=False)
                .astype(np.float64, copy=False)
            )
            if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
                raise LeRobotV2Error(f"timestamps are not strictly increasing in {relative}")
            states = np.asarray(
                table.column("observation.state").combine_chunks().to_pylist(),
                dtype=np.float32,
            )
            actions = np.asarray(
                table.column("action").combine_chunks().to_pylist(),
                dtype=np.float32,
            )
            if states.shape != (table.num_rows, state_dimension):
                raise LeRobotV2Error(f"unexpected state shape in {relative}: {states.shape}")
            if actions.shape != (table.num_rows, action_dimension):
                raise LeRobotV2Error(f"unexpected action shape in {relative}: {actions.shape}")
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise LeRobotV2Error(f"non-finite state/action value in {relative}")
            yield CanonicalEpisode(
                episode_id=str(episode_index),
                robot_wref=self.robot_wref,
                dataset_wref=self.dataset_wref,
                timestamps_seconds=timestamps,
                observations={"state": states},
                actions=actions,
                task=None,
                modality_mask={
                    "sensor/proprioception": True,
                    "state-space/joint-position": True,
                    "action-space/joint-position": True,
                    "sensor/rgb": False,
                },
            )
