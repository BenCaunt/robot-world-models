from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_world_models.adapters.base import (
    CanonicalEpisode,
    SourceReceipt,
    VideoSegment,
)


class LeRobotV3CollectionError(RuntimeError):
    """Raised when a nested LeRobot v3 collection violates its declared contract."""


@dataclass(frozen=True)
class _EpisodeLocation:
    episode_id: str
    member_root: str
    local_episode_index: int


class LeRobotV3CollectionAdapter:
    """Read a repository containing multiple homogeneous LeRobot v3 datasets."""

    adapter_id = "lerobot-v3-nested"

    def __init__(
        self,
        *,
        dataset_wref: str,
        robot_wref: str,
        member_roots: Sequence[str],
        max_episodes: int | None = None,
    ) -> None:
        if not member_roots:
            raise LeRobotV3CollectionError("at least one member root is required")
        if len(set(member_roots)) != len(member_roots):
            raise LeRobotV3CollectionError("member roots must be unique")
        self.dataset_wref = dataset_wref
        self.robot_wref = robot_wref
        self.member_roots = list(member_roots)
        self.max_episodes = max_episodes

    @staticmethod
    def _info_path(source: SourceReceipt, member_root: str) -> Path:
        return source.destination / member_root / "meta" / "info.json"

    @classmethod
    def _info(cls, source: SourceReceipt, member_root: str) -> dict[str, Any]:
        path = cls._info_path(source, member_root)
        if not path.exists():
            raise LeRobotV3CollectionError(f"missing member metadata: {path}")
        info = json.loads(path.read_text())
        if info.get("codebase_version") != "v3.0":
            raise LeRobotV3CollectionError(
                "lerobot-v3-nested only accepts codebase_version v3.0; "
                f"found {info.get('codebase_version')!r} in {member_root}"
            )
        return info

    def _episode_locations(self, source: SourceReceipt) -> list[_EpisodeLocation]:
        locations: list[_EpisodeLocation] = []
        for member_root in self.member_roots:
            total = int(self._info(source, member_root)["total_episodes"])
            for local_index in range(total):
                locations.append(
                    _EpisodeLocation(
                        episode_id=str(len(locations)),
                        member_root=member_root,
                        local_episode_index=local_index,
                    )
                )
                if self.max_episodes is not None and len(locations) == self.max_episodes:
                    return locations
        return locations

    def inspect(self, source: SourceReceipt) -> dict[str, Any]:
        infos = [self._info(source, root) for root in self.member_roots]
        first = infos[0]
        required_features = (
            "observation.state",
            "action",
            "episode_index",
            "frame_index",
        )
        for root, info in zip(self.member_roots, infos, strict=True):
            features = info.get("features", {})
            for required in required_features:
                if required not in features:
                    raise LeRobotV3CollectionError(
                        f"missing required feature {required!r} in {root}"
                    )
            for field in ("fps", "robot_type"):
                if info.get(field) != first.get(field):
                    raise LeRobotV3CollectionError(
                        f"collection members disagree on {field}: {root}"
                    )
            for feature in ("observation.state", "action"):
                if features[feature] != first["features"][feature]:
                    raise LeRobotV3CollectionError(
                        f"collection members disagree on {feature}: {root}"
                    )
        locations = self._episode_locations(source)
        metadata_by_member = {
            root: self._episode_metadata(source, root).to_pydict()
            for root in self.member_roots
        }
        selected_frames = 0
        for location in locations:
            metadata = metadata_by_member[location.member_root]
            rows = {
                int(value): index
                for index, value in enumerate(metadata["episode_index"])
            }
            selected_frames += int(
                metadata["length"][rows[location.local_episode_index]]
            )
        return {
            "codebaseVersion": first["codebase_version"],
            "fps": first["fps"],
            "totalEpisodes": len(locations),
            "totalFrames": selected_frames,
            "state": first["features"]["observation.state"],
            "action": first["features"]["action"],
            "cameraKeys": sorted(
                key
                for key, value in first["features"].items()
                if value.get("dtype") == "video"
            ),
            "memberRoots": list(self.member_roots),
        }

    @staticmethod
    def materialization_patterns(
        upstream_files: Sequence[str],
        member_roots: Sequence[str],
        cameras: Sequence[str] = (),
    ) -> list[str]:
        listed = set(upstream_files)
        selected: list[str] = []
        for root in member_roots:
            required_exact = [
                f"{root}/meta/info.json",
                f"{root}/meta/tasks.parquet",
            ]
            missing = [path for path in required_exact if path not in listed]
            if missing:
                raise LeRobotV3CollectionError(
                    f"preflight did not find member metadata: {missing}"
                )
            member_episode_metadata = sorted(
                path
                for path in listed
                if path.startswith(f"{root}/meta/episodes/") and path.endswith(".parquet")
            )
            member_data = sorted(
                path
                for path in listed
                if path.startswith(f"{root}/data/") and path.endswith(".parquet")
            )
            if not member_episode_metadata or not member_data:
                raise LeRobotV3CollectionError(
                    f"preflight did not find episode metadata and data for {root}"
                )
            selected.extend([*required_exact, *member_episode_metadata, *member_data])
            for camera in cameras:
                camera_videos = sorted(
                    path
                    for path in listed
                    if path.startswith(f"{root}/videos/{camera}/")
                    and path.endswith(".mp4")
                )
                if not camera_videos:
                    raise LeRobotV3CollectionError(
                        f"preflight did not find camera {camera!r} for {root}"
                    )
                selected.extend(camera_videos)
        return sorted(set(selected))

    @staticmethod
    def _parquet_tables(paths: Sequence[Path], *, columns: Sequence[str]):
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise LeRobotV3CollectionError(
                "LeRobot v3 support requires: uv sync --extra lerobot"
            ) from error
        if not paths:
            raise LeRobotV3CollectionError("no Parquet files were materialized")
        return arrow.concat_tables(
            [parquet.read_table(path, columns=list(columns)) for path in paths]
        )

    @classmethod
    def _episode_metadata(cls, source: SourceReceipt, member_root: str):
        paths = sorted(
            (source.destination / member_root / "meta" / "episodes").rglob("*.parquet")
        )
        return cls._parquet_tables(
            paths,
            columns=[
                "episode_index",
                "tasks",
                "length",
                "data/chunk_index",
                "data/file_index",
                "dataset_from_index",
                "dataset_to_index",
            ],
        )

    def episodes(self, source: SourceReceipt) -> Iterator[CanonicalEpisode]:
        locations = self._episode_locations(source)
        selected_by_member: dict[str, list[_EpisodeLocation]] = {}
        for location in locations:
            selected_by_member.setdefault(location.member_root, []).append(location)
        for member_root, selected_locations in selected_by_member.items():
            info = self._info(source, member_root)
            paths = sorted((source.destination / member_root / "data").rglob("*.parquet"))
            table = self._parquet_tables(
                paths,
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
            episode_metadata = self._episode_metadata(source, member_root).to_pydict()
            metadata_rows = {
                int(value): index
                for index, value in enumerate(episode_metadata["episode_index"])
            }
            state_dimension = int(info["features"]["observation.state"]["shape"][0])
            action_dimension = int(info["features"]["action"]["shape"][0])
            for location in selected_locations:
                local_index = location.local_episode_index
                row_indices = np.flatnonzero(episode_values == local_index)
                if not len(row_indices):
                    raise LeRobotV3CollectionError(
                        f"episode {local_index} has no rows in {member_root}"
                    )
                episode = table.take(row_indices)
                frame_indices = (
                    episode.column("frame_index")
                    .combine_chunks()
                    .to_numpy(zero_copy_only=False)
                )
                if not np.array_equal(frame_indices, np.arange(episode.num_rows)):
                    raise LeRobotV3CollectionError(
                        f"non-contiguous frame indices in {member_root}:{local_index}"
                    )
                timestamps = (
                    episode.column("timestamp")
                    .combine_chunks()
                    .to_numpy(zero_copy_only=False)
                    .astype(np.float64, copy=False)
                )
                if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
                    raise LeRobotV3CollectionError(
                        f"timestamps are not strictly increasing in {member_root}:{local_index}"
                    )
                states = np.asarray(
                    episode.column("observation.state").combine_chunks().to_pylist(),
                    dtype=np.float32,
                )
                actions = np.asarray(
                    episode.column("action").combine_chunks().to_pylist(),
                    dtype=np.float32,
                )
                if states.shape != (episode.num_rows, state_dimension):
                    raise LeRobotV3CollectionError(
                        f"unexpected state shape in {member_root}:{local_index}: {states.shape}"
                    )
                if actions.shape != (episode.num_rows, action_dimension):
                    raise LeRobotV3CollectionError(
                        f"unexpected action shape in {member_root}:{local_index}: {actions.shape}"
                    )
                if not np.isfinite(states).all() or not np.isfinite(actions).all():
                    raise LeRobotV3CollectionError(
                        f"non-finite state/action value in {member_root}:{local_index}"
                    )
                metadata_index = metadata_rows.get(local_index)
                if metadata_index is None:
                    raise LeRobotV3CollectionError(
                        f"missing episode metadata for {member_root}:{local_index}"
                    )
                expected_length = int(episode_metadata["length"][metadata_index])
                if expected_length != episode.num_rows:
                    raise LeRobotV3CollectionError(
                        f"episode length mismatch in {member_root}:{local_index}"
                    )
                tasks = episode_metadata["tasks"][metadata_index] or []
                yield CanonicalEpisode(
                    episode_id=location.episode_id,
                    robot_wref=self.robot_wref,
                    dataset_wref=self.dataset_wref,
                    timestamps_seconds=timestamps,
                    observations={"state": states},
                    actions=actions,
                    task=str(tasks[0]) if tasks else None,
                    modality_mask={
                        "sensor/proprioception": True,
                        "state-space/joint-position": True,
                        "action-space/joint-position": True,
                        "sensor/rgb": bool(
                            any(
                                value.get("dtype") == "video"
                                for value in info["features"].values()
                            )
                        ),
                    },
                    source_member=member_root,
                )

    def video_segment(
        self,
        source: SourceReceipt,
        *,
        camera: str,
        episode_id: str,
    ) -> VideoSegment:
        locations = {item.episode_id: item for item in self._episode_locations(source)}
        if episode_id not in locations:
            raise LeRobotV3CollectionError(f"unknown selected episode: {episode_id}")
        location = locations[episode_id]
        info = self._info(source, location.member_root)
        feature = info.get("features", {}).get(camera)
        if feature is None or feature.get("dtype") != "video":
            raise LeRobotV3CollectionError(
                f"camera is not declared as a video feature: {camera}"
            )
        metadata = self._episode_metadata(source, location.member_root).to_pydict()
        rows = {
            int(value): index for index, value in enumerate(metadata["episode_index"])
        }
        row = rows[location.local_episode_index]
        chunk_column = f"videos/{camera}/chunk_index"
        file_column = f"videos/{camera}/file_index"
        start_column = f"videos/{camera}/from_timestamp"
        end_column = f"videos/{camera}/to_timestamp"
        if chunk_column not in metadata:
            metadata = self._parquet_tables(
                sorted(
                    (
                        source.destination
                        / location.member_root
                        / "meta"
                        / "episodes"
                    ).rglob("*.parquet")
                ),
                columns=[
                    "episode_index",
                    "length",
                    chunk_column,
                    file_column,
                    start_column,
                    end_column,
                ],
            ).to_pydict()
            rows = {
                int(value): index for index, value in enumerate(metadata["episode_index"])
            }
            row = rows[location.local_episode_index]
        relative = info["video_path"].format(
            video_key=camera,
            chunk_index=int(metadata[chunk_column][row]),
            file_index=int(metadata[file_column][row]),
        )
        path = source.destination / location.member_root / relative
        if not path.exists():
            raise LeRobotV3CollectionError(f"missing collection video: {path}")
        return VideoSegment(
            path=path,
            start_seconds=float(metadata[start_column][row]),
            end_seconds=float(metadata[end_column][row]),
            frame_count=int(metadata["length"][row]),
        )
