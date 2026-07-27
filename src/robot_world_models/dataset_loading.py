from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from robot_world_models.adapters.base import (
    CanonicalEpisode,
    SourceReceipt,
    VideoSegment,
)
from robot_world_models.adapters.formats.lerobot_v2 import LeRobotV2Adapter
from robot_world_models.adapters.formats.lerobot_v3_collection import (
    LeRobotV3CollectionAdapter,
)
from robot_world_models.contracts import DatasetManifest, RobotManifest, TrainingSubset


class DatasetLoadingError(RuntimeError):
    """Raised when a manifest requests an unsupported or inconsistent adapter."""


class PreparedFormatAdapter(Protocol):
    def inspect(self, source: SourceReceipt) -> dict[str, object]: ...

    def episodes(self, source: SourceReceipt): ...


@dataclass(frozen=True)
class PreparedDataset:
    adapter: PreparedFormatAdapter
    include_patterns: list[str]
    episode_count: int

    def video_segment(
        self,
        source: SourceReceipt,
        *,
        camera: str,
        episode: CanonicalEpisode,
    ) -> VideoSegment:
        if isinstance(self.adapter, LeRobotV2Adapter):
            path = self.adapter.video_path(
                source,
                camera=camera,
                episode_index=int(episode.episode_id),
            )
            return VideoSegment(
                path=path,
                start_seconds=0.0,
                end_seconds=float(len(episode.timestamps_seconds)),
                frame_count=len(episode.timestamps_seconds),
            )
        if isinstance(self.adapter, LeRobotV3CollectionAdapter):
            return self.adapter.video_segment(
                source,
                camera=camera,
                episode_id=episode.episode_id,
            )
        raise DatasetLoadingError("prepared adapter does not provide visual segments")


def prepare_dataset(
    *,
    dataset: DatasetManifest,
    robot: RobotManifest,
    subset: TrainingSubset,
    upstream_files: list[str],
    max_episodes: int,
    cameras: list[str] | None = None,
) -> PreparedDataset:
    cameras = cameras or []
    episode_count = min(max_episodes, dataset.episode_schema.total_episodes)
    if dataset.format.adapter == "lerobot-v2":
        if subset.member_roots:
            raise DatasetLoadingError("lerobot-v2 datasets cannot select collection members")
        episode_indices = list(range(episode_count))
        adapter = LeRobotV2Adapter(
            dataset_wref=dataset.warmhub.wref,
            robot_wref=robot.warmhub.wref,
            episode_indices=episode_indices,
        )
        return PreparedDataset(
            adapter=adapter,
            include_patterns=adapter.materialization_patterns(
                upstream_files,
                episode_indices,
                cameras=cameras,
            ),
            episode_count=episode_count,
        )
    if dataset.format.adapter == "lerobot-v3-nested":
        if dataset.collection is None:
            raise DatasetLoadingError(
                "lerobot-v3-nested requires a declared dataset collection"
            )
        member_roots = subset.member_roots or dataset.collection.members
        unknown = sorted(set(member_roots) - set(dataset.collection.members))
        if unknown:
            raise DatasetLoadingError(f"unknown collection members: {unknown}")
        adapter = LeRobotV3CollectionAdapter(
            dataset_wref=dataset.warmhub.wref,
            robot_wref=robot.warmhub.wref,
            member_roots=member_roots,
            max_episodes=episode_count,
        )
        return PreparedDataset(
            adapter=adapter,
            include_patterns=adapter.materialization_patterns(
                upstream_files,
                member_roots,
                cameras=cameras,
            ),
            episode_count=episode_count,
        )
    raise DatasetLoadingError(f"unsupported dataset format adapter: {dataset.format.adapter}")
