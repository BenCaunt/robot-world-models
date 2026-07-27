from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robot_world_models.adapters.base import SourceReceipt
from robot_world_models.adapters.formats.lerobot_v3_collection import (
    LeRobotV3CollectionAdapter,
    LeRobotV3CollectionError,
)

MEMBER = "recording-a"
CAMERA = "observation.images.desk_view"


def _fixture(tmp_path: Path, *, bad_frames: bool = False) -> SourceReceipt:
    root = tmp_path / MEMBER
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / CAMERA / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "so_follower",
        "total_episodes": 2,
        "total_frames": 5,
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [2],
                "names": ["shoulder.pos", "gripper.pos"],
            },
            "action": {
                "dtype": "float32",
                "shape": [2],
                "names": ["shoulder.pos", "gripper.pos"],
            },
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            CAMERA: {"dtype": "video", "shape": [8, 8, 3], "names": None},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.table({"task_index": [0], "task": ["move block"]}),
        root / "meta" / "tasks.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "tasks": [["move block"], ["move block"]],
                "length": [3, 2],
                "data/chunk_index": [0, 0],
                "data/file_index": [0, 0],
                "dataset_from_index": [0, 3],
                "dataset_to_index": [3, 5],
                f"videos/{CAMERA}/chunk_index": [0, 0],
                f"videos/{CAMERA}/file_index": [0, 0],
                f"videos/{CAMERA}/from_timestamp": [0.0, 0.1],
                f"videos/{CAMERA}/to_timestamp": [0.1, 0.1666666667],
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    frame_indices = [0, 1, 2, 1 if bad_frames else 0, 1]
    pq.write_table(
        pa.table(
            {
                "observation.state": [
                    [1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                    [6.0, 7.0],
                ],
                "action": [
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                ],
                "timestamp": np.array(
                    [0.0, 1 / 30, 2 / 30, 0.0, 1 / 30],
                    dtype=np.float32,
                ),
                "frame_index": frame_indices,
                "episode_index": [0, 0, 0, 1, 1],
            }
        ),
        root / "data" / "chunk-000" / "file-000.parquet",
    )
    (root / "videos" / CAMERA / "chunk-000" / "file-000.mp4").write_bytes(b"fixture")
    return SourceReceipt(
        adapter_id="test",
        source_revision="abc",
        destination=tmp_path,
        content_sha256="0" * 64,
        bytes_written=1,
        files_written=5,
        resumed=False,
    )


def test_materialization_selects_only_declared_members_and_camera() -> None:
    files = [
        f"{MEMBER}/meta/info.json",
        f"{MEMBER}/meta/tasks.parquet",
        f"{MEMBER}/meta/episodes/chunk-000/file-000.parquet",
        f"{MEMBER}/data/chunk-000/file-000.parquet",
        f"{MEMBER}/videos/{CAMERA}/chunk-000/file-000.mp4",
        "recording-b/data/chunk-000/file-000.parquet",
    ]

    patterns = LeRobotV3CollectionAdapter.materialization_patterns(
        files,
        [MEMBER],
        cameras=[CAMERA],
    )

    assert patterns == sorted(files[:-1])


def test_adapter_splits_shared_parquet_and_preserves_tasks(tmp_path: Path) -> None:
    receipt = _fixture(tmp_path)
    adapter = LeRobotV3CollectionAdapter(
        dataset_wref="Dataset/test",
        robot_wref="Robot/test",
        member_roots=[MEMBER],
    )

    inspection = adapter.inspect(receipt)
    episodes = list(adapter.episodes(receipt))

    assert inspection["codebaseVersion"] == "v3.0"
    assert inspection["totalEpisodes"] == 2
    assert [episode.episode_id for episode in episodes] == ["0", "1"]
    assert [len(episode.actions) for episode in episodes] == [3, 2]
    assert [episode.source_member for episode in episodes] == [MEMBER, MEMBER]
    assert episodes[1].task == "move block"
    assert np.asarray(episodes[1].observations["state"]).shape == (2, 2)


def test_adapter_reports_only_selected_episode_frames(tmp_path: Path) -> None:
    receipt = _fixture(tmp_path)
    adapter = LeRobotV3CollectionAdapter(
        dataset_wref="Dataset/test",
        robot_wref="Robot/test",
        member_roots=[MEMBER],
        max_episodes=1,
    )

    inspection = adapter.inspect(receipt)
    episodes = list(adapter.episodes(receipt))

    assert inspection["totalEpisodes"] == 1
    assert inspection["totalFrames"] == 3
    assert [episode.episode_id for episode in episodes] == ["0"]


def test_adapter_exposes_exact_shared_video_segment(tmp_path: Path) -> None:
    receipt = _fixture(tmp_path)
    adapter = LeRobotV3CollectionAdapter(
        dataset_wref="Dataset/test",
        robot_wref="Robot/test",
        member_roots=[MEMBER],
    )

    segment = adapter.video_segment(receipt, camera=CAMERA, episode_id="1")

    assert segment.path.name == "file-000.mp4"
    assert segment.start_seconds == pytest.approx(0.1)
    assert segment.end_seconds == pytest.approx(0.1666666667)
    assert segment.frame_count == 2


def test_adapter_rejects_noncontiguous_member_episode(tmp_path: Path) -> None:
    receipt = _fixture(tmp_path, bad_frames=True)
    adapter = LeRobotV3CollectionAdapter(
        dataset_wref="Dataset/test",
        robot_wref="Robot/test",
        member_roots=[MEMBER],
    )

    with pytest.raises(LeRobotV3CollectionError, match="non-contiguous frame indices"):
        list(adapter.episodes(receipt))
