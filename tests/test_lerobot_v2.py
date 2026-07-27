from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robot_world_models.adapters.base import SourceReceipt
from robot_world_models.adapters.formats.lerobot_v2 import LeRobotV2Adapter


def test_materialization_patterns_select_only_requested_episode_payloads() -> None:
    files = [
        "meta/info.json",
        "data/chunk-000/episode_000000.parquet",
        "data/chunk-000/episode_000001.parquet",
        "videos/chunk-000/camera/episode_000000.mp4",
    ]

    patterns = LeRobotV2Adapter.materialization_patterns(files, [1])

    assert patterns == ["meta/*", "data/chunk-000/episode_000001.parquet"]


def test_materialization_patterns_include_only_selected_camera_payloads() -> None:
    files = [
        "meta/info.json",
        "data/chunk-000/episode_000000.parquet",
        "data/chunk-000/episode_000001.parquet",
        "videos/chunk-000/observation.images.laptop/episode_000000.mp4",
        "videos/chunk-000/observation.images.laptop/episode_000001.mp4",
        "videos/chunk-000/observation.images.phone1/episode_000000.mp4",
        "videos/chunk-000/observation.images.phone1/episode_000001.mp4",
    ]

    patterns = LeRobotV2Adapter.materialization_patterns(
        files,
        [1],
        cameras=["observation.images.laptop"],
    )

    assert patterns == [
        "meta/*",
        "data/chunk-000/episode_000001.parquet",
        "videos/chunk-000/observation.images.laptop/episode_000001.mp4",
    ]


def test_adapter_preserves_episode_boundary(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": 3,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "observation.state": {"shape": [2], "names": ["a", "b"]},
            "action": {"shape": [2], "names": ["a", "b"]},
            "episode_index": {"shape": [1]},
            "frame_index": {"shape": [1]},
        },
    }
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))
    table = pa.table(
        {
            "observation.state": [[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]],
            "action": [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            "timestamp": np.array([0.0, 1 / 30, 2 / 30], dtype=np.float32),
            "frame_index": [0, 1, 2],
            "episode_index": [0, 0, 0],
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")
    receipt = SourceReceipt(
        adapter_id="test",
        source_revision="abc",
        destination=tmp_path,
        content_sha256="0" * 64,
        bytes_written=1,
        files_written=2,
        resumed=False,
    )

    episode = next(
        LeRobotV2Adapter(
            dataset_wref="Dataset/test",
            robot_wref="Robot/test",
        ).episodes(receipt)
    )

    assert episode.episode_id == "0"
    assert np.asarray(episode.observations["state"]).shape == (3, 2)
    assert np.asarray(episode.actions).shape == (3, 2)
