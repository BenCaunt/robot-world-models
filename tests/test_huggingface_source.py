from __future__ import annotations

import pytest

from robot_world_models.adapters.sources.huggingface import (
    HuggingFaceDatasetSource,
    HuggingFaceSourceError,
)


def test_download_estimate_counts_only_approved_patterns() -> None:
    files = [
        {"path": "a/meta/info.json", "size": 10},
        {"path": "a/data/file.parquet", "size": 20},
        {"path": "a/videos/camera/file.mp4", "size": 1000},
        {"path": "b/videos/camera/file.mp4", "size": 2000},
    ]

    result = HuggingFaceDatasetSource.estimate_download_bytes(
        files,
        ["a/meta/info.json", "a/data/*"],
    )

    assert result == 30


def test_download_estimate_fails_closed_when_size_is_unknown() -> None:
    files = [{"path": "a/videos/camera/file.mp4", "size": None}]

    with pytest.raises(HuggingFaceSourceError, match="sizes are missing"):
        HuggingFaceDatasetSource.estimate_download_bytes(files, ["a/videos/*"])
