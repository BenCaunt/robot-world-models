from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SourceReceipt:
    adapter_id: str
    source_revision: str
    destination: Path
    content_sha256: str
    bytes_written: int
    files_written: int
    resumed: bool


@dataclass(frozen=True)
class CanonicalEpisode:
    episode_id: str
    robot_wref: str
    dataset_wref: str
    timestamps_seconds: Sequence[float]
    observations: Mapping[str, object]
    actions: Sequence[Sequence[float]]
    task: str | None
    modality_mask: Mapping[str, bool]


@dataclass(frozen=True)
class VideoSegment:
    path: Path
    start_seconds: float
    end_seconds: float
    frame_count: int


class SourceAdapter(Protocol):
    """Materialize immutable upstream bytes from a WarmHub-resolved location."""

    adapter_id: str

    def fetch(
        self,
        *,
        location: str,
        revision: str,
        destination: Path,
    ) -> SourceReceipt: ...


class FormatAdapter(Protocol):
    """Convert one source layout into canonical episodes."""

    adapter_id: str

    def inspect(self, source: SourceReceipt) -> Mapping[str, object]: ...

    def episodes(self, source: SourceReceipt) -> Iterator[CanonicalEpisode]: ...
