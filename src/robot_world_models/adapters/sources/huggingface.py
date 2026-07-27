from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import asdict
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from robot_world_models.adapters.base import SourceReceipt


class HuggingFaceSourceError(RuntimeError):
    """Raised when an immutable Hugging Face dataset cannot be materialized."""


def _payload_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    )


def _content_digest(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class HuggingFaceDatasetSource:
    """Fetch metadata and numeric episode data without pulling optional videos."""

    adapter_id = "huggingface-hub"

    @staticmethod
    def _token_mode() -> str | bool | None:
        # The token is only read by the Hub library and is never serialized into receipts.
        return None if os.environ.get("HF_TOKEN") else False

    def preflight(
        self,
        *,
        location: str,
        revision: str,
        required_files: Sequence[str] = (),
    ) -> dict[str, Any]:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise HuggingFaceSourceError(
                "Hugging Face support requires: uv sync --extra lerobot"
            ) from error

        try:
            info = HfApi(token=self._token_mode()).dataset_info(
                location,
                revision=revision,
                files_metadata=True,
            )
        except Exception as error:
            raise HuggingFaceSourceError(
                f"dataset preflight failed for {location}@{revision}: "
                f"{type(error).__name__}: {str(error).splitlines()[0]}"
            ) from error

        files = [
            {"path": sibling.rfilename, "size": sibling.size}
            for sibling in info.siblings
        ]
        unknown_size_files = [
            item["path"] for item in files if item["size"] is None
        ]
        listed = {item["path"] for item in files}
        missing = sorted(set(required_files) - listed)
        if missing:
            raise HuggingFaceSourceError(f"dataset is missing required files: {missing}")
        return {
            "repoId": location,
            "requestedRevision": revision,
            "resolvedRevision": info.sha,
            "private": info.private,
            "gated": info.gated,
            "fileCount": len(files),
            "repositoryBytes": (
                None
                if unknown_size_files
                else sum(int(item["size"]) for item in files)
            ),
            "unknownSizeFileCount": len(unknown_size_files),
            "files": files,
        }

    @staticmethod
    def estimate_download_bytes(
        files: Sequence[dict[str, Any]],
        include_patterns: Sequence[str],
    ) -> int:
        selected = [
            item
            for item in files
            if any(fnmatchcase(item["path"], pattern) for pattern in include_patterns)
        ]
        unknown = [item["path"] for item in selected if item["size"] is None]
        if unknown:
            raise HuggingFaceSourceError(
                f"cannot enforce a byte ceiling because sizes are missing: {unknown}"
            )
        return sum(int(item["size"]) for item in selected)

    def fetch(
        self,
        *,
        location: str,
        revision: str,
        destination: Path,
        include_patterns: Sequence[str] | None = None,
        max_download_bytes: int | None = None,
    ) -> SourceReceipt:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise HuggingFaceSourceError(
                "Hugging Face support requires: uv sync --extra lerobot"
            ) from error

        preflight = self.preflight(location=location, revision=revision)
        if preflight["resolvedRevision"] != revision:
            raise HuggingFaceSourceError(
                "upstream resolved a different commit than the WarmHub-pinned revision"
            )
        patterns = list(include_patterns or ["meta/*", "data/*"])
        expected_bytes = self.estimate_download_bytes(preflight["files"], patterns)
        if max_download_bytes is not None and expected_bytes > max_download_bytes:
            raise HuggingFaceSourceError(
                "approved download ceiling exceeded: "
                f"{expected_bytes} > {max_download_bytes} bytes"
            )
        resumed = destination.exists() and any(_payload_files(destination))
        destination.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=location,
                repo_type="dataset",
                revision=revision,
                local_dir=destination,
                allow_patterns=patterns,
                token=self._token_mode(),
            )
        except Exception as error:
            raise HuggingFaceSourceError(
                f"dataset materialization failed for {location}@{revision}: "
                f"{type(error).__name__}: {str(error).splitlines()[0]}"
            ) from error

        files = _payload_files(destination)
        return SourceReceipt(
            adapter_id=self.adapter_id,
            source_revision=revision,
            destination=destination,
            content_sha256=_content_digest(destination, files),
            bytes_written=sum(path.stat().st_size for path in files),
            files_written=len(files),
            resumed=resumed,
        )

    @staticmethod
    def receipt_dict(receipt: SourceReceipt) -> dict[str, Any]:
        payload = asdict(receipt)
        payload["destination"] = str(receipt.destination)
        return payload
