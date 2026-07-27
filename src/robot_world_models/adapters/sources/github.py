from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from robot_world_models.adapters.base import SourceReceipt


class GitHubSourceError(RuntimeError):
    """Raised when a pinned robot-description package cannot be materialized."""


def _run(*arguments: str, cwd: Path | None = None) -> None:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GitHubSourceError(f"{' '.join(arguments[:3])} failed: {detail}")


class GitHubSparseCheckoutSource:
    adapter_id = "github-sparse-checkout"

    def fetch_package(
        self,
        *,
        location: str,
        revision: str,
        package_root: str,
        destination: Path,
    ) -> SourceReceipt:
        resumed = (destination / ".git").is_dir()
        destination.mkdir(parents=True, exist_ok=True)
        if not resumed:
            if any(destination.iterdir()):
                raise GitHubSourceError(
                    f"refusing to initialize a non-empty destination: {destination}"
                )
            _run("git", "init", str(destination))
            _run(
                "git",
                "-C",
                str(destination),
                "remote",
                "add",
                "origin",
                f"https://github.com/{location}.git",
            )
            _run("git", "-C", str(destination), "sparse-checkout", "init", "--cone")
        _run(
            "git",
            "-C",
            str(destination),
            "sparse-checkout",
            "set",
            package_root,
        )
        _run(
            "git",
            "-C",
            str(destination),
            "fetch",
            "--depth",
            "1",
            "--filter=blob:none",
            "origin",
            revision,
        )
        _run("git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD")

        files = sorted(
            path
            for path in (destination / package_root).rglob("*")
            if path.is_file()
        )
        if not files:
            raise GitHubSourceError(f"package root has no files: {package_root}")
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.relative_to(destination).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
        return SourceReceipt(
            adapter_id=self.adapter_id,
            source_revision=revision,
            destination=destination,
            content_sha256=digest.hexdigest(),
            bytes_written=sum(path.stat().st_size for path in files),
            files_written=len(files),
            resumed=resumed,
        )
