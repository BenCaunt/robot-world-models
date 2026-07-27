from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

DEFAULT_MODELS_REPO = "bencaunt/robot-models"
DEFAULT_DATASETS_REPO = "bencaunt/robot-datasets"


class WarmHubError(RuntimeError):
    """Raised when the read-only WarmHub CLI path fails."""


def normalize_wref(wref: str) -> str:
    head, separator, version = wref.rpartition("@v")
    return head if separator and version.isdigit() else wref


@dataclass(frozen=True)
class WarmHubCLI:
    models_repo: str = DEFAULT_MODELS_REPO
    datasets_repo: str = DEFAULT_DATASETS_REPO

    @classmethod
    def from_environment(cls) -> WarmHubCLI:
        return cls(
            models_repo=os.environ.get("WARMHUB_MODELS_REPO", DEFAULT_MODELS_REPO),
            datasets_repo=os.environ.get("WARMHUB_DATASETS_REPO", DEFAULT_DATASETS_REPO),
        )

    def available(self) -> bool:
        return shutil.which("wh") is not None

    def _json(self, *arguments: str) -> dict[str, Any]:
        if not self.available():
            raise WarmHubError("wh CLI is required; install or update the WarmHub CLI first")
        completed = subprocess.run(
            ["wh", *arguments, "--json"],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise WarmHubError(f"wh command failed: {detail}")
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise WarmHubError("wh returned non-JSON output") from error

    def search(self, query: str, repo: str, *, limit: int = 30) -> dict[str, Any]:
        return self._json(
            "thing",
            "search",
            query,
            "--repo",
            repo,
            "--mode",
            "hybrid",
            "--limit",
            str(limit),
        )

    def discover(self, query: str) -> dict[str, Any]:
        models = self.search(query, self.models_repo, limit=30)
        datasets = self.search(query, self.datasets_repo, limit=50)
        return {
            "query": query,
            "sources": {
                "models": self.models_repo,
                "datasets": self.datasets_repo,
            },
            "models": models.get("items", []),
            "datasets": datasets.get("items", []),
        }

