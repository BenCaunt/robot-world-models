from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot_world_models.contracts import DatasetManifest, RobotManifest

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

    def view(self, wref: str, repo: str) -> dict[str, Any]:
        return self._json("thing", "view", wref, "--repo", repo)

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

    def resolve_dataset(self, manifest: DatasetManifest) -> dict[str, Any]:
        dataset = self.view(manifest.warmhub.wref, manifest.warmhub.repo)
        profile = self.view(manifest.profile_wref, manifest.warmhub.repo)
        recorded_with = self.view(
            manifest.robot_evidence.recorded_with_wref,
            manifest.warmhub.repo,
        )
        data = dataset.get("data", {})
        profile_data = profile.get("data", {})
        repo_id = f"{data.get('org')}/{data.get('name')}"
        revision = profile_data.get("commitSha")
        if "/" not in repo_id or "None" in repo_id:
            raise WarmHubError("Dataset does not contain a usable upstream org/name")
        if revision != manifest.upstream_revision:
            raise WarmHubError(
                "WarmHub DatasetProfile revision does not match the reviewed manifest: "
                f"{revision!r} != {manifest.upstream_revision!r}"
            )
        if normalize_wref(profile_data.get("datasetWref", "")) != manifest.warmhub.wref:
            raise WarmHubError("DatasetProfile is not about the selected Dataset")
        recorded_data = recorded_with.get("data", {})
        if normalize_wref(recorded_data.get("datasetWref", "")) != manifest.warmhub.wref:
            raise WarmHubError("RecordedWith is not about the selected Dataset")
        return {
            "repoId": repo_id,
            "revision": revision,
            "dataset": dataset,
            "profile": profile,
            "recordedWith": recorded_with,
        }

    def resolve_robot(self, manifest: RobotManifest) -> dict[str, Any]:
        robot = self.view(manifest.warmhub.wref, manifest.warmhub.repo)
        description = self.view(manifest.description.wref, manifest.warmhub.repo)
        model_profile = self.view(
            manifest.description.model_profile_wref,
            manifest.warmhub.repo,
        )
        description_data = description.get("data", {})
        if description_data.get("pinnedCommit") != manifest.description.pinned_commit:
            raise WarmHubError(
                "WarmHub Description revision does not match the reviewed manifest"
            )
        if description_data.get("entrypointPath") != manifest.description.entrypoint:
            raise WarmHubError(
                "WarmHub Description entrypoint does not match the reviewed manifest"
            )
        return {
            "robot": robot,
            "description": description,
            "modelProfile": model_profile,
        }
