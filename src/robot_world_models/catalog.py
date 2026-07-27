from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from robot_world_models.contracts import (
    MANIFEST_ADAPTER,
    DatasetManifest,
    JointMappingManifest,
    Manifest,
    RecipeManifest,
    RobotManifest,
)

CATALOG_SCHEMA_VERSION = "robot-world-models.catalog.v1"
MANIFEST_DIRECTORIES = ("datasets", "robots", "mappings", "recipes")


class CatalogError(ValueError):
    """Raised when repository manifests fail their contracts."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def manifest_paths(root: Path | None = None) -> list[Path]:
    root = root or repository_root()
    paths: list[Path] = []
    for directory in MANIFEST_DIRECTORIES:
        paths.extend((root / "catalog" / directory).glob("*.yaml"))
    return sorted(paths)


def load_manifest(path: Path) -> Manifest:
    try:
        payload = yaml.safe_load(path.read_text())
        return MANIFEST_ADAPTER.validate_python(payload)
    except Exception as error:
        raise CatalogError(f"{path}: {error}") from error


def validate_repository(root: Path | None = None) -> list[tuple[Path, Manifest]]:
    root = root or repository_root()
    loaded = [(path, load_manifest(path)) for path in manifest_paths(root)]
    if not loaded:
        raise CatalogError("no manifests found")

    ids: dict[str, Path] = {}
    for path, manifest in loaded:
        if manifest.id in ids:
            raise CatalogError(
                f"duplicate manifest id {manifest.id}: {ids[manifest.id]} and {path}"
            )
        ids[manifest.id] = path

    dataset_ids = {
        manifest.id for _, manifest in loaded if isinstance(manifest, DatasetManifest)
    }
    robot_ids = {manifest.id for _, manifest in loaded if isinstance(manifest, RobotManifest)}
    mapping_ids = {
        manifest.id for _, manifest in loaded if isinstance(manifest, JointMappingManifest)
    }
    for path, manifest in loaded:
        if not isinstance(manifest, JointMappingManifest):
            continue
        if manifest.dataset not in dataset_ids:
            raise CatalogError(f"{path}: unknown dataset {manifest.dataset}")
        if manifest.robot not in robot_ids:
            raise CatalogError(f"{path}: unknown robot {manifest.robot}")
    for path, manifest in loaded:
        if not isinstance(manifest, RecipeManifest):
            continue
        missing_datasets = sorted(set(manifest.mixture.datasets) - dataset_ids)
        if missing_datasets:
            raise CatalogError(f"{path}: unknown datasets {missing_datasets}")
        if manifest.mixture.robot not in robot_ids:
            raise CatalogError(f"{path}: unknown robot {manifest.mixture.robot}")
        if manifest.joint_mapping is not None:
            if manifest.joint_mapping not in mapping_ids:
                raise CatalogError(f"{path}: unknown joint mapping {manifest.joint_mapping}")
            mapping = next(
                item
                for _, item in loaded
                if isinstance(item, JointMappingManifest)
                and item.id == manifest.joint_mapping
            )
            if mapping.dataset not in manifest.mixture.datasets:
                raise CatalogError(f"{path}: joint mapping dataset is not in the recipe mixture")
            if mapping.robot != manifest.mixture.robot:
                raise CatalogError(f"{path}: joint mapping robot does not match the recipe robot")
        if len(manifest.mixture.datasets) == 1:
            dataset = next(
                item
                for _, item in loaded
                if isinstance(item, DatasetManifest)
                and item.id == manifest.mixture.datasets[0]
            )
            selected_members = manifest.training.subset.member_roots
            if selected_members and dataset.collection is None:
                raise CatalogError(
                    f"{path}: member_roots require a collection dataset"
                )
            if dataset.collection is not None:
                unknown_members = sorted(
                    set(selected_members) - set(dataset.collection.members)
                )
                if unknown_members:
                    raise CatalogError(
                        f"{path}: unknown collection members {unknown_members}"
                    )

    return loaded


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_catalog(root: Path | None = None) -> str:
    root = root or repository_root()
    loaded = validate_repository(root)
    entries = [
        {
            "path": str(path.relative_to(root)),
            "manifest": manifest.model_dump(mode="json"),
        }
        for path, manifest in loaded
    ]
    digest_input = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    source_adapters = sorted(
        {
            manifest.source.adapter
            for _, manifest in loaded
            if isinstance(manifest, (DatasetManifest, RobotManifest))
        }
    )
    format_adapters = sorted(
        {
            manifest.format.adapter
            for _, manifest in loaded
            if isinstance(manifest, DatasetManifest)
        }
    )
    payload = {
        "schemaVersion": CATALOG_SCHEMA_VERSION,
        "contentSha256": hashlib.sha256(digest_input).hexdigest(),
        "capabilities": {
            "sourceAdapters": source_adapters,
            "formatAdapters": format_adapters,
        },
        "entries": entries,
    }
    return _canonical_json(payload)


def rendered_schemas() -> dict[str, str]:
    return {
        "dataset.schema.json": _canonical_json(DatasetManifest.model_json_schema()),
        "robot.schema.json": _canonical_json(RobotManifest.model_json_schema()),
        "joint-mapping.schema.json": _canonical_json(
            JointMappingManifest.model_json_schema()
        ),
        "recipe.schema.json": _canonical_json(RecipeManifest.model_json_schema()),
    }


def build_catalog(root: Path | None = None, *, check: bool = False) -> list[Path]:
    root = root or repository_root()
    expected: dict[Path, str] = {
        root / "catalog" / "catalog.json": render_catalog(root),
        **{
            root / "schemas" / filename: content
            for filename, content in rendered_schemas().items()
        },
    }
    stale = [
        path
        for path, content in expected.items()
        if not path.exists() or path.read_text() != content
    ]
    if check:
        if stale:
            relative = ", ".join(str(path.relative_to(root)) for path in stale)
            raise CatalogError(f"generated files are stale: {relative}")
        return []

    for path, content in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return sorted(expected)


def catalog_summary(root: Path | None = None) -> list[dict[str, str]]:
    loaded = validate_repository(root)
    return [
        {
            "kind": manifest.kind,
            "id": manifest.id,
            "displayName": manifest.display_name,
            "path": str(path),
        }
        for path, manifest in loaded
    ]


def manifest_by_id(identifier: str, root: Path | None = None) -> Manifest:
    matches = [
        manifest
        for _, manifest in validate_repository(root)
        if manifest.id == identifier
    ]
    if not matches:
        raise CatalogError(f"unknown manifest id: {identifier}")
    return matches[0]
