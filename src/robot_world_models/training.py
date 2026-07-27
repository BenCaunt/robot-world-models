from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_world_models.adapters.base import CanonicalEpisode
from robot_world_models.adapters.sources.github import GitHubSparseCheckoutSource
from robot_world_models.adapters.sources.huggingface import HuggingFaceDatasetSource
from robot_world_models.catalog import manifest_by_id, repository_root
from robot_world_models.contracts import (
    DatasetManifest,
    JointMappingManifest,
    RecipeManifest,
    RobotManifest,
)
from robot_world_models.dataset_loading import prepare_dataset
from robot_world_models.devices import select_device
from robot_world_models.eval.rerun_eval import write_state_evaluation
from robot_world_models.warmhub import WarmHubCLI


class TrainingSpikeError(RuntimeError):
    """Raised when a recipe cannot complete its bounded local proof."""


@dataclass(frozen=True)
class Normalization:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def to_json(self) -> dict[str, list[float]]:
        return {
            "stateMean": self.state_mean.tolist(),
            "stateStd": self.state_std.tolist(),
            "actionMean": self.action_mean.tolist(),
            "actionStd": self.action_std.tolist(),
        }


@dataclass(frozen=True)
class Transitions:
    states: np.ndarray
    actions: np.ndarray
    targets: np.ndarray
    episode_ids: np.ndarray


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_receipt() -> dict[str, Any]:
    root = repository_root()

    def command(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unknown"

    return {
        "commit": command("rev-parse", "HEAD"),
        "dirty": bool(command("status", "--porcelain")),
    }


def split_episode_ids(
    episode_ids: list[str],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, list[str]]:
    if len(episode_ids) < 3:
        raise TrainingSpikeError("an episode split requires at least three episodes")
    shuffled = list(episode_ids)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(len(shuffled) * validation_fraction))
    test_fraction = 1.0 - train_fraction - validation_fraction
    test_count = max(1, int(len(shuffled) * test_fraction))
    train_count = len(shuffled) - validation_count - test_count
    if train_count < 1:
        raise TrainingSpikeError("split fractions leave no training episodes")
    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count : train_count + validation_count],
        "test": shuffled[train_count + validation_count :],
    }


def split_source_held_out_episode_ids(
    episodes: list[CanonicalEpisode],
    *,
    test_source_members: list[str],
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, list[str]]:
    if not test_source_members:
        raise TrainingSpikeError(
            "a source split requires explicit training.subset.test_member_roots"
        )
    missing_source = [
        episode.episode_id for episode in episodes if episode.source_member is None
    ]
    if missing_source:
        raise TrainingSpikeError(
            "a source split requires source_member on every episode; missing for "
            f"{missing_source[:5]}"
        )
    by_source: dict[str, list[str]] = {}
    for episode in episodes:
        assert episode.source_member is not None
        by_source.setdefault(episode.source_member, []).append(episode.episode_id)
    unknown = sorted(set(test_source_members) - set(by_source))
    if unknown:
        raise TrainingSpikeError(f"test source members were not materialized: {unknown}")

    test_sources = set(test_source_members)
    development = {
        source: identifiers
        for source, identifiers in by_source.items()
        if source not in test_sources
    }
    if not development:
        raise TrainingSpikeError("source holdout leaves no development members")
    development_fraction = train_fraction + validation_fraction
    if development_fraction <= 0:
        raise TrainingSpikeError("source holdout leaves no train/validation fraction")
    relative_validation_fraction = validation_fraction / development_fraction

    train: list[str] = []
    validation: list[str] = []
    for source in sorted(development):
        shuffled = list(development[source])
        random.Random(f"{seed}:{source}").shuffle(shuffled)
        if len(shuffled) < 2:
            raise TrainingSpikeError(
                f"development source {source} needs at least two episodes"
            )
        validation_count = max(
            1,
            int(round(len(shuffled) * relative_validation_fraction)),
        )
        validation_count = min(validation_count, len(shuffled) - 1)
        validation.extend(shuffled[:validation_count])
        train.extend(shuffled[validation_count:])

    test = [
        episode.episode_id
        for episode in episodes
        if episode.source_member in test_sources
    ]
    if not test:
        raise TrainingSpikeError("source holdout leaves no test episodes")
    declared_test_fraction = 1.0 - train_fraction - validation_fraction
    actual_test_fraction = len(test) / len(episodes)
    episode_rounding_tolerance = 1 / len(episodes)
    if abs(actual_test_fraction - declared_test_fraction) > (
        episode_rounding_tolerance + 1e-9
    ):
        raise TrainingSpikeError(
            "test_member_roots do not match the declared test fraction: "
            f"{actual_test_fraction:.6f} != {declared_test_fraction:.6f}"
        )
    return {"train": train, "validation": validation, "test": test}


def split_canonical_episodes(
    episodes: list[CanonicalEpisode],
    *,
    unit: str,
    test_source_members: list[str],
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, list[str]]:
    if unit == "episode":
        if test_source_members:
            raise TrainingSpikeError(
                "test_member_roots may only be used with a source split"
            )
        return split_episode_ids(
            [episode.episode_id for episode in episodes],
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
    if unit == "source":
        return split_source_held_out_episode_ids(
            episodes,
            test_source_members=test_source_members,
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
    raise TrainingSpikeError(f"split unit is not implemented: {unit}")


def transitions_from_episodes(episodes: list[CanonicalEpisode]) -> Transitions:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    for episode in episodes:
        episode_states = np.asarray(episode.observations["state"], dtype=np.float32)
        episode_actions = np.asarray(episode.actions, dtype=np.float32)
        if len(episode_states) != len(episode_actions):
            raise TrainingSpikeError(
                f"state/action length mismatch in episode {episode.episode_id}"
            )
        if len(episode_states) < 2:
            continue
        states.append(episode_states[:-1])
        actions.append(episode_actions[:-1])
        targets.append(episode_states[1:])
        episode_ids.append(np.full(len(episode_states) - 1, episode.episode_id, dtype=object))
    if not states:
        raise TrainingSpikeError("no within-episode transitions were extracted")
    return Transitions(
        states=np.concatenate(states),
        actions=np.concatenate(actions),
        targets=np.concatenate(targets),
        episode_ids=np.concatenate(episode_ids),
    )


def fit_normalization(transitions: Transitions) -> Normalization:
    state_std = transitions.states.std(axis=0)
    action_std = transitions.actions.std(axis=0)
    return Normalization(
        state_mean=transitions.states.mean(axis=0),
        state_std=np.maximum(state_std, 1e-6),
        action_mean=transitions.actions.mean(axis=0),
        action_std=np.maximum(action_std, 1e-6),
    )


def _normalized_arrays(
    transitions: Transitions,
    normalization: Normalization,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        (transitions.states - normalization.state_mean) / normalization.state_std,
        (transitions.actions - normalization.action_mean) / normalization.action_std,
        (transitions.targets - normalization.state_mean) / normalization.state_std,
    )


def _seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_model(recipe: RecipeManifest):
    from robot_world_models.models.state_dynamics import StateDynamicsMLP

    return StateDynamicsMLP(
        state_dimension=recipe.model.state_dimension,
        action_dimension=recipe.model.action_dimension,
        hidden_dimension=recipe.model.hidden_dimension,
        hidden_layers=recipe.model.hidden_layers,
    )


def _smoke_test(
    *,
    recipe: RecipeManifest,
    transitions: Transitions,
    normalization: Normalization,
    device: str,
    steps: int,
) -> dict[str, float | int | bool]:
    import torch
    from torch.nn import functional as functional

    states, actions, targets = _normalized_arrays(transitions, normalization)
    batch_size = min(recipe.training.batch_size, len(states))
    batch_states = torch.from_numpy(states[:batch_size]).to(device)
    batch_actions = torch.from_numpy(actions[:batch_size]).to(device)
    batch_targets = torch.from_numpy(targets[:batch_size]).to(device)
    _seed_all(recipe.training.seed)
    model = _make_model(recipe).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.training.learning_rate)
    with torch.no_grad():
        initial_loss = float(
            functional.mse_loss(model(batch_states, batch_actions), batch_targets).item()
        )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = functional.mse_loss(model(batch_states, batch_actions), batch_targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(
            functional.mse_loss(model(batch_states, batch_actions), batch_targets).item()
        )
    passed = final_loss < initial_loss
    if not passed:
        raise TrainingSpikeError(
            f"overfit smoke test did not reduce loss: {initial_loss} -> {final_loss}"
        )
    return {
        "steps": steps,
        "batchSize": batch_size,
        "initialNormalizedMse": initial_loss,
        "finalNormalizedMse": final_loss,
        "passed": passed,
    }


def _train(
    *,
    recipe: RecipeManifest,
    transitions: Transitions,
    normalization: Normalization,
    device: str,
    steps: int,
    checkpoint_dir: Path,
):
    import torch
    from torch.nn import functional as functional

    states, actions, targets = _normalized_arrays(transitions, normalization)
    _seed_all(recipe.training.seed)
    model = _make_model(recipe).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.training.learning_rate)
    generator = np.random.default_rng(recipe.training.seed)
    history: list[dict[str, float | int]] = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model.train()
    for step in range(1, steps + 1):
        indices = generator.integers(0, len(states), size=recipe.training.batch_size)
        batch_states = torch.from_numpy(states[indices]).to(device)
        batch_actions = torch.from_numpy(actions[indices]).to(device)
        batch_targets = torch.from_numpy(targets[indices]).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = functional.mse_loss(model(batch_states, batch_actions), batch_targets)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            history.append({"step": step, "normalizedMse": float(loss.item())})
        if step % recipe.training.checkpoint_every_steps == 0 and step != steps:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint_dir / f"step-{step:06d}.pt",
            )
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    final_path = checkpoint_dir / "final.pt"
    torch.save(
        {
            "step": steps,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "normalization": normalization.to_json(),
            "recipe": recipe.model_dump(mode="json"),
        },
        final_path,
    )
    return model, {
        "steps": steps,
        "seconds": time.monotonic() - started,
        "history": history,
        "checkpoint": str(final_path),
        "checkpointSha256": _sha256(final_path),
    }


def _predict(
    *,
    model,
    states: np.ndarray,
    actions: np.ndarray,
    normalization: Normalization,
    device: str,
) -> np.ndarray:
    import torch

    normalized_states = (states - normalization.state_mean) / normalization.state_std
    normalized_actions = (actions - normalization.action_mean) / normalization.action_std
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(states), 4096):
            stop = start + 4096
            batch_states = torch.from_numpy(normalized_states[start:stop]).to(device)
            batch_actions = torch.from_numpy(normalized_actions[start:stop]).to(device)
            batch = model(batch_states, batch_actions).cpu().numpy()
            predictions.append(batch)
    normalized_prediction = np.concatenate(predictions)
    return (
        normalized_prediction * normalization.state_std + normalization.state_mean
    ).astype(np.float32)


def _rollout_metrics(
    *,
    model,
    episodes: list[CanonicalEpisode],
    normalization: Normalization,
    device: str,
    horizons: list[int],
) -> dict[str, float]:
    squared_errors: dict[int, list[np.ndarray]] = {horizon: [] for horizon in horizons}
    maximum_horizon = max(horizons)
    for episode in episodes:
        states = np.asarray(episode.observations["state"], dtype=np.float32)
        actions = np.asarray(episode.actions, dtype=np.float32)
        count = len(states) - maximum_horizon
        if count <= 0:
            continue
        starts = np.arange(count)
        predicted = states[starts].copy()
        for step in range(1, maximum_horizon + 1):
            predicted = _predict(
                model=model,
                states=predicted,
                actions=actions[starts + step - 1],
                normalization=normalization,
                device=device,
            )
            if step in squared_errors:
                target = states[starts + step]
                squared_errors[step].append((predicted - target) ** 2)
    return {
        f"rollout_mse_h{horizon}": float(np.concatenate(errors).mean())
        for horizon, errors in squared_errors.items()
        if errors
    }


def _evaluate(
    *,
    model,
    transitions: Transitions,
    episodes: list[CanonicalEpisode],
    normalization: Normalization,
    joint_names: list[str],
    horizons: list[int],
    device: str,
) -> tuple[dict[str, float], np.ndarray]:
    started = time.monotonic()
    predictions = _predict(
        model=model,
        states=transitions.states,
        actions=transitions.actions,
        normalization=normalization,
        device=device,
    )
    errors = predictions - transitions.targets
    persistence_errors = transitions.states - transitions.targets
    metrics: dict[str, float] = {
        "one_step_mse": float(np.mean(errors**2)),
        "one_step_mae": float(np.mean(np.abs(errors))),
        "persistence_baseline_mse": float(np.mean(persistence_errors**2)),
        "inference_ms_per_transition": (
            (time.monotonic() - started) * 1000 / len(transitions.states)
        ),
    }
    metrics["improvement_over_persistence_fraction"] = float(
        1.0 - metrics["one_step_mse"] / metrics["persistence_baseline_mse"]
    )
    per_joint = np.mean(np.abs(errors), axis=0)
    metrics.update(
        {
            f"per_joint_mae/{joint_name}": float(per_joint[index])
            for index, joint_name in enumerate(joint_names)
        }
    )
    metrics.update(
        _rollout_metrics(
            model=model,
            episodes=episodes,
            normalization=normalization,
            device=device,
            horizons=horizons,
        )
    )
    return metrics, predictions


def _validate_dataset_contract(
    *,
    manifest: DatasetManifest,
    inspection: dict[str, Any],
) -> None:
    expected = manifest.episode_schema
    if inspection["state"]["shape"] != [expected.state.dimension]:
        raise TrainingSpikeError("materialized state dimension differs from the manifest")
    if inspection["action"]["shape"] != [expected.action.dimension]:
        raise TrainingSpikeError("materialized action dimension differs from the manifest")
    if inspection["state"]["names"] != expected.state.names:
        raise TrainingSpikeError("materialized state names differ from the manifest")
    if inspection["action"]["names"] != expected.action.names:
        raise TrainingSpikeError("materialized action names differ from the manifest")


def _run_recipe(
    *,
    recipe_id: str,
    run_dir: Path,
    max_steps: int | None,
    smoke_test_steps: int | None,
    max_episodes: int | None,
) -> dict[str, Any]:
    recipe = manifest_by_id(recipe_id)
    if not isinstance(recipe, RecipeManifest):
        raise TrainingSpikeError(f"{recipe_id} is not a recipe")
    if recipe.model.vision is not None:
        from robot_world_models.visual_training import run_visual_recipe

        return run_visual_recipe(
            recipe=recipe,
            run_dir=run_dir,
            max_steps=max_steps,
            smoke_test_steps=smoke_test_steps,
            max_episodes=max_episodes,
        )
    if recipe.mixture.type != "homogeneous" or len(recipe.mixture.datasets) != 1:
        raise TrainingSpikeError("the first trainer supports one homogeneous dataset")
    dataset = manifest_by_id(recipe.mixture.datasets[0])
    robot = manifest_by_id(recipe.mixture.robot)
    if not isinstance(dataset, DatasetManifest) or not isinstance(robot, RobotManifest):
        raise TrainingSpikeError("recipe references invalid dataset or robot manifests")
    mapping = manifest_by_id(recipe.joint_mapping) if recipe.joint_mapping else None
    if mapping is not None and not isinstance(mapping, JointMappingManifest):
        raise TrainingSpikeError("recipe joint_mapping is not a mapping manifest")

    effective_steps = max_steps or recipe.training.max_steps
    effective_smoke_steps = smoke_test_steps or recipe.training.smoke_test_steps
    effective_episodes = max_episodes or recipe.training.subset.max_episodes
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "config.json",
        {
            "recipe": recipe.model_dump(mode="json"),
            "overrides": {
                "maxSteps": max_steps,
                "smokeTestSteps": smoke_test_steps,
                "maxEpisodes": max_episodes,
            },
            "effective": {
                "maxSteps": effective_steps,
                "smokeTestSteps": effective_smoke_steps,
                "maxEpisodes": effective_episodes,
            },
        },
    )

    warmhub = WarmHubCLI.from_environment()
    dataset_resolution = warmhub.resolve_dataset(dataset)
    robot_resolution = warmhub.resolve_robot(robot)
    discovery = {
        "sources": {
            "datasets": dataset.warmhub.repo,
            "models": robot.warmhub.repo,
        },
        "dataset": dataset_resolution,
        "robot": robot_resolution,
    }
    _write_json(run_dir / "discovery.json", discovery)

    dataset_source = HuggingFaceDatasetSource()
    preflight = dataset_source.preflight(
        location=dataset_resolution["repoId"],
        revision=dataset_resolution["revision"],
    )
    _write_json(run_dir / "source-preflight.json", preflight)
    prepared = prepare_dataset(
        dataset=dataset,
        robot=robot,
        subset=recipe.training.subset,
        upstream_files=[item["path"] for item in preflight["files"]],
        max_episodes=effective_episodes,
    )
    source_receipt = dataset_source.fetch(
        location=dataset_resolution["repoId"],
        revision=dataset_resolution["revision"],
        destination=run_dir / "data" / dataset.id,
        include_patterns=prepared.include_patterns,
        max_download_bytes=recipe.training.subset.max_download_bytes,
    )
    _write_json(
        run_dir / "data-receipt.json",
        HuggingFaceDatasetSource.receipt_dict(source_receipt),
    )

    inspection = prepared.adapter.inspect(source_receipt)
    _validate_dataset_contract(manifest=dataset, inspection=inspection)
    episodes = list(prepared.adapter.episodes(source_receipt))
    by_id = {episode.episode_id: episode for episode in episodes}
    split = split_canonical_episodes(
        episodes,
        unit=recipe.mixture.split.unit,
        test_source_members=recipe.training.subset.test_member_roots,
        seed=recipe.training.seed,
        train_fraction=recipe.mixture.split.train_fraction,
        validation_fraction=recipe.mixture.split.validation_fraction,
    )
    _write_json(
        run_dir / "materialization.json",
        {
            "inspection": inspection,
            "episodesLoaded": len(episodes),
            "split": split,
            "transitionConvention": "observation.state[t] + action[t] -> observation.state[t+1]",
            "episodeBoundaryTransitions": False,
            "videosDownloaded": False,
        },
    )
    train_episodes = [by_id[identifier] for identifier in split["train"]]
    validation_episodes = [by_id[identifier] for identifier in split["validation"]]
    test_episodes = [by_id[identifier] for identifier in split["test"]]
    train_transitions = transitions_from_episodes(train_episodes)
    validation_transitions = transitions_from_episodes(validation_episodes)
    test_transitions = transitions_from_episodes(test_episodes)
    normalization = fit_normalization(train_transitions)
    _write_json(run_dir / "normalization.json", normalization.to_json())

    device = select_device(recipe.training.local_devices)
    smoke = _smoke_test(
        recipe=recipe,
        transitions=train_transitions,
        normalization=normalization,
        device=device,
        steps=effective_smoke_steps,
    )
    _write_json(run_dir / "smoke-test.json", smoke)
    model, training = _train(
        recipe=recipe,
        transitions=train_transitions,
        normalization=normalization,
        device=device,
        steps=effective_steps,
        checkpoint_dir=run_dir / "checkpoints",
    )
    _write_json(run_dir / "training.json", training)
    validation_metrics, _ = _evaluate(
        model=model,
        transitions=validation_transitions,
        episodes=validation_episodes,
        normalization=normalization,
        joint_names=dataset.episode_schema.state.names,
        horizons=recipe.evaluation.rollout_horizons,
        device=device,
    )
    test_metrics, test_predictions = _evaluate(
        model=model,
        transitions=test_transitions,
        episodes=test_episodes,
        normalization=normalization,
        joint_names=dataset.episode_schema.state.names,
        horizons=recipe.evaluation.rollout_horizons,
        device=device,
    )

    description = robot_resolution["description"]["data"]
    robot_source = GitHubSparseCheckoutSource()
    robot_receipt = robot_source.fetch_package(
        location=f"{description['org']}/{description['repo']}",
        revision=description["pinnedCommit"],
        package_root=description["packageRootPath"],
        destination=run_dir / "robot",
    )
    urdf_path = run_dir / "robot" / description["entrypointPath"]
    if not urdf_path.exists():
        raise TrainingSpikeError(f"WarmHub-resolved URDF is missing: {urdf_path}")
    _write_json(
        run_dir / "robot-receipt.json",
        {
            **asdict(robot_receipt),
            "destination": str(robot_receipt.destination),
            "urdf": str(urdf_path),
        },
    )

    lock_path = repository_root() / "uv.lock"
    provenance = {
        "recipeId": recipe.id,
        "datasetWref": dataset_resolution["dataset"]["pinnedWref"],
        "datasetProfileWref": dataset_resolution["profile"]["pinnedWref"],
        "recordedWithWref": dataset_resolution["recordedWith"]["pinnedWref"],
        "robotWref": robot_resolution["robot"]["pinnedWref"],
        "descriptionWref": robot_resolution["description"]["pinnedWref"],
        "upstreamDatasetRevision": source_receipt.source_revision,
        "datasetContentSha256": source_receipt.content_sha256,
        "robotPackageSha256": robot_receipt.content_sha256,
        "uvLockSha256": _sha256(lock_path),
        "git": _git_receipt(),
        "device": device,
        "seed": recipe.training.seed,
        "split": split,
        "mappingStatus": mapping.mapping.status if mapping else "absent",
        "mappingCoverage": mapping.mapping.coverage if mapping else "absent",
        "datasetStateUnits": dataset.episode_schema.state.units,
        "datasetActionUnits": dataset.episode_schema.action.units,
        "skippedMetrics": {
            "gripper_joint_limit_violation_rate": (
                "the LeRobot 0-100 gripper command is not yet mapped to the URDF jaw angle"
            )
        },
    }
    display_count = min(300, len(test_predictions))
    animation_enabled = bool(
        mapping is not None
        and mapping.mapping.status == "validated"
        and mapping.mapping.animate_in_rerun
    )
    rerun_path, animation = write_state_evaluation(
        output_path=run_dir / "evaluation.rrd",
        run_id=run_dir.name,
        joint_names=dataset.episode_schema.state.names,
        actual_states=test_transitions.targets[:display_count].tolist(),
        predicted_states=test_predictions[:display_count].tolist(),
        metrics=test_metrics,
        provenance=provenance,
        urdf_path=urdf_path,
        joint_mapping=mapping.mapping.entries
        if animation_enabled and mapping is not None
        else None,
        unmapped_features=(
            mapping.mapping.unmapped_features if animation_enabled and mapping is not None else ()
        ),
        out_of_range_policy=(
            mapping.mapping.out_of_range_policy
            if animation_enabled and mapping is not None
            else "reject"
        ),
    )
    evaluation = {
        "validation": validation_metrics,
        "test": test_metrics,
        "skipped": provenance["skippedMetrics"],
        "metricUnits": {
            "mae": dataset.episode_schema.state.units,
            "mse": f"squared {dataset.episode_schema.state.units}",
            "inference": "milliseconds per transition",
        },
        "rerun": str(rerun_path),
        "rerunSha256": _sha256(rerun_path),
        "rerunFrames": display_count,
        "urdfAnimation": animation,
    }
    _write_json(run_dir / "evaluation.json", evaluation)
    result = {
        "status": "complete",
        "runDir": str(run_dir),
        "device": device,
        "episodes": {
            "train": len(train_episodes),
            "validation": len(validation_episodes),
            "test": len(test_episodes),
        },
        "transitions": {
            "train": len(train_transitions.states),
            "validation": len(validation_transitions.states),
            "test": len(test_transitions.states),
        },
        "smokeTest": smoke,
        "training": training,
        "testMetrics": test_metrics,
        "rerun": str(rerun_path),
    }
    _write_json(run_dir / "result.json", result)
    return result


def run_recipe(
    *,
    recipe_id: str,
    run_dir: Path,
    max_steps: int | None = None,
    smoke_test_steps: int | None = None,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    try:
        return _run_recipe(
            recipe_id=recipe_id,
            run_dir=run_dir,
            max_steps=max_steps,
            smoke_test_steps=smoke_test_steps,
            max_episodes=max_episodes,
        )
    except Exception as error:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            run_dir / "failure.json",
            {
                "status": "failed",
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
