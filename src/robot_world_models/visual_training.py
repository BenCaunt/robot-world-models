from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_world_models.adapters.sources.github import GitHubSparseCheckoutSource
from robot_world_models.adapters.sources.huggingface import HuggingFaceDatasetSource
from robot_world_models.catalog import manifest_by_id, repository_root
from robot_world_models.contracts import (
    VISUAL_MLP_IMPLEMENTATION,
    VISUAL_TRANSFORMER_IMPLEMENTATION,
    DatasetManifest,
    JointMappingManifest,
    RecipeManifest,
    RobotManifest,
)
from robot_world_models.dataset_loading import prepare_dataset
from robot_world_models.devices import select_device
from robot_world_models.eval.rerun_eval import write_state_evaluation
from robot_world_models.training import (
    Normalization,
    TrainingSpikeError,
    _git_receipt,
    _seed_all,
    _sha256,
    _write_json,
    fit_normalization,
    split_canonical_episodes,
    transitions_from_episodes,
)
from robot_world_models.visual_data import (
    CachedVisualEpisode,
    DinoV2FeatureEncoder,
    cache_visual_episode,
    load_cached_visual_episode,
)
from robot_world_models.warmhub import WarmHubCLI

if TYPE_CHECKING:
    from robot_world_models.models.visual_latent import VisualLatentDynamics
    from robot_world_models.models.visual_transformer import (
        VisualSpatiotemporalTransformer,
    )

    VisualModel = VisualLatentDynamics | VisualSpatiotemporalTransformer
else:
    VisualModel = Any


@dataclass(frozen=True)
class WindowRef:
    episode: int
    target: int


def visual_window_refs(
    episodes: list[CachedVisualEpisode],
    context_frames: int,
    rollout_horizon: int = 1,
) -> list[WindowRef]:
    return [
        WindowRef(episode=episode_index, target=target)
        for episode_index, episode in enumerate(episodes)
        for target in range(
            context_frames,
            len(episode.states) - rollout_horizon + 1,
        )
    ]


def _make_model(recipe: RecipeManifest) -> VisualModel:
    vision = recipe.model.vision
    if vision is None:
        raise TrainingSpikeError("visual trainer requires model.vision")
    common = {
        "state_dimension": recipe.model.state_dimension,
        "action_dimension": recipe.model.action_dimension,
        "latent_dimension": vision.encoder.latent_dimension,
        "context_frames": vision.context_frames,
        "patch_grid": vision.encoder.patch_pool_grid,
        "output_size": vision.output_size,
        "hidden_dimension": vision.predictor_hidden_dimension,
        "hidden_layers": vision.predictor_hidden_layers,
    }
    if recipe.model.implementation == VISUAL_MLP_IMPLEMENTATION:
        from robot_world_models.models.visual_latent import VisualLatentDynamics

        return VisualLatentDynamics(**common)
    if recipe.model.implementation == VISUAL_TRANSFORMER_IMPLEMENTATION:
        from robot_world_models.models.visual_transformer import (
            VisualSpatiotemporalTransformer,
        )

        if vision.attention_heads is None:
            raise TrainingSpikeError("visual transformer requires attention_heads")
        return VisualSpatiotemporalTransformer(
            **common,
            attention_heads=vision.attention_heads,
        )
    raise TrainingSpikeError(
        f"unsupported visual model implementation: {recipe.model.implementation}"
    )


def _batch_arrays(
    episodes: list[CachedVisualEpisode],
    refs: list[WindowRef],
    normalization: Normalization,
    context_frames: int,
    rollout_horizon: int,
) -> tuple[np.ndarray, ...]:
    contexts = np.stack(
        [
            episodes[ref.episode].features[ref.target - context_frames : ref.target]
            for ref in refs
        ]
    )
    states = np.stack([episodes[ref.episode].states[ref.target - 1] for ref in refs])
    actions = np.stack(
        [
            episodes[ref.episode].actions[
                ref.target - 1 : ref.target - 1 + rollout_horizon
            ]
            for ref in refs
        ]
    )
    target_features = np.stack(
        [
            episodes[ref.episode].features[
                ref.target : ref.target + rollout_horizon
            ]
            for ref in refs
        ]
    )
    target_states = np.stack(
        [
            episodes[ref.episode].states[
                ref.target : ref.target + rollout_horizon
            ]
            for ref in refs
        ]
    )
    target_frames = np.stack([episodes[ref.episode].frames[ref.target] for ref in refs])
    return (
        contexts,
        (states - normalization.state_mean) / normalization.state_std,
        (actions - normalization.action_mean) / normalization.action_std,
        target_features,
        (target_states - normalization.state_mean) / normalization.state_std,
        target_frames,
    )


def _torch_batch(arrays: tuple[np.ndarray, ...], device: str):
    import torch

    contexts, states, actions, target_features, target_states, target_frames = arrays
    return (
        torch.from_numpy(contexts).to(device),
        torch.from_numpy(states.astype(np.float32)).to(device),
        torch.from_numpy(actions.astype(np.float32)).to(device),
        torch.from_numpy(target_features).to(device),
        torch.from_numpy(target_states.astype(np.float32)).to(device),
        torch.from_numpy(target_frames).permute(0, 3, 1, 2).float().div(255).to(device),
    )


def _loss(
    *,
    model: VisualModel,
    batch,
    recipe: RecipeManifest,
):
    import torch
    from torch.nn import functional as functional

    contexts, states, actions, target_features, target_states, target_frames = batch
    vision = recipe.model.vision
    assert vision is not None
    predicted_by_step = []
    latent_by_step = []
    state_by_step = []
    rollout_context = contexts
    rollout_state = states
    for step in range(vision.training_rollout_horizon):
        predicted_features, predicted_states = model(
            rollout_context,
            rollout_state,
            actions[:, step],
        )
        predicted_by_step.append(predicted_features)
        latent_by_step.append(
            (
                1
                - functional.cosine_similarity(
                    predicted_features,
                    target_features[:, step],
                    dim=-1,
                )
            ).mean()
        )
        state_by_step.append(
            functional.mse_loss(predicted_states, target_states[:, step])
        )
        rollout_context = torch.cat(
            (rollout_context[:, 1:], predicted_features[:, None]),
            dim=1,
        )
        rollout_state = predicted_states
    weights = [
        vision.rollout_loss_discount**step
        for step in range(vision.training_rollout_horizon)
    ]
    weight_sum = sum(weights)
    latent = sum(
        weight * value for weight, value in zip(weights, latent_by_step, strict=True)
    ) / weight_sum
    state = sum(
        weight * value for weight, value in zip(weights, state_by_step, strict=True)
    ) / weight_sum
    target_reconstruction = model.decode(target_features[:, 0])
    predicted_reconstruction = model.decode(predicted_by_step[0])
    decoder = functional.l1_loss(target_reconstruction, target_frames)
    predicted_pixel = functional.l1_loss(predicted_reconstruction, target_frames)
    total = (
        latent
        + vision.state_loss_weight * state
        + vision.decoder_loss_weight * decoder
        + vision.predicted_pixel_loss_weight * predicted_pixel
    )
    terms = {
        "total": float(total.detach().item()),
        "latentCosineError": float(latent.detach().item()),
        "normalizedStateMse": float(state.detach().item()),
        "decoderL1": float(decoder.detach().item()),
        "predictedPixelL1": float(predicted_pixel.detach().item()),
    }
    terms.update(
        {
            f"latentCosineErrorH{step + 1}": float(value.detach().item())
            for step, value in enumerate(latent_by_step)
        }
    )
    return total, terms


def _smoke_test(
    *,
    recipe: RecipeManifest,
    episodes: list[CachedVisualEpisode],
    refs: list[WindowRef],
    normalization: Normalization,
    device: str,
    steps: int,
) -> dict[str, float | int | bool]:
    import torch

    batch_refs = refs[: min(recipe.training.batch_size, len(refs))]
    batch = _torch_batch(
        _batch_arrays(
            episodes,
            batch_refs,
            normalization,
            recipe.model.vision.context_frames,  # type: ignore[union-attr]
            recipe.model.vision.training_rollout_horizon,  # type: ignore[union-attr]
        ),
        device,
    )
    _seed_all(recipe.training.seed)
    model = _make_model(recipe).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.training.learning_rate)
    model.train()
    initial, initial_terms = _loss(model=model, batch=batch, recipe=recipe)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _loss(model=model, batch=batch, recipe=recipe)
        loss.backward()
        optimizer.step()
    final, final_terms = _loss(model=model, batch=batch, recipe=recipe)
    passed = float(final.item()) < float(initial.item())
    if not passed:
        raise TrainingSpikeError(
            f"visual overfit smoke test did not reduce loss: {initial.item()} -> {final.item()}"
        )
    return {
        "steps": steps,
        "batchSize": len(batch_refs),
        "initialLoss": float(initial.item()),
        "finalLoss": float(final.item()),
        "initialTerms": initial_terms,
        "finalTerms": final_terms,
        "passed": passed,
    }


def _train(
    *,
    recipe: RecipeManifest,
    episodes: list[CachedVisualEpisode],
    refs: list[WindowRef],
    normalization: Normalization,
    device: str,
    steps: int,
    checkpoint_dir: Path,
):
    import torch

    _seed_all(recipe.training.seed)
    model = _make_model(recipe).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.training.learning_rate)
    generator = np.random.default_rng(recipe.training.seed)
    history: list[dict[str, Any]] = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model.train()
    vision = recipe.model.vision
    assert vision is not None
    for step in range(1, steps + 1):
        selected = generator.integers(0, len(refs), size=recipe.training.batch_size)
        batch_refs = [refs[index] for index in selected]
        batch = _torch_batch(
            _batch_arrays(
                episodes,
                batch_refs,
                normalization,
                vision.context_frames,
                vision.training_rollout_horizon,
            ),
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, terms = _loss(model=model, batch=batch, recipe=recipe)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            history.append({"step": step, **terms})
        if step % recipe.training.checkpoint_every_steps == 0 and step != steps:
            torch.save(
                {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
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
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(final_path),
        "checkpointSha256": _sha256(final_path),
    }


def _step_arrays(
    *,
    model: VisualModel,
    contexts: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    device: str,
    decode: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    import torch

    feature_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(contexts), 128):
            stop = start + 128
            context = torch.from_numpy(contexts[start:stop]).to(device)
            state = torch.from_numpy(states[start:stop].astype(np.float32)).to(device)
            action = torch.from_numpy(actions[start:stop].astype(np.float32)).to(device)
            features, predicted_state = model(context, state, action)
            feature_parts.append(features.cpu().numpy())
            state_parts.append(predicted_state.cpu().numpy())
            if decode:
                frames = model.decode(features).mul(255).clamp(0, 255).byte()
                frame_parts.append(frames.permute(0, 2, 3, 1).cpu().numpy())
    return (
        np.concatenate(feature_parts),
        np.concatenate(state_parts),
        np.concatenate(frame_parts) if frame_parts else None,
    )


def _decode_arrays(
    *,
    model: VisualModel,
    features: np.ndarray,
    device: str,
) -> np.ndarray:
    import torch

    frame_parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 128):
            batch = torch.from_numpy(features[start : start + 128]).to(device)
            frames = model.decode(batch).mul(255).clamp(0, 255).byte()
            frame_parts.append(frames.permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(frame_parts)


def _evaluate(
    *,
    model: VisualModel,
    episodes: list[CachedVisualEpisode],
    normalization: Normalization,
    context_frames: int,
    horizons: list[int],
    device: str,
) -> dict[str, float]:
    refs = visual_window_refs(episodes, context_frames)
    metrics_parts: dict[str, list[np.ndarray]] = {
        "latent": [],
        "persistence": [],
        "action_ablation": [],
        "state_absolute": [],
        "pixel_absolute": [],
        "decoder_absolute": [],
    }
    started = time.monotonic()
    for start in range(0, len(refs), 128):
        batch_refs = refs[start : start + 128]
        arrays = _batch_arrays(
            episodes,
            batch_refs,
            normalization,
            context_frames,
            rollout_horizon=1,
        )
        contexts, states, actions, targets, target_states, target_frames = arrays
        actions = actions[:, 0]
        targets = targets[:, 0]
        target_states = target_states[:, 0]
        predictions, predicted_states, predicted_frames = _step_arrays(
            model=model,
            contexts=contexts,
            states=states,
            actions=actions,
            device=device,
            decode=True,
        )
        ablated, _, _ = _step_arrays(
            model=model,
            contexts=contexts,
            states=states,
            actions=np.zeros_like(actions),
            device=device,
        )
        metrics_parts["latent"].append(1 - np.sum(predictions * targets, axis=-1).mean(axis=1))
        metrics_parts["persistence"].append(
            1 - np.sum(contexts[:, -1] * targets, axis=-1).mean(axis=1)
        )
        metrics_parts["action_ablation"].append(
            1 - np.sum(ablated * targets, axis=-1).mean(axis=1)
        )
        state_error = (
            predicted_states * normalization.state_std
            + normalization.state_mean
            - (
                target_states * normalization.state_std
                + normalization.state_mean
            )
        )
        metrics_parts["state_absolute"].append(np.abs(state_error))
        assert predicted_frames is not None
        metrics_parts["pixel_absolute"].append(
            np.abs(predicted_frames.astype(np.float32) - target_frames).mean(axis=(1, 2, 3))
            / 255
        )
        decoded_targets = _decode_arrays(model=model, features=targets, device=device)
        metrics_parts["decoder_absolute"].append(
            np.abs(decoded_targets.astype(np.float32) - target_frames).mean(axis=(1, 2, 3))
            / 255
        )
    metrics = {
        "one_step_latent_cosine_error": float(np.concatenate(metrics_parts["latent"]).mean()),
        "latent_persistence_baseline_cosine_error": float(
            np.concatenate(metrics_parts["persistence"]).mean()
        ),
        "mean_action_ablation_cosine_error": float(
            np.concatenate(metrics_parts["action_ablation"]).mean()
        ),
        "one_step_state_mae": float(
            np.concatenate(metrics_parts["state_absolute"]).mean()
        ),
        "one_step_pixel_mae_0_1": float(
            np.concatenate(metrics_parts["pixel_absolute"]).mean()
        ),
        "decoder_reconstruction_mae_0_1": float(
            np.concatenate(metrics_parts["decoder_absolute"]).mean()
        ),
        "inference_ms_per_transition": (
            (time.monotonic() - started) * 1000 / len(refs)
        ),
    }
    maximum_horizon = max(horizons)
    rollout_errors: dict[int, list[np.ndarray]] = {horizon: [] for horizon in horizons}
    for episode in episodes:
        starts = np.arange(context_frames, len(episode.states) - maximum_horizon + 1)
        if not len(starts):
            continue
        contexts = np.stack(
            [
                episode.features[target - context_frames : target]
                for target in starts
            ]
        )
        states = (
            episode.states[starts - 1] - normalization.state_mean
        ) / normalization.state_std
        for step in range(1, maximum_horizon + 1):
            action_indices = starts + step - 2
            actions = (
                episode.actions[action_indices] - normalization.action_mean
            ) / normalization.action_std
            predicted, states, _ = _step_arrays(
                model=model,
                contexts=contexts,
                states=states,
                actions=actions,
                device=device,
            )
            contexts = np.concatenate((contexts[:, 1:], predicted[:, None]), axis=1)
            if step in rollout_errors:
                targets = episode.features[starts + step - 1]
                rollout_errors[step].append(
                    1 - np.sum(predicted * targets, axis=-1).mean(axis=1)
                )
    metrics.update(
        {
            f"rollout_latent_cosine_error_h{horizon}": float(
                np.concatenate(errors).mean()
            )
            for horizon, errors in rollout_errors.items()
            if errors
        }
    )
    metrics["improvement_over_latent_persistence_fraction"] = float(
        1
        - metrics["one_step_latent_cosine_error"]
        / metrics["latent_persistence_baseline_cosine_error"]
    )
    metrics["improvement_from_action_fraction"] = float(
        1
        - metrics["one_step_latent_cosine_error"]
        / metrics["mean_action_ablation_cosine_error"]
    )
    return metrics


def _evaluate_action_baselines(
    *,
    model: VisualModel,
    episodes: list[CachedVisualEpisode],
    normalization: Normalization,
    context_frames: int,
    device: str,
) -> dict[str, float | int]:
    refs = visual_window_refs(episodes, context_frames)
    if not refs:
        raise TrainingSpikeError("source member has no visual evaluation windows")
    latent_parts: list[np.ndarray] = []
    persistence_parts: list[np.ndarray] = []
    ablation_parts: list[np.ndarray] = []
    for start in range(0, len(refs), 128):
        arrays = _batch_arrays(
            episodes,
            refs[start : start + 128],
            normalization,
            context_frames,
            rollout_horizon=1,
        )
        contexts, states, actions, targets, _, _ = arrays
        actions = actions[:, 0]
        targets = targets[:, 0]
        predictions, _, _ = _step_arrays(
            model=model,
            contexts=contexts,
            states=states,
            actions=actions,
            device=device,
        )
        ablated, _, _ = _step_arrays(
            model=model,
            contexts=contexts,
            states=states,
            actions=np.zeros_like(actions),
            device=device,
        )
        latent_parts.append(
            1 - np.sum(predictions * targets, axis=-1).mean(axis=1)
        )
        persistence_parts.append(
            1 - np.sum(contexts[:, -1] * targets, axis=-1).mean(axis=1)
        )
        ablation_parts.append(
            1 - np.sum(ablated * targets, axis=-1).mean(axis=1)
        )
    latent = float(np.concatenate(latent_parts).mean())
    persistence = float(np.concatenate(persistence_parts).mean())
    ablation = float(np.concatenate(ablation_parts).mean())
    return {
        "visual_window_count": len(refs),
        "one_step_latent_cosine_error": latent,
        "latent_persistence_baseline_cosine_error": persistence,
        "mean_action_ablation_cosine_error": ablation,
        "latent_improvement_over_persistence_absolute": persistence - latent,
        "action_ablation_gap_absolute": ablation - latent,
        "improvement_over_latent_persistence_fraction": 1 - latent / persistence,
        "improvement_from_action_fraction": 1 - latent / ablation,
    }


def _evaluate_action_baselines_by_source(
    *,
    model: VisualModel,
    episodes: list[CachedVisualEpisode],
    split: dict[str, list[str]],
    normalization: Normalization,
    context_frames: int,
    device: str,
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[CachedVisualEpisode]] = {}
    for episode in episodes:
        if episode.source_member is None:
            raise TrainingSpikeError(
                "per-source visual metrics require source_member on every episode"
            )
        grouped.setdefault(episode.source_member, []).append(episode)
    role_by_episode = {
        episode_id: role
        for role, episode_ids in split.items()
        for episode_id in episode_ids
    }
    return {
        source: {
            "episodes": len(member_episodes),
            "splitRoles": sorted(
                {role_by_episode[episode.episode_id] for episode in member_episodes}
            ),
            "metrics": _evaluate_action_baselines(
                model=model,
                episodes=member_episodes,
                normalization=normalization,
                context_frames=context_frames,
                device=device,
            ),
        }
        for source, member_episodes in sorted(grouped.items())
    }


def _visual_rollout(
    *,
    model: VisualModel,
    episode: CachedVisualEpisode,
    normalization: Normalization,
    context_frames: int,
    device: str,
    length: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    maximum_start = len(episode.states) - length
    candidates = np.arange(context_frames, maximum_start)
    if not len(candidates):
        raise TrainingSpikeError("test episode is too short for a visual rollout")
    scores = np.asarray(
        [
            np.abs(np.diff(episode.states[start : start + length], axis=0)).mean()
            for start in candidates
        ]
    )
    target = int(candidates[int(scores.argmax())])
    context = episode.features[target - context_frames : target][None]
    state = (episode.states[target - 1] - normalization.state_mean)[None] / normalization.state_std
    predicted_states: list[np.ndarray] = []
    predicted_frames: list[np.ndarray] = []
    for offset in range(length):
        action_index = target + offset - 1
        action = (
            episode.actions[action_index] - normalization.action_mean
        )[None] / normalization.action_std
        predicted, state, frame = _step_arrays(
            model=model,
            contexts=context,
            states=state,
            actions=action,
            device=device,
            decode=True,
        )
        context = np.concatenate((context[:, 1:], predicted[:, None]), axis=1)
        predicted_states.append(
            state[0] * normalization.state_std + normalization.state_mean
        )
        assert frame is not None
        predicted_frames.append(frame[0])
    return (
        episode.states[target : target + length],
        np.stack(predicted_states),
        np.stack(predicted_frames),
        target,
    )


def _write_rollout_preview(
    *,
    output_path: Path,
    actual_frames: np.ndarray,
    predicted_frames: np.ndarray,
) -> Path:
    from PIL import Image, ImageDraw

    offsets = sorted(
        {
            0,
            min(1, len(actual_frames) - 1),
            min(4, len(actual_frames) - 1),
            min(9, len(actual_frames) - 1),
            min(19, len(actual_frames) - 1),
            len(actual_frames) - 1,
        }
    )
    cell_size = 192
    label_height = 24
    canvas = Image.new(
        "RGB",
        (cell_size * len(offsets), cell_size * 2 + label_height * 2),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, offset in enumerate(offsets):
        for row, frames in enumerate((actual_frames, predicted_frames)):
            image = Image.fromarray(frames[offset]).resize(
                (cell_size, cell_size),
                Image.Resampling.NEAREST,
            )
            canvas.paste(
                image,
                (column * cell_size, label_height + row * cell_size),
            )
        draw.text((column * cell_size + 4, 4), f"t+{offset + 1}", fill="black")
    draw.text(
        (4, label_height + 4),
        "actual",
        fill="white",
        stroke_width=2,
        stroke_fill="black",
    )
    draw.text(
        (4, label_height + cell_size + 4),
        "predicted",
        fill="white",
        stroke_width=2,
        stroke_fill="black",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def run_visual_recipe(
    *,
    recipe: RecipeManifest,
    run_dir: Path,
    max_steps: int | None,
    smoke_test_steps: int | None,
    max_episodes: int | None,
) -> dict[str, Any]:
    vision = recipe.model.vision
    if vision is None:
        raise TrainingSpikeError("visual recipe is missing its vision contract")
    if recipe.mixture.type != "homogeneous" or len(recipe.mixture.datasets) != 1:
        raise TrainingSpikeError("the first visual trainer supports one homogeneous dataset")
    dataset = manifest_by_id(recipe.mixture.datasets[0])
    robot = manifest_by_id(recipe.mixture.robot)
    mapping = manifest_by_id(recipe.joint_mapping) if recipe.joint_mapping else None
    if not isinstance(dataset, DatasetManifest) or not isinstance(robot, RobotManifest):
        raise TrainingSpikeError("visual recipe references invalid dataset or robot manifests")
    if mapping is not None and not isinstance(mapping, JointMappingManifest):
        raise TrainingSpikeError("visual recipe joint_mapping is not a mapping manifest")
    if vision.camera not in dataset.episode_schema.cameras:
        raise TrainingSpikeError(f"dataset does not declare camera {vision.camera}")

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
    backbone_matches = {
        "models": warmhub.exact_match("dinov2", warmhub.models_repo).get("items", []),
        "datasets": warmhub.exact_match("dinov2", warmhub.datasets_repo).get("items", []),
    }
    if vision.encoder.warmhub_resolution == "registry-gap" and any(backbone_matches.values()):
        raise TrainingSpikeError(
            "the visual encoder recipe declares a WarmHub registry gap, "
            "but an exact match now exists"
        )
    discovery = {
        "sources": {
            "datasets": warmhub.datasets_repo,
            "models": warmhub.models_repo,
        },
        "dataset": dataset_resolution,
        "robot": robot_resolution,
        "visualEncoder": {
            "query": "dinov2",
            "exactMatches": backbone_matches,
            "status": vision.encoder.warmhub_resolution,
            "fallback": {
                "modelId": vision.encoder.model_id,
                "revision": vision.encoder.revision,
                "sourceUrl": vision.encoder.source_url,
                "license": vision.encoder.license,
            },
        },
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
        cameras=[vision.camera],
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
    device = select_device(recipe.training.local_devices)
    encoder = DinoV2FeatureEncoder(
        model_id=vision.encoder.model_id,
        revision=vision.encoder.revision,
        device=device,
        patch_pool_grid=vision.encoder.patch_pool_grid,
        expected_input_size=vision.encoder.input_size,
    )
    if encoder.hidden_size != vision.encoder.latent_dimension:
        raise TrainingSpikeError("materialized encoder dimension differs from the recipe")
    cache_receipts = []
    for episode in episodes:
        episode_index = int(episode.episode_id)
        video_segment = prepared.video_segment(
            source_receipt,
            camera=vision.camera,
            episode=episode,
        )
        cache_receipts.append(
            cache_visual_episode(
                episode_id=episode.episode_id,
                source_member=episode.source_member,
                video_path=video_segment.path,
                video_start_seconds=video_segment.start_seconds,
                video_frame_count=video_segment.frame_count,
                states=np.asarray(episode.observations["state"], dtype=np.float32),
                actions=np.asarray(episode.actions, dtype=np.float32),
                encoder=encoder,
                encoder_batch_size=vision.encoder_batch_size,
                output_size=vision.output_size,
                cache_path=run_dir / "features" / f"episode-{episode_index:06d}.npz",
            )
        )
    _write_json(
        run_dir / "feature-cache.json",
        {
            "encoder": vision.encoder.model_dump(mode="json"),
            "camera": vision.camera,
            "alignment": "one decoded video frame per LeRobot row; exact count required",
            "cacheFormat": "NPZ: float16 L2-normalized pooled patch tokens + uint8 RGB",
            "rgbTargetTransform": (
                "the exact DINOv2 processor crop, denormalized and resized to output_size"
            ),
            "episodes": cache_receipts,
        },
    )
    cached_by_id = {
        episode.episode_id: load_cached_visual_episode(
            episode.episode_id,
            run_dir / "features" / f"episode-{int(episode.episode_id):06d}.npz",
            source_member=episode.source_member,
        )
        for episode in episodes
    }
    _write_json(
        run_dir / "materialization.json",
        {
            "inspection": inspection,
            "episodesLoaded": len(episodes),
            "split": split,
            "sourceMembers": {
                episode.episode_id: episode.source_member for episode in episodes
            },
            "transitionConvention": (
                "camera[t-context:t] + state[t-1] + "
                "action[t-1:t-1+h] -> camera latent[t:t+h] + state[t:t+h]"
            ),
            "trainingRolloutHorizon": vision.training_rollout_horizon,
            "rolloutLossDiscount": vision.rollout_loss_discount,
            "episodeBoundaryTransitions": False,
            "videosDownloaded": True,
            "camera": vision.camera,
        },
    )

    train_episodes = [cached_by_id[identifier] for identifier in split["train"]]
    validation_episodes = [cached_by_id[identifier] for identifier in split["validation"]]
    test_episodes = [cached_by_id[identifier] for identifier in split["test"]]
    numeric_train = [by_id[identifier] for identifier in split["train"]]
    normalization = fit_normalization(transitions_from_episodes(numeric_train))
    _write_json(run_dir / "normalization.json", normalization.to_json())
    train_refs = visual_window_refs(
        train_episodes,
        vision.context_frames,
        vision.training_rollout_horizon,
    )
    smoke = _smoke_test(
        recipe=recipe,
        episodes=train_episodes,
        refs=train_refs,
        normalization=normalization,
        device=device,
        steps=effective_smoke_steps,
    )
    _write_json(run_dir / "smoke-test.json", smoke)
    model, training = _train(
        recipe=recipe,
        episodes=train_episodes,
        refs=train_refs,
        normalization=normalization,
        device=device,
        steps=effective_steps,
        checkpoint_dir=run_dir / "checkpoints",
    )
    _write_json(run_dir / "training.json", training)
    validation_metrics = _evaluate(
        model=model,
        episodes=validation_episodes,
        normalization=normalization,
        context_frames=vision.context_frames,
        horizons=recipe.evaluation.rollout_horizons,
        device=device,
    )
    test_metrics = _evaluate(
        model=model,
        episodes=test_episodes,
        normalization=normalization,
        context_frames=vision.context_frames,
        horizons=recipe.evaluation.rollout_horizons,
        device=device,
    )
    per_member_metrics = _evaluate_action_baselines_by_source(
        model=model,
        episodes=list(cached_by_id.values()),
        split=split,
        normalization=normalization,
        context_frames=vision.context_frames,
        device=device,
    )

    description = robot_resolution["description"]["data"]
    robot_receipt = GitHubSparseCheckoutSource().fetch_package(
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
    actual_states, predicted_states, predicted_frames, rollout_start = _visual_rollout(
        model=model,
        episode=test_episodes[0],
        normalization=normalization,
        context_frames=vision.context_frames,
        device=device,
    )
    actual_frames = test_episodes[0].frames[
        rollout_start : rollout_start + len(predicted_frames)
    ]
    preview_path = _write_rollout_preview(
        output_path=run_dir / "visual-rollout-preview.png",
        actual_frames=actual_frames,
        predicted_frames=predicted_frames,
    )
    provenance = {
        "recipeId": recipe.id,
        "datasetWref": dataset_resolution["dataset"]["pinnedWref"],
        "robotWref": robot_resolution["robot"]["pinnedWref"],
        "visualEncoder": vision.encoder.model_dump(mode="json"),
        "trainingRolloutHorizon": vision.training_rollout_horizon,
        "rolloutLossDiscount": vision.rollout_loss_discount,
        "warmhubEncoderResolution": vision.encoder.warmhub_resolution,
        "upstreamDatasetRevision": source_receipt.source_revision,
        "datasetContentSha256": source_receipt.content_sha256,
        "robotPackageSha256": robot_receipt.content_sha256,
        "uvLockSha256": _sha256(repository_root() / "uv.lock"),
        "git": _git_receipt(),
        "device": device,
        "seed": recipe.training.seed,
        "split": split,
        "sourceMembers": {
            source: report["splitRoles"]
            for source, report in per_member_metrics.items()
        },
        "camera": vision.camera,
        "visualRolloutEpisode": test_episodes[0].episode_id,
        "visualRolloutStartFrame": rollout_start,
    }
    animation_enabled = bool(
        mapping is not None
        and mapping.mapping.status == "validated"
        and mapping.mapping.animate_in_rerun
    )
    rerun_metrics = dict(test_metrics)
    for source, report in per_member_metrics.items():
        member_metrics = report["metrics"]
        assert isinstance(member_metrics, dict)
        for metric_name in (
            "one_step_latent_cosine_error",
            "latent_persistence_baseline_cosine_error",
            "mean_action_ablation_cosine_error",
            "latent_improvement_over_persistence_absolute",
            "action_ablation_gap_absolute",
            "improvement_over_latent_persistence_fraction",
            "improvement_from_action_fraction",
        ):
            rerun_metrics[f"source/{source}/{metric_name}"] = float(
                member_metrics[metric_name]
            )
    rerun_path, animation = write_state_evaluation(
        output_path=run_dir / "evaluation.rrd",
        run_id=run_dir.name,
        joint_names=dataset.episode_schema.state.names,
        actual_states=actual_states.tolist(),
        predicted_states=predicted_states.tolist(),
        metrics=rerun_metrics,
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
        actual_images=list(actual_frames),
        predicted_images=list(predicted_frames),
    )
    evaluation = {
        "validation": validation_metrics,
        "test": test_metrics,
        "perMember": per_member_metrics,
        "metricUnits": {
            "latentCosineError": "unitless; lower is better",
            "pixelMae": "normalized RGB [0,1]",
            "stateMae": dataset.episode_schema.state.units,
        },
        "rerun": str(rerun_path),
        "rerunSha256": _sha256(rerun_path),
        "rerunFrames": len(predicted_frames),
        "preview": str(preview_path),
        "previewSha256": _sha256(preview_path),
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
        "visualWindows": {
            "train": len(train_refs),
            "validation": len(visual_window_refs(validation_episodes, vision.context_frames)),
            "test": len(visual_window_refs(test_episodes, vision.context_frames)),
        },
        "trainingRolloutHorizon": vision.training_rollout_horizon,
        "rolloutLossDiscount": vision.rollout_loss_discount,
        "smokeTest": smoke,
        "training": training,
        "testMetrics": test_metrics,
        "perMemberMetrics": per_member_metrics,
        "rerun": str(rerun_path),
        "preview": str(preview_path),
    }
    _write_json(run_dir / "result.json", result)
    return result
