# robot-world-models

An open-source, agent-friendly starting point for building robot world models from data and robot
descriptions discovered through WarmHub.

The repository treats these registries as co-equal sources:

- [`bencaunt/robot-datasets`](https://app.warmhub.ai/orgs/bencaunt/repos/robot-datasets) supplies
  dataset identity, modality, provenance, license, QC, and robot-link evidence.
- [`bencaunt/robot-models`](https://app.warmhub.ai/orgs/bencaunt/repos/robot-models) supplies robot
  identity, pinned URDF/MJCF descriptions, physical metadata, license, and QC.

WarmHub is the discovery and provenance control plane. Large dataset files and model assets remain
at their attributed upstream sources and are resolved from registry facts rather than silently
hard-coded.

## Current milestone

The first local spike is executable and provides:

- the workflow and contribution contract in [`SPEC.md`](SPEC.md);
- agent instructions in [`AGENTS.md`](AGENTS.md);
- the reviewed upstream interfaces in [`docs/references.md`](docs/references.md);
- validated dataset, robot, dataset-to-robot mapping, and recipe manifests;
- a generated machine-readable catalog;
- live, Arc-aware discovery across both WarmHub registries;
- reusable Hugging Face, LeRobot v2.1, and GitHub description adapters;
- SO-101 state and DINOv2 visual-latent proof-of-concept trainers for MPS, CUDA, and CPU;
- selectable tokenwise-MLP and spatial-transformer visual predictors with controlled H1/H5 recipe
  contracts;
- reusable AV1 video materialization, exact frame alignment, and revisioned visual-feature caches;
- mandatory Rerun evaluation output, including separated actual/predicted URDF animation for
  validated joint transforms and actual/predicted RGB for visual recipes;
- a cost-bounded RunPod planning contract and a separate deployment prompt.

The RunPod path is deliberately plan-only in this milestone: no command creates paid infrastructure
until the provisioning implementation and its teardown tests have been reviewed.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync
uv run rwm catalog validate
uv run rwm catalog build --check
uv run rwm warmhub discover "SO-101"
uv run pytest
```

Run the reviewed SO-101 proof of concept:

```bash
uv sync --extra train --extra lerobot
uv run rwm device
uv run rwm train so101-state-dynamics-poc --run-dir runs/so101-demo
uv run rerun runs/so101-demo/evaluation.rrd
```

Run the one-camera visual-latent proof. The five-step recipe is the current reference for
open-loop rollout work; use `so101-dinov2-visual-poc` when one-step fidelity is the priority:

```bash
uv sync --extra train --extra lerobot --extra vision
uv run rwm train so101-dinov2-visual-h5-poc --run-dir runs/so101-visual
uv run rerun runs/so101-visual/evaluation.rrd
```

Start an agent with [`prompts/create-world-model.md`](prompts/create-world-model.md). The agent
should work through data intent, mixture type, modality compatibility, local proof of concept, Rerun
evaluation, and only then an explicitly approved remote run.

## Repository growth

A new dataset normally contributes:

1. one manifest in `catalog/datasets/`;
2. a small, license-safe fixture;
3. a contract test;
4. code only when the source access method or storage format is genuinely new.

A dataset-specific feature-to-URDF mapping belongs in `catalog/mappings/`, not on the robot
manifest. Mapping entries include the numeric transform and evidence. Provisional mappings cannot
animate; validated partial mappings may animate covered joints while naming every omitted feature.

Run `uv run rwm catalog build` after adding a manifest. CI rejects invalid manifests and a stale
`catalog/catalog.json`.

## First spike result

The 2026-07-27 MPS spike used the WarmHub-resolved `nashmo/so101` dataset: 10 episodes, 8/1/1
episode split, 8,089 training transitions, a 100-step smoke test, and 2,000 training steps. It
completed training in about 3 seconds and reached held-out one-step MSE 0.0719 versus 0.3553 for the
persistence baseline. Open-loop MSE rose to 0.733 at ten steps.

See [`docs/spikes/so101-state-dynamics-2026-07-27.md`](docs/spikes/so101-state-dynamics-2026-07-27.md)
for the full result, failures, and next experiment.

The follow-on visual spike used five episodes and a frozen, revision-pinned DINOv2-S encoder. The
corrected 2.00M-parameter one-step model reduced held-out latent cosine error by 16.9% relative to
persistence and by 20.2% relative to replacing the action with its training mean. Its resize-conv
decoder reached 0.1157 ground-truth-latent RGB MAE, but the deliberately small 4x4 spatial
bottleneck still produced only coarse 64x64 reconstructions.

See [`docs/spikes/so101-visual-latent-2026-07-27.md`](docs/spikes/so101-visual-latent-2026-07-27.md)
for the corrected reference contract and initial findings.

A controlled follow-up rejected an 8x8 pooled-token default: it used 300.4 MiB of compressed
feature cache versus 120.3 MiB for 4x4 and made decoder MAE 15.1% worse. Retaining 4x4 tokens and
training with a discounted five-step open-loop loss improved held-out rollout cosine error by 10.7%
at horizon 5 and 11.5% at horizon 10, with a 5.6% one-step tradeoff. See
[`docs/spikes/so101-visual-rollout-ablation-2026-07-27.md`](docs/spikes/so101-visual-rollout-ablation-2026-07-27.md)
for the experiment matrix and recommendation.

A compact 4.50M-parameter spatiotemporal transformer was then tested against the one-step MLP
control. After correcting residual-head initialization and using a stable `0.0001` learning rate,
it beat persistence by 14.0% and improved H5/H10 open-loop latent error. However, its mean-action
ablation gap was only 0.61%, versus 20.2% for the MLP control. It is retained as an executable
negative architecture ablation, not promoted as the default. See
[`docs/spikes/so101-visual-transformer-ablation-2026-07-27.md`](docs/spikes/so101-visual-transformer-ablation-2026-07-27.md).

## Status

WarmHub resolution, bounded state-data materialization, local training, checkpointing, baseline
evaluation, one-camera visual-latent training, cached DINOv2 features, and actual/predicted Rerun
images plus five-joint URDF animation are implemented. Controlled representation ablation and
five-step visual training are also implemented. The spatial-transformer path is implemented and
tested, but remains experimental because its tiny-data run mostly ignored action. Task-relevant
visual metrics, sharper reconstruction, more action-diverse data, the SO-101 gripper transform,
and paid RunPod provisioning remain future milestones. The RunPod CLI remains intentionally
plan-only.
