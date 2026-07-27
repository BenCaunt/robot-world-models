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

This first scaffold provides:

- the workflow and contribution contract in [`SPEC.md`](SPEC.md);
- agent instructions in [`AGENTS.md`](AGENTS.md);
- the reviewed upstream interfaces in [`docs/references.md`](docs/references.md);
- validated dataset, robot, and recipe manifests;
- a generated machine-readable catalog;
- live, Arc-aware discovery across both WarmHub registries;
- an SO-101 state-dynamics proof-of-concept recipe;
- mandatory Rerun evaluation output, including a WarmHub-resolved URDF when joint semantics permit;
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

For the small PyTorch proof of concept:

```bash
uv sync --extra train
uv run rwm device
```

For LeRobot dataset support:

```bash
uv sync --extra train --extra lerobot
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

Run `uv run rwm catalog build` after adding a manifest. CI rejects invalid manifests and a stale
`catalog/catalog.json`.

## Status

The scaffold and contracts are executable. Dataset download, complete LeRobot materialization,
training, and paid RunPod provisioning are the next implementation milestones; the SO-101 recipe
makes their expected boundaries explicit.
