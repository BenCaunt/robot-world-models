# AGENTS.md

This repository is designed to be operated and extended by coding agents with a human collaborator.

## Start here

1. Read `SPEC.md`.
2. Run `uv sync` and use `uv run` for every Python command.
3. Run `uv run rwm catalog validate` and `uv run pytest`.
4. Query WarmHub before external search:

   ```bash
   uv run rwm warmhub discover "<robot or task>"
   ```

5. Read `catalog/catalog.json` before adding an adapter.

Do not use `pip install` directly into the project environment. Add dependencies with `uv add`,
update `pyproject.toml`, and commit `uv.lock`.

## Collaboration workflow

For a new model request, use `prompts/create-world-model.md`.

Work with the user through these approvals:

1. model intent;
2. candidate data and evidence;
3. homogeneous or heterogeneous mixture and modality policy;
4. smallest local proof of concept;
5. Rerun evaluation;
6. remote plan, if still useful;
7. paid deployment.

Do not skip directly from a robot name to architecture or paid compute.

## WarmHub rules

- Treat `bencaunt/robot-datasets` and `bencaunt/robot-models` as co-equal inputs.
- Retain durable wrefs and observed pinned versions in run receipts.
- Normalize trailing `@vN` only for identity comparison.
- `DatasetModality` is about `Arc[Dataset -> Modality]`.
- `DescribesRobot` is about `Arc[Description -> Robot]`.
- Resolve Arc endpoints with `--resolve-collections` and filter with `--role from|to`.
- `RecordedWith` is currently about a Dataset and contains a cross-repo `robotWref`; preserve its
  `matchMethod`, `confidence`, and notes.
- Never emit Pair in a new contribution. Legacy Pair may be read only for historical compatibility.
- If WarmHub lacks a fact, label a local mapping provisional and make the registry gap visible.
- Before approving a dataset, preflight the exact upstream revision and required files. A WarmHub
  record is provenance evidence, not a guarantee that payload bytes remain public.
- Do not write to a WarmHub repo unless the user explicitly expands the task to ingestion.

## Contribution rules

A dataset normally adds a manifest, fixture, and contract test. Search for an existing:

- source adapter: transport and authentication;
- format adapter: storage and schema;
- modality mapping;
- robot mapping.

Add code only when none applies. Keep adapters independent from recipes. A recipe composes
capabilities; it does not own download or extraction logic.

Feature-name, unit, offset, and sign mappings belong to a dataset-to-robot manifest under
`catalog/mappings/`. Never put a dataset-specific feature map on a global robot manifest.
Each animated entry must carry an explicit numeric transform and evidence. Partial coverage is
allowed only when every omitted feature is named and the Rerun recording displays the limitation.

After manifest changes:

```bash
uv run rwm catalog validate
uv run rwm catalog build
uv run rwm catalog build --check
```

Do not hand-edit `catalog/catalog.json` or files under `schemas/`.

## Training rules

- Use episode-aware splits; never randomly split adjacent frames across train and validation.
- Fit normalization on training data only.
- Preserve dataset identity in heterogeneous batches.
- Seed Python, NumPy, and PyTorch.
- Start with an overfit smoke test and a bounded subset.
- Download only modalities and episodes required by the approved proof; state-only runs must not
  pull videos.
- Select devices in order requested by the recipe; support `mps`, `cuda`, then `cpu`.
- Record exact config, package lock hash, git commit, WarmHub snapshot, dataset fingerprint, device,
  seed, and checkpoint hash.
- Compare against a naive baseline.
- Skip and explain physical metrics whose units or robot mapping are not validated.

The current homogeneous reference command is:

```bash
uv sync --extra train --extra lerobot
uv run rwm train so101-state-dynamics-poc --run-dir runs/<run-id>
uv run rerun rrd verify runs/<run-id>/evaluation.rrd
```

## Rerun rules

Every evaluation must write an `.rrd`.

- Log input observations, predictions, targets, metrics, provenance, and timings.
- Resolve robot descriptions through WarmHub.
- Use the Rerun URDF importer for a URDF resolved from `robot-models`.
- Only animate joints after validating names, order, units, zero offsets, and limits.
- Use distinct entity paths and frame prefixes for actual and predicted robots.
- Separate or visually distinguish actual and predicted geometry so overlap cannot hide motion.
- Verify dynamic `Transform3D` rows on the evaluation timeline and inspect at least two different
  steps. Static URDF geometry alone does not satisfy evaluation.
- When mapping is uncertain, log a visible warning and omit misleading transforms.
- A validated partial mapping may animate its covered joints while explicitly unmapped joints stay
  at their URDF defaults.

## Secrets and RunPod

Read `SECURITY.md` and `docs/runpod-safety.md` before remote work.

- Never ask the user to paste an API key into chat.
- Read `RUNPOD_API_KEY` from the environment.
- Never print, serialize, or pass the key on the command line.
- A deploy command must default to dry-run.
- Render live GPU ID, price, maximum runtime/cost, storage, ports, and teardown before approval.
- Possession of a key is not approval.
- Use one GPU, secure cloud, no persistent volume, and no public service ports by default.
- Record the Pod ID immediately, install a watchdog, retrieve artifacts, verify checksums, terminate
  in `finally`, and verify termination.
- Never leave the user with only “stop the Pod”; stopped storage can remain billable.

The v0.1 CLI is plan-only. Do not improvise paid provisioning around that boundary. Use
`prompts/runpod-gpu.md` to implement and review the next milestone first.

## Verification

Before handing work back:

```bash
uv run ruff check .
uv run pytest
uv run rwm catalog validate
uv run rwm catalog build --check
```

Report skipped live, GPU, dataset-download, or paid-compute checks explicitly.
