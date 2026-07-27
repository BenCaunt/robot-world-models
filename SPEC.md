# Robot World Models Specification

Status: draft v0.2 — first bounded SO-101 local spike implemented

## 1. Purpose

`robot-world-models` is a companion repository for coding agents and humans who want to answer:

> “Help me build a world model for this robot.”

The repository turns that request into a reviewable sequence: discover evidence through WarmHub,
design a compatible dataset mixture, train the smallest useful local proof of concept, evaluate it
with Rerun and a registry-resolved robot description, and only then scale an approved run to remote
compute.

It is a recipe and capability repository, not a warehouse. WarmHub provides identities, graph
relationships, provenance, license information, modality metadata, and QC. Dataset bytes, robot
assets, and checkpoints remain in their attributed upstream locations.

## 2. Principles

1. **WarmHub first.** Query `bencaunt/robot-datasets` and `bencaunt/robot-models` before searching
   upstream hosts directly. If registry coverage is missing, record the gap and create a provisional
   contribution rather than bypassing it silently.
2. **Two co-equal registries.** Dataset suitability cannot be inferred from robot identity alone,
   and robot compatibility cannot be inferred from a dataset name alone.
3. **Evidence is data.** Preserve match method, confidence, QC verdicts, licenses, version pins, and
   source attribution in the training plan and run receipt.
4. **Ask before optimizing.** Work through task, modality, mixture, prediction target, and evaluation
   intent with the user before selecting architecture or compute.
5. **Small proof first.** Prove loading, batching, loss reduction, checkpointing, and evaluation on
   MPS, CUDA, or CPU before requesting paid infrastructure.
6. **Rerun is required.** Every evaluation produces a local `.rrd` recording with inputs,
   predictions, metrics, provenance, and a robot description when semantics permit.
7. **Modularity compounds.** Add declarative manifests first. Add source or format code only when a
   reusable adapter does not already exist.
8. **Remote compute is an important action.** Render and approve a live-priced, cost-bounded plan
   before creating a Pod. Retrieve artifacts and terminate compute reliably.

## 3. User journey and gates

### Gate 0 — Frame the request

The agent asks enough questions to produce a `ModelIntent`:

- robot identity or a description of the robot;
- task or environment of interest;
- prediction target: state, image, latent, reward, contact, or a combination;
- planning horizon and control frequency;
- available observations and actions;
- homogeneous versus heterogeneous data intent;
- real, simulated, or mixed data;
- whether the goal is a quick feasibility result, fine-tuning an existing model, or training a new
  model;
- local hardware and storage limits;
- acceptable data licenses and use restrictions;
- success criteria.

The agent recommends a default but does not manufacture missing choices.

**Exit artifact:** `runs/<run-id>/intent.yaml`.

### Gate 1 — Discover through WarmHub

The agent queries both registries:

1. Find the durable `Robot` identity.
2. Traverse `DescribesRobot` through `Arc[Description -> Robot]`, using
   `--resolve-collections --role to`, and inspect candidate `Description`, `ModelProfile`, and
   `Assessment` records.
3. Discover candidate datasets by robot identity, name, task, and modality.
4. Read `RecordedWith` evidence, including `matchMethod` and `confidence`.
5. Traverse `DatasetModality` through `Arc[Dataset -> Modality]`, using
   `--resolve-collections --role from`.
6. Inspect `DatasetProfile`, `CatalogProfile`, licenses, source URLs, and `Assessment` records.
7. Normalize trailing `@vN` pins before client-side identity comparison while retaining the observed
   pinned wrefs in the receipt.
8. Preflight the exact upstream revision, access/gating state, required metadata, and payload paths.
   A registry record proves identity and evidence, not continued public byte availability.

New contributions must emit Arc or Bond relationships according to the live registry model. Pair is
legacy read compatibility only.

**Exit artifact:** `runs/<run-id>/discovery.json`, containing every selected wref and observed
version.

### Gate 2 — Design the data mixture

The agent builds a compatibility table before downloading payload bytes.

Every selected dataset row includes:

- durable dataset wref and observed pinned version;
- robot-link evidence;
- source host and upstream revision;
- license and restrictions;
- real/sim classification;
- task labels;
- observation, action, and state keys;
- units, coordinate frames, rates, shapes, and dtypes;
- camera count, names, viewpoints, resolution, and encoding;
- episode/frame counts;
- missing or uncertain fields;
- adapter and estimated storage requirements.

#### Homogeneous mixture

A homogeneous mixture targets one embodiment and one canonical feature schema. Datasets may differ
in scene, task, or collector, but must map cleanly to the same:

- joint ordering, action semantics, units, and coordinate frames;
- observation keys and temporal rate;
- camera roles or an explicit camera-selection policy;
- normalization policy.

This is the default for the first proof of concept.

#### Heterogeneous mixture

A heterogeneous mixture spans robots, schemas, viewpoints, domains, or action representations. It
must add:

- an embodiment or dataset identity token;
- explicit per-dataset mappings into a canonical schema;
- per-dataset normalization statistics;
- a missing-modality mask and policy;
- a balanced sampling policy that prevents large datasets from silently dominating;
- source-aware train/validation/test splits;
- a statement of which parameters are shared and which are embodiment-specific;
- per-domain metrics in addition to aggregate metrics.

The agent must not concatenate heterogeneous data and call it compatible.

#### Mixture approval

The user sees:

- recommended datasets and rejected candidates;
- homogeneous/heterogeneous classification;
- modality intersections and gaps;
- evidence and license caveats;
- estimated local download and training envelope;
- the smallest experiment that can disprove the idea.

**Exit artifact:** `runs/<run-id>/data-plan.yaml`, approved by the user.

### Gate 3 — Materialize and validate data

The pipeline resolves upstream locations from current WarmHub facts, downloads only the approved
slice, and converts it through:

```text
Dataset manifest
  -> source adapter
  -> format adapter
  -> canonical episode contract
  -> split and normalization artifacts
```

The canonical episode contract contains observations, actions, timestamps, episode boundaries,
dataset identity, robot identity, task metadata, and modality masks. Conversion must be resumable
and content-addressed. Raw files remain immutable.

Source adapters accept a format-adapter-produced list of approved payload paths. This keeps
transport independent from schema while ensuring a state-only spike does not download camera video
or unselected episodes. Storage versions are explicit capabilities: an installed library rejecting
an older dataset does not require an upstream conversion when a bounded, tested adapter can read
the immutable format directly.

For a new dataset, the normal contribution is a manifest, fixture, and contract test. A new source
adapter is justified only by a new transport or authentication method; a new format adapter is
justified only by a new storage/schema family.

**Exit artifacts:** `data/processed/<fingerprint>/manifest.json`, split indices, normalization
statistics, and adapter receipts.

### Gate 4 — Train the local proof of concept

The first experiment must be small enough to run on Apple MPS, one CUDA GPU, or CPU:

- a bounded episode and frame subset;
- deterministic seed;
- short training budget;
- frequent checkpoint and metric writes;
- no architecture sweep;
- clear overfit-smoke-test before generalization claims.

The initial SO-101 recipe uses an action-conditioned state-dynamics baseline:

```text
(state_t, action_t) -> delta_state_(t+1)
```

This baseline validates alignment, units, action/state semantics, splits, device selection, training,
checkpointing, and Rerun output. It is a pipeline proof, not the final visual world model.

Once it passes, the next model tier is an action-conditioned visual latent model:

```text
image_t -> encoder -> latent_t
(latent_t, state_t, action_t) -> latent_(t+1:t+h)
latent -> decoder or predictive head
```

Existing open weights may be fine-tuned when their license, modality contract, encoder resolution,
action representation, and embodiment assumptions match. Otherwise, begin with the small baseline
instead of forcing an incompatible checkpoint.

**Exit artifacts:** checkpoint, config, environment receipt, training metrics, and failure report if
the smoke test does not pass.

### Gate 5 — Evaluate with Rerun

Evaluation is not a terminal scalar. Every run writes `evaluation.rrd` with:

- selected observation images when RGB is part of the recipe;
- actual and predicted state/action-aligned trajectories;
- one-step and rollout errors;
- per-joint and aggregate metrics;
- dataset, robot, recipe, checkpoint, and git identities;
- split and seed;
- model latency and device;
- the WarmHub-resolved URDF/MJCF source and pinned commit;
- actual and predicted robot joint transforms when the dataset-to-URDF name and unit mapping has
  been validated.

Rerun entity paths should separate observed and predicted worlds. For URDF:

- log the description as static geometry;
- use distinct entity/frame prefixes for observed and predicted robots;
- animate joints only after mapping names, order, units, zero offsets, and limits;
- log a visible warning instead of an animation when mapping is uncertain.

Minimum state-model metrics:

- one-step MSE and MAE;
- per-joint error;
- open-loop rollout error by horizon;
- stability or limit-violation rate only when units and robot mapping are validated;
- naive persistence baseline;
- metrics per dataset for heterogeneous mixtures.

Visual-model evaluation later adds perceptual and task-relevant metrics, but visual inspection in
Rerun remains mandatory.

**Exit artifact:** `runs/<run-id>/evaluation.rrd` plus `metrics.json`.

### Gate 6 — Plan and approve RunPod

Remote compute is considered only after Gate 5 produces a usable local receipt.

The user supplies a RunPod key through the local environment, never through chat:

```bash
read -s RUNPOD_API_KEY
export RUNPOD_API_KEY
```

The planning command queries the live RunPod catalog. “RTX 5090” is a preference, not a hard-coded
ID or availability promise. The rendered plan must include:

- selected live GPU ID and display name;
- live hourly price and secure/community cloud;
- one GPU unless the approved recipe explicitly requires more;
- maximum runtime and estimated maximum compute cost;
- image and CUDA assumptions;
- container, temporary, and persistent disk sizes and costs;
- exposed ports and justification;
- repository commit and exact command;
- data and artifact transfer paths;
- watchdog, stop, termination, and recovery commands.

Defaults: secure cloud, one GPU, no persistent volume, no public service ports, explicit hourly cap,
explicit runtime cap, and terminate after verified artifact retrieval.

The agent displays the plan and waits for explicit user approval. Possession of
`RUNPOD_API_KEY` is not approval.

### Gate 7 — Execute, retrieve, evaluate, terminate

Provisioning must:

1. record the Pod ID immediately;
2. install a remote watchdog;
3. run the pinned repository commit with `uv`;
4. stream sanitized progress;
5. save resumable checkpoints;
6. produce metrics and `.rrd`;
7. retrieve weights, config, logs, metrics, and Rerun recording;
8. verify local checksums;
9. terminate the Pod in `finally`;
10. verify that no active Pod or unintended persistent volume remains.

If the agent loses contact with the training process, cleanup takes priority over further diagnosis.
Stopping alone is not the default terminal state because storage can remain billable.

Remote evaluation uses the same recipe and Rerun contract as local evaluation. A larger result is not
accepted merely because it used a larger GPU.

### Gate 8 — Contribute the capability

If the run required a new mapping or adapter, the agent prepares a pull request containing:

- declarative manifest;
- bounded fixture;
- contract tests;
- generated catalog update;
- documentation of provenance, license, and known limitations;
- no raw dataset, checkpoint, secret, or bulky `.rrd`.

Future agents search the generated catalog before writing new download or extraction code.

## 4. Repository contracts

### Dataset manifest

One file under `catalog/datasets/` identifies:

- WarmHub repo and Dataset wref;
- expected profile and relationship evidence;
- source and format adapter IDs;
- upstream revision observed during curation;
- license;
- canonical modality mapping;
- mixture compatibility labels;
- fixture and known limitations.

### Robot manifest

One file under `catalog/robots/` identifies:

- WarmHub Robot wref;
- selected Description and ModelProfile wrefs;
- pinned upstream description revision;
- source adapter;
- description format;
- licensing and QC requirements.

### Dataset-to-robot mapping manifest

One file under `catalog/mappings/` binds one dataset manifest to one robot manifest and records:

- semantic feature-to-joint correspondence;
- dataset and robot units;
- mapping validation status;
- required checks for order, offsets, signs, and limits;
- whether Rerun animation is permitted.

Mappings are not robot-global because datasets for the same embodiment can use different feature
names and numeric conventions. A provisional mapping can preserve evidence, but schema validation
prevents it from enabling animation.

### Recipe manifest

One file under `catalog/recipes/` binds:

- model intent;
- homogeneous/heterogeneous policy;
- datasets and robot;
- optional dataset-to-robot mapping;
- canonical modality contract;
- architecture and training budget;
- supported local devices;
- evaluation and Rerun requirements;
- remote-compute constraints.

### Generated catalog

`catalog/catalog.json` is deterministic and generated from validated manifests. It is the first
search surface for agents and CI. Hand editing is prohibited.

## 5. Initial SO-101 spike

Curated from live WarmHub reads on 2026-07-27:

- Robot: `bencaunt/robot-models/Robot/the-robot-studio/so-arm101`
- Description:
  `Description/github/therobotstudio/so-arm100/simulation/so101/so101-new-calib-urdf`
- Description format/license: URDF, Apache-2.0
- Dataset: `bencaunt/robot-datasets/Dataset/huggingface/nashmo/so101`
- Dataset license: Apache-2.0
- Dataset summary: 10 episodes, 10,112 frames at 30 FPS, laptop and phone RGB cameras,
  six-dimensional joint state, and six-dimensional joint action
- Registry modalities: RGB, proprioception, joint-position state, joint-position action,
  gripper command, manipulation
- Dataset QC: source listed, metadata present, feature schema parsed, robot type present, and license
  present
- Robot-link evidence: alias-derived `so101 -> so101`, confidence 0.9

Because the robot link is high-confidence alias evidence rather than an exact catalog identifier,
the first plan must show it to the user for confirmation. The URDF joint names are numeric while the
dataset uses semantic feature names; the initial mapping is provisional until values, units, offsets,
and limits are checked against real frames.

The originally selected `qb1t/so101-teleop-cubes` record remains in the catalog as useful evidence,
but its pinned Hugging Face source returned 401/repository-not-found during materialization. The
pipeline therefore returned to WarmHub discovery and selected `nashmo/so101`; it did not silently
substitute an unregistered source.

The first complete MPS run used an 8/1/1 episode split, 8,089 training transitions, a 100-step
overfit smoke test, and 2,000 training steps. Training took about 3 seconds. Held-out one-step MSE was
0.0719 versus 0.3553 for persistence; open-loop MSE grew from 0.0726 at one step to 0.733 at ten
steps. Metrics are in the dataset's unverified raw units. Rerun verified the 300-frame
actual/predicted recording and static WarmHub-resolved URDF without animating the provisional map.

## 6. Completion criteria

The local milestone is complete:

- live WarmHub discovery reproduces the SO-101 evidence;
- a bounded LeRobot slice downloads and validates through reusable adapters;
- the state-dynamics baseline trains on MPS or CUDA;
- evaluation beats a persistence baseline on held-out episodes;
- Rerun verifies metrics, actual/predicted trajectories, and the SO-101 URDF;
- the implementation leaves a reusable contribution path.

The remote milestone remains open until:

- the same locked recipe runs on one approved RunPod GPU;
- weights and evaluation artifacts return locally;
- the Pod and any billable storage are verified terminated.

## 7. Explicit non-goals for v0.1

- claiming a universal world-model architecture;
- automatic training on every discovered dataset;
- silent heterogeneous data concatenation;
- autonomous paid infrastructure creation;
- storing dataset payloads or checkpoints in Git;
- writing inferred facts back to WarmHub without a separately reviewed ingestion path.
