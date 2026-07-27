# SO-101 state-dynamics spike — 2026-07-27

## Outcome

The first WarmHub-first local recipe completed end to end on Apple MPS. It resolved dataset and
robot facts from `bencaunt/robot-datasets` and `bencaunt/robot-models`, materialized only state/action
Parquet plus metadata, trained and checkpointed a small dynamics MLP, beat a persistence baseline,
and wrote a verified Rerun recording with separated actual/predicted animation of the pinned SO-101
URDF.

This is a pipeline proof. It is not yet a visual world model or a planning-quality rollout model.

## Reproduce

```bash
uv sync --extra train --extra lerobot
uv run rwm train so101-state-dynamics-poc --run-dir runs/so101-spike
uv run rerun rrd verify runs/so101-spike/evaluation.rrd
uv run rerun runs/so101-spike/evaluation.rrd
```

The command writes config, discovery, source preflight, materialization, normalization, smoke-test,
training, checkpoint, robot, evaluation, and result receipts under the run directory. `runs/` is
gitignored.

## Reviewed inputs

- Dataset: `Dataset/huggingface/nashmo/so101@v1`
- Dataset profile: `DatasetProfile/huggingface/nashmo/so101@v1`
- Upstream dataset commit: `7e8ba026d3040061561c9fb15976f33a4bc1d275`
- Robot: `Robot/the-robot-studio/so-arm101@v1`
- URDF description:
  `Description/github/therobotstudio/so-arm100/simulation/so101/so101-new-calib-urdf@v1`
- Upstream robot commit: `63eede5a636e548eb8f2854e558bd343c21db9f7`
- Robot-link evidence: alias map, confidence 0.9, user confirmation still required
- Recipe: homogeneous, state/action only, 10 episodes, seed 7

## Observed result

| Measurement | Result |
| --- | ---: |
| Device | MPS, PyTorch 2.11.0 |
| Split | 8 train / 1 validation / 1 test episodes |
| Transitions | 8,089 train / 1,026 validation / 987 test |
| Smoke test | 0.00491 → 0.000141 normalized MSE in 100 steps |
| Full training | 2,000 steps in about 3 seconds |
| Test one-step MAE | 0.1815 raw dataset units |
| Test one-step MSE | 0.0719 squared raw dataset units |
| Persistence MSE | 0.3553 squared raw dataset units |
| Improvement over persistence | 79.8% |
| Rollout MSE, horizon 1 / 5 / 10 | 0.0726 / 0.4706 / 0.7330 |
| Rerun | 300 frames, 118 entity paths, 1,500 dynamic transforms per robot, verified without error |

## What worked

- WarmHub facts were sufficient to resolve immutable upstream dataset and URDF commits without
  hard-coding payload locations into the trainer.
- Metadata/data-only selective materialization avoided roughly 210 MB of optional camera videos.
- Reading LeRobot v2.1 episode Parquet directly kept episode boundaries explicit and required only
  about 420 KB of numeric payload.
- The overfit smoke test caught the complete batching/model/optimizer path before the full run.
- Episode-aware splitting and train-only normalization produced a meaningful held-out baseline
  comparison.
- Rerun embedded actual/predicted scalar trajectories, metrics, provenance, two tinted URDF
  instances, and dynamic transforms for five arm joints in a valid `.rrd`.

## What did not work or remains unsafe

1. The originally reviewed `qb1t/so101-teleop-cubes` source returned
   401/repository-not-found at its pinned commit. WarmHub catalog presence alone cannot be treated as
   a payload availability guarantee.
2. LeRobot 0.6 uses dataset codebase v3.0 and intentionally rejects v2.1 repositories. The spike
   uses a narrowly versioned v2.1 Parquet adapter instead of mutating or republishing upstream data.
3. The dataset uses LeRobot's legacy degree calibration. The official compatibility conversion
   validates five arm-joint transforms into the new-calibration URDF. The gripper remains unmapped
   because its 0-100 command-to-jaw-angle conversion is not present in the robot package.
4. Across the displayed test slice, 83 of 1,500 actual mapped values (5.53%) and 83 predicted values
   exceeded current URDF limits. They are counted and clamped for rendering rather than hidden.
5. Ten-step rollout error is about ten times one-step error. The model is useful as a pipeline and
   alignment proof, not as a long-horizon planner.
6. RGB was optional and intentionally omitted. This result says nothing about visual prediction or
   object interaction.

## Spec refinements from the spike

- Add exact-revision payload access preflight before dataset approval.
- Let format adapters select payload paths so source adapters can avoid unused episodes/modalities.
- Model joint mappings as dataset-to-robot manifests rather than robot-global fields.
- Store explicit affine transforms, evidence, coverage, unmapped features, and out-of-range policy
  in each mapping manifest.
- Treat storage version as an adapter capability; do not equate installed LeRobot support with all
  historical LeRobot formats.
- Mark semantic/physical metrics as skipped when their mapping evidence is insufficient.
- Require dynamic transform rows and visual comparison of two timesteps; a static URDF import is
  not proof of motion evaluation.

## Next experiment

Validate several converted arm poses against source video and derive an authoritative SO-101
gripper transform. Then add one camera stream and test an action-conditioned visual latent
predictor locally before deciding whether a RunPod GPU is justified.
