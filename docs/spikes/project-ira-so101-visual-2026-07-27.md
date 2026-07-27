# Project IRA SO-101 nested-collection visual spike — 2026-07-27

## Outcome

The first external data-side expansion now runs end to end on Apple MPS. It adds a reusable nested
LeRobot v3 collection adapter, exact shared-video episode slicing, byte-ceiling enforcement, a
Project IRA dataset manifest, a dataset-specific SO-101 joint mapping, and a one-camera DINOv2
visual recipe.

The 12-episode proof beat latent persistence by 60.3% and retained a 16.8% improvement over the
training-mean-action ablation. Its relative action gap did not exceed the earlier five-episode
Nashmo control's 20.2%, so the experiment validates the new data path and a materially
action-conditioned model, but does not yet prove that more diverse data increased action
sensitivity.

## WarmHub and payload boundary

The upstream Hugging Face repository is 10,943,240,337 bytes because it contains raw recordings,
one redundant recording, and two incomplete merged copies. WarmHub commit 70 added only:

- one aggregate `DatasetProfile`;
- one `RecordedWith` SO-101 link;
- six `DatasetModality` assertions about directed Arcs;
- five QC `Assessment` assertions about directed Arcs.

No Parquet, video, feature-cache, checkpoint, or Rerun bytes were written to WarmHub. The dataset
manifest makes this machine-readable with `warmhub_payload_bytes: 0`.

The full advertised set is the 46 nonredundant raw member roots, each with ten episodes: 460
episodes and 407,233 frames. The redundant root contains 50 episodes. The two merged roots contain
only 290 and 300 episodes, so neither is a complete materialization.

## Bounded qualification

The initial all-member qualification downloaded 18,113,973 bytes of Parquet and task metadata under
a 25 MB ceiling. It did not download video. Across all 460 episodes:

| Joint | Action standard deviation | Action range |
| --- | ---: | ---: |
| shoulder pan | 43.28 | -108.44 to 86.02 |
| shoulder lift | 50.87 | -104.70 to 86.42 |
| elbow flex | 44.46 | -96.48 to 96.92 |
| wrist flex | 26.02 | -36.48 to 102.59 |
| wrist roll | 20.78 | -167.25 to 27.38 |
| gripper | 14.57 | 0.38 to 100.00 |

A normalized mean/range/motion maximin heuristic selected one yellow single-colour member and one
multi-colour member for the MPS proof. Hugging Face preflight resolved ten exact metadata, Parquet,
and desk-view files totaling 65,176,159 bytes under the recipe's 75 MB hard ceiling. The
`wrist_left` videos remained upstream.

## Reproduce

```bash
uv sync --extra train --extra lerobot --extra vision
uv run rwm train project-ira-so101-dinov2-visual-poc \
  --run-dir runs/project-ira-so101-visual-poc
uv run rerun rrd verify runs/project-ira-so101-visual-poc/evaluation.rrd
uv run rerun runs/project-ira-so101-visual-poc/evaluation.rrd
```

The run loaded 12 episodes: all ten episodes from the first member and two from the second. Shared
MP4 chunks were sliced by the LeRobot v3 episode timestamp metadata. All 8,177 selected video
frames matched their numeric rows exactly. The resulting local feature cache was 192 MB.

## Result

| Measurement | Project IRA 12 episodes | Earlier Nashmo 5-episode control |
| --- | ---: | ---: |
| Train / validation / test episodes | 8 / 2 / 2 | 3 / 1 / 1 |
| MPS training time, 1,000 steps | 77.2 s | 75.8 s |
| One-step latent cosine error | 0.01184 | 0.0569 |
| Persistence cosine error | 0.02985 | 0.0684 |
| Improvement over persistence | 60.3% | 16.9% |
| Mean-action ablation error | 0.01424 | 0.0713 |
| Improvement from action | 16.8% | 20.2% |
| Rollout error H5 | 0.04807 | 0.2068 |
| Rollout error H10 | 0.06419 | 0.2737 |
| Decoder reconstruction MAE | 0.04024 | 0.1157 |
| Predicted RGB MAE | 0.04049 | 0.1160 |
| State MAE, calibrated mixed scale | 0.2212 | 0.2769 |

The lower absolute error is encouraging, but this is not a controlled dataset ablation: camera,
scene, task, and episode duration also differ. The random episode split placed both second-member
episodes in training, so validation and test measure held-out episodes from the first member rather
than source-held-out generalization.

## Calibration correction

The first Rerun pass incorrectly reused the old LeRobot degree transform because the raw ranges
looked similar. The `.pos` feature names without the old `main_` prefix are evidence for LeRobot's
current calibrated convention: nominally -100 to 100 for arm joints and 0 to 100 for the gripper.
That initial transform clamped 44.7% of plotted values and was rejected.

The corrected mapping linearly maps the calibrated arm range into the pinned new-calibration URDF
limits. On the evaluation frames, nominal-range overshoot is at most 3.38% for shoulder lift, 1.58%
for wrist flex, and 0.53% for wrist roll. Rerun still counts and clamps every overshoot, making the
20% value-level violation rate visible. The gripper remains unmapped.

## Decision and next experiment

Keep the nested collection adapter and the 12-episode recipe as the local regression proof. Keep the
tokenwise MLP as the visual control; this run does not justify returning to the transformer.

The next useful data experiment is a source-held-out split over at least four selected members.
Train on three task/setup roots and hold out the fourth as a complete source. Compare persistence
and mean-action ablations per member, not only in aggregate. That run will require more desk-view
video and feature-cache storage, so it should use a separately reviewed byte ceiling rather than
silently widening this MPS recipe.
