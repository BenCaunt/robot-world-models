# Project IRA SO-101 source-held-out visual spike — 2026-07-27

## Outcome

The four-member MPS run completed end to end with source identity preserved from the nested
LeRobot collection through splitting, feature caching, evaluation, and Rerun provenance. Three
development members contributed nine training episodes and one validation episode each. All ten
episodes from the yellow-plate member were excluded from training, validation, and normalization.

The initial yellow-plate holdout beat visual persistence by 26.5%, but replacing the real action
with the training-mean action worsened error by only 5.7%. Rotating every member through the test
role then showed action benefits of 8.8% on red, 15.5% on green variation, 9.6% on multi-color, and
5.7% on yellow plate. Every fold beat persistence, but the action response remains source-dependent
and below the earlier 20.2% five-episode control. Improve source coverage and sampling before
scaling the architecture or renting a GPU.

## Exact data and storage boundary

The recipe selected one camera and 40 episodes from four of the 46 reviewed raw members:

| Member | Role | Frames | Selected upstream bytes | Local feature cache |
| --- | --- | ---: | ---: | ---: |
| `2026_06_11-19h48m_Sort_Lego_Color_red_1` | 9 train + 1 validation | 6,941 | 17.51 MB | 163.0 MiB |
| `2026_06_12-14h28m_Sort_Lego_Color_green_var_1` | 9 train + 1 validation | 9,144 | 20.50 MB | 214.8 MiB |
| `2026_06_03-20h21m_Sort_Lego_Color_multi_3` | 9 train + 1 validation | 14,077 | 18.93 MB | 330.6 MiB |
| `2026_07_03-13h24m_Sort_Lego_Color_yellow_plate_1` | 10 test | 5,578 | 46.25 MB | 131.0 MiB |
| **Total** | **27 / 3 / 10 episodes** | **35,740** | **103.19 MB** | **839.4 MiB** |

The exact transfer was 103,190,662 bytes in 20 files under a fail-closed 105 MB recipe ceiling.
The upstream repository is 10.94 GB. Wrist video, unselected members, checkpoints, features, and
Rerun data were not written to WarmHub; its payload contribution for this dataset remains zero
bytes.

One unauthenticated Hugging Face probe transiently returned 401 before the run. The normal pipeline
preflight subsequently resolved the pinned commit and completed without credentials. A single
authorization-shaped response is not proof of a durable visibility change; preflight should be
retried once through the normal adapter and still fail visibly if access remains unavailable.

## Split contract

Every canonical episode from a collection now carries `source_member`. A source recipe names
`test_member_roots` explicitly. The splitter:

1. assigns every episode from those roots to test;
2. independently shuffles each remaining development member;
3. selects one validation episode per development member for this recipe;
4. fits state and action normalization only on the 27 training episodes;
5. records every episode-to-source and episode-to-split assignment.

The source-held-out test is therefore leakage-free. Per-member diagnostic metrics use all ten
episodes from a member; the development-member rows below include training data and must not be
read as held-out generalization estimates. Their purpose is to expose source-dependent persistence
and action sensitivity.

The rotation command reuses the completed primary fold, verifies the encoder/camera contract and
all 40 cache checksums, then trains only the three missing folds. Each fold has its own training
normalization, split, checkpoint, metrics, preview, and Rerun recording. Completed folds are
resumable.

## Result

The 1,997,737-parameter tokenwise MLP passed the 30-step overfit smoke test and trained for 1,000
steps on MPS in 76.9 seconds.

| Evaluation scope | One-step latent error | Persistence | Better than persistence | Mean-action ablation | Better with real action | Absolute action gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Multi, all 10 episodes | 0.01379 | 0.03276 | 57.9% | 0.01494 | 7.7% | 0.00115 |
| Red, all 10 episodes | 0.01368 | 0.03252 | 57.9% | 0.01491 | 8.3% | 0.00123 |
| Green variation, all 10 episodes | 0.01378 | 0.03372 | 59.1% | 0.01511 | 8.8% | 0.00133 |
| **Yellow plate, held-out test** | **0.02168** | **0.02951** | **26.5%** | **0.02300** | **5.7%** | **0.00132** |

The three-episode validation aggregate reached 0.01430 one-step error versus 0.03311 persistence
and 0.01550 under action ablation. The complete held-out member also measured:

| Held-out measurement | Value |
| --- | ---: |
| Rollout latent cosine error H5 | 0.11708 |
| Rollout latent cosine error H10 | 0.14153 |
| Ground-truth-latent decoder MAE | 0.06678 |
| Predicted-latent pixel MAE | 0.06792 |
| Raw calibrated-state MAE | 0.21882 |
| Inference latency | 0.397 ms / transition |

The earlier random-episode Project IRA run reported 60.3% improvement over persistence and 16.8%
improvement from action, but its test episodes came from a member also present in training. The new
26.5% and 5.7% source-held-out results show why that random split was insufficient. The comparison
is diagnostic rather than a controlled data ablation because the development mixture also changed.

Within the primary model's all-member diagnostic, the absolute action gap is similar across the
four members, roughly 0.0012–0.0013, while the held-out model error is 57% higher than the
development-member errors. Action signal survives the source shift, but source mismatch dominates
enough of the error budget that the relative action benefit falls below the architecture gate.

## Four-fold rotation

The three additional folds each trained for 1,000 MPS steps in approximately 77 seconds. They
reused 103,190,662 selected dataset bytes and the 880,164,532-byte visual cache. The rotation wrote
zero new dataset bytes and zero new feature-cache bytes; its 388 MB of new local artifacts are
checkpoints, metrics, previews, and Rerun recordings.

| Held-out source | Visual windows | Model error | Persistence | Better than persistence | Mean-action ablation | Better with real action | H5 | H10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Red | 6,911 | 0.01579 | 0.03252 | 51.4% | 0.01732 | 8.8% | 0.07398 | 0.09574 |
| Green variation | 9,114 | 0.01694 | 0.03372 | 49.7% | 0.02005 | 15.5% | 0.09335 | 0.12381 |
| Multi-color | 14,047 | 0.01653 | 0.03276 | 49.5% | 0.01828 | 9.6% | 0.08227 | 0.09905 |
| Yellow plate | 5,548 | 0.02168 | 0.02951 | 26.5% | 0.02300 | 5.7% | 0.11708 | 0.14153 |

Across 35,620 held-out visual windows, the window-weighted model error was 0.01729, improvement
over persistence was 46.4%, and improvement from real action was 10.4%. All four folds benefited
from action, so the predictor is not universally ignoring control. Yellow plate is the hardest
source by one-step and rollout error and has the weakest relative action benefit.

The actual sampler is uniform over eligible visual windows, not uniform over episodes or sources.
Because member lengths range from 5,578 to 14,077 frames, longer development sources receive more
training draws. Every fold used the same policy, so the rotation is internally comparable, but a
source-balanced sampler is the next controlled data-side change.

## Rerun and visual check

`evaluation.rrd` passed `rerun rrd verify`. It contains 291 chunks, 121 entity paths, held-out
actual/predicted/error images, the aggregate and per-member metric receipt, and 150 timeline-indexed
transform rows for each of the actual and predicted robots: five mapped joints over 30 steps.
For example, actual shoulder pan moves from -1.040 radians at step 0 to -0.093 at step 29, while
the predicted trajectory moves from -1.045 to -0.112. The gripper remains explicitly unmapped.

All three additional fold recordings also pass `rerun rrd verify`. The combined contact sheet
shows that red, green variation, and multi-color share a similar calibration-board composition,
while yellow plate includes a distinctly positioned, highly visible robot arm. The single yellow
result was therefore a real domain-shift warning, not evidence that every source generalized
equally poorly.

The preview is honest but coarse. It captures large color regions and motion trend while blurring
the arm and calibration-board detail. Decoder-oracle MAE is already 0.06678, close to the predicted
pixel MAE, so sharper images remain primarily a representation/decoder problem and should not be
mixed into the next source-generalization test.

## Reproduce

```bash
uv sync --extra train --extra lerobot --extra vision
uv run rwm train project-ira-so101-dinov2-source-held-out-poc \
  --run-dir runs/project-ira-so101-source-held-out-poc
uv run rwm evaluate source-rotation \
  project-ira-so101-dinov2-source-held-out-poc \
  --run-dir runs/project-ira-so101-source-held-out-poc
uv run rerun rrd verify runs/project-ira-so101-source-held-out-poc/evaluation.rrd
uv run rerun runs/project-ira-so101-source-held-out-poc/evaluation.rrd
```

## Decision

Keep this recipe and rotation command as the source-leakage regression, and keep the tokenwise MLP
as the control. Do not scale it to RunPod yet. The rotation distinguishes a yellow-plate-specific
domain shift from universal failure, but its 5.7–15.5% action benefit remains inconsistent.

The next worthwhile experiment is a source-balanced sampler using the same data, cache,
architecture, and four-fold assignment. If that does not narrow the fold spread, add another
plate-layout development member and hold out a different plate member. Both steps target the
observed domain-coverage problem before adding model capacity.
