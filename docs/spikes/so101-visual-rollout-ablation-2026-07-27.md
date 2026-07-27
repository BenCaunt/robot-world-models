# SO-101 visual representation and rollout ablation — 2026-07-27

## Outcome

Two controlled experiments refined the first visual-latent recipe:

1. Increasing pooled DINOv2 spatial tokens from 4x4 to 8x8 did not improve this bounded model. It
   increased compressed feature storage by 2.5x, nearly doubled inference latency, and worsened
   ground-truth-latent decoder reconstruction by 15.1%.
2. Retaining 4x4 tokens and training the predictor with a discounted five-step open-loop
   latent/state loss reduced held-out rollout error by 10.7% at horizon five and 11.5% at horizon
   ten. It traded away 5.6% at horizon one and 13.5% on raw-unit state MAE.

The recommended local recipe now depends on intent:

- use `so101-dinov2-visual-h5-poc` for rollout or planning experiments;
- use `so101-dinov2-visual-poc` as the one-step fidelity control;
- keep `so101-dinov2-visual-8x8-poc` as a reproducible negative ablation, not the default.

## Reproduce

```bash
uv sync --extra train --extra lerobot --extra vision

uv run rwm train so101-dinov2-visual-poc \
  --run-dir runs/so101-visual-h1
uv run rwm train so101-dinov2-visual-8x8-poc \
  --run-dir runs/so101-visual-8x8-h1
uv run rwm train so101-dinov2-visual-h5-poc \
  --run-dir runs/so101-visual-h5

uv run rerun rrd verify runs/so101-visual-h1/evaluation.rrd
uv run rerun rrd verify runs/so101-visual-8x8-h1/evaluation.rrd
uv run rerun rrd verify runs/so101-visual-h5/evaluation.rrd
```

Each recipe uses the same WarmHub-resolved dataset, robot, camera, episode split, frozen encoder
revision, RGB target transform, and evaluation horizons. All runs use three train, one validation,
and one held-out test episode.

## Controlled matrix

| Measurement | 4x4, one-step | 8x8, one-step | 4x4, five-step |
| --- | ---: | ---: | ---: |
| Training rollout horizon | 1 | 1 | 5 |
| Rollout discount | 1.0 | 1.0 | 0.8 |
| Trainable parameters | 1,997,737 | 1,993,545 | 1,997,737 |
| Train windows | 3,058 | 3,058 | 3,046 |
| Feature cache | 120.3 MiB | 300.4 MiB | 120.3 MiB |
| Decoder reconstruction MAE | 0.1157 | 0.1332 | 0.1181 |
| Predicted RGB MAE | 0.1160 | 0.1333 | 0.1179 |
| One-step latent cosine error | 0.0569 | 0.1088 | 0.0600 |
| Persistence cosine error | 0.0684 | 0.1392 | 0.0684 |
| Mean-action ablation error | 0.0713 | 0.1314 | 0.0718 |
| Improvement over persistence | 16.9% | 21.9% | 12.3% |
| Improvement from action | 20.2% | 17.2% | 16.4% |
| Rollout error H1 | 0.0570 | 0.1090 | 0.0602 |
| Rollout error H5 | 0.2285 | 0.2915 | 0.2040 |
| Rollout error H10 | 0.2737 | 0.3450 | 0.2423 |
| State MAE, raw mixed units | 0.2769 | 0.2736 | 0.3142 |
| Inference per transition | 0.37 ms | 0.68 ms | 0.37 ms |

Raw latent cosine values in the 4x4 and 8x8 columns describe different pooled representations and
must not be compared as though they shared the same target space. The cross-grid decision instead
uses each model's improvement over its own persistence/action baselines together with
ground-truth-latent decoder reconstruction, RGB error, storage, latency, and visual inspection.

## What changed in the implementation

- Visual recipes now declare `training_rollout_horizon` and `rollout_loss_discount`.
- The contract rejects a visual recipe whose descriptive `intent.horizon_steps` differs from its
  executable training horizon.
- Episode window construction reserves every required future target without crossing episode
  boundaries.
- For the five-step objective, the first prediction receives real context and state. Subsequent
  steps receive the model's predicted latent context and predicted state, making the objective
  genuinely open loop.
- The loss records per-horizon latent terms and combines discounted latent/state error across the
  training horizon. Decoder and predicted-pixel supervision remain on the first step to isolate the
  dynamics objective.
- The RGB decoder uses bilinear resize plus 3x3 convolutions. The earlier
  transposed-convolution variant created visible banding, which could have made a representation
  experiment look worse for the wrong reason.

## Interpretation

The 8x8 result is a useful negative result, not evidence that spatial features are unhelpful. The
small decoder and predictor did not capitalize on four times as many tokens, while compressed NPZ
storage still grew 2.5x. A future full-grid experiment should first justify a decoder and model
capacity change; it should not be presented as a pure resolution comparison.

The five-step objective moved error in the intended direction at longer horizons without changing
the feature representation or model size. Its weaker one-step and state results are real tradeoffs.
The repository therefore keeps both recipes instead of replacing the one-step reference.

The decoded frames remain coarse. Pixel MAE is dominated by the representation/decoder oracle and
is not yet a task-success metric. Rerun is still required because it reveals spatial blur,
artifacts, rollout stability, and correspondence with the animated SO-101 state that scalar metrics
cannot establish.

## Next experiment

Add a task-relevant held-out metric before scaling compute. For the current cube-manipulation data,
the smallest useful addition is a frozen or human-reviewed object-region tracker that measures
whether predicted object position and robot end-effector proximity remain accurate by horizon.
Keep the 4x4 one-step and five-step recipes as controls.

Only after that metric distinguishes useful from merely smooth predictions should a paid GPU test
larger predictors, full 16x16 tokens, or a video-native encoder. Remote provisioning remains
plan-only until its teardown path is implemented and reviewed.
