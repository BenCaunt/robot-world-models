# SO-101 spatiotemporal-transformer ablation — 2026-07-27

## Outcome

A compact spatiotemporal transformer now runs through the same visual-cache, training, rollout,
decoder, Rerun, and URDF evaluation pipeline as the tokenwise MLP reference. The implementation
attends across three frames of 4x4 DINOv2 tokens plus state and action, then uses learned output
queries to predict the next spatial token grid.

The corrected one-step transformer beat latent persistence by 14.0%, but replacing every action
with the training-mean action worsened its error by only 0.61%. The smaller MLP control achieved a
20.2% action gap. The transformer is therefore retained as an executable experimental recipe, not
promoted as the default. On this five-episode homogeneous subset it mostly learned passive visual
continuity rather than action-conditioned dynamics.

## Reproduce

```bash
uv sync --extra train --extra lerobot --extra vision
uv run rwm train so101-dinov2-transformer-poc \
  --run-dir runs/so101-transformer-h1
uv run rerun rrd verify runs/so101-transformer-h1/evaluation.rrd
uv run rerun runs/so101-transformer-h1/evaluation.rrd
```

The five-step contract is available as `so101-dinov2-transformer-h5-poc`, but it was not rerun
after the corrected H1 experiment failed the action-sensitivity gate. This avoids treating more
compute as a substitute for evidence.

## Architecture

```text
3 frames x 16 DINOv2 tokens
  -> per-token projection + learned time/space positions
normalized state + action
  -> condition tokens
all 50 tokens
  -> 2-layer TransformerEncoder
16 learned spatial queries
  -> 2-layer TransformerDecoder
  -> residual next-token grid + residual next state
  -> shared resize-convolution RGB decoder
```

The model uses width 256, eight attention heads, no dropout, and 4,504,873 trainable parameters.
The MLP and transformer share the frozen encoder revision, 4x4 cache, decoder implementation,
camera, data split, loss terms, evaluation horizons, and Rerun path.

## Controlled result

| Measurement | MLP H1 control | Transformer H1 |
| --- | ---: | ---: |
| Trainable parameters | 1,997,737 | 4,504,873 |
| Learning rate | 0.001 | 0.0001 |
| MPS training time, 1,000 steps | 75.8 s | 84.3 s |
| One-step latent cosine error | 0.0569 | 0.0588 |
| Persistence cosine error | 0.0684 | 0.0684 |
| Improvement over persistence | 16.9% | 14.0% |
| Mean-action ablation error | 0.0713 | 0.0592 |
| Improvement from action | 20.2% | 0.61% |
| Rollout error H1 | 0.0570 | 0.0590 |
| Rollout error H5 | 0.2285 | 0.1914 |
| Rollout error H10 | 0.2737 | 0.2273 |
| Decoder reconstruction MAE | 0.1157 | 0.1335 |
| Predicted RGB MAE | 0.1160 | 0.1228 |
| State MAE, raw mixed units | 0.2769 | 0.3742 |
| Inference per transition | 0.37 ms | 0.46 ms |

The transformer's lower long-horizon latent error is interesting, but it is not sufficient for
promotion because the action ablation shows that the predictor barely uses the control input. A
passive-continuity model can look stable in open loop while being unsuitable for planning.

The final Rerun recording verified without error and contains 1,788 rows across 121 entity paths.
It includes timeline-indexed actual and predicted robot transforms, actual/predicted/error images,
joint traces, metrics, and provenance. Visual inspection shows coherent but very blurry decoded
frames, consistent with the weak 4x4 decoder oracle.

## Initialization failure and correction

The first implementation randomly initialized the final feature-delta head. That destroyed the
residual model's useful persistence prior: initial latent error was about 0.91 while persistence on
the held-out split was about 0.07. Although the smoke loss fell, the one-step model finished 5.1%
worse than persistence and used action only weakly.

The corrected model zero-initializes the final feature-delta and state-delta layers, so its first
prediction is normalized persistence and unchanged state. At the MLP learning rate of 0.001, the
corrected model failed the smoke test (0.2910 to 0.4156 total loss). A bounded sweep on the same
cached batch found:

| Learning rate | 30-step total loss | 30-step latent error | Decision |
| --- | ---: | ---: | --- |
| 0.0003 | 0.2910 → 0.2541 | 0.0485 → 0.0492 | reject: latent term worsened |
| 0.0001 | 0.2910 → 0.2646 | 0.0485 → 0.0468 | use |
| 0.00003 | 0.2910 → 0.2880 | 0.0485 → 0.0466 | stable but too slow |

The transformer recipes record `0.0001` explicitly. This means the final comparison changes
architecture and its necessary optimizer setting; it is not presented as a perfectly
single-variable ablation.

An initial five-step transformer run with the flawed random residual head reached one-step error
0.2066—202% worse than persistence—and was discarded as a diagnostic failure. It was not used to
judge the corrected architecture.

## Implementation and test coverage

- Visual recipes select their model through the declared implementation contract.
- Transformer recipes must declare a positive attention-head count and a divisible hidden width.
- Shape tests cover forward prediction and decoder output.
- A spatial-coupling test proves that changing one input token can affect another output token,
  which distinguishes the transformer from the tokenwise MLP.
- A residual-initialization test proves that a fresh transformer starts at normalized persistence
  with unchanged state.
- Unsupported implementations fail explicitly instead of silently falling back to the MLP.

## Decision and next experiment

Keep `so101-dinov2-visual-poc` and `so101-dinov2-visual-h5-poc` as the recommended controls. Keep
the transformer recipes as reproducible architecture experiments.

Do not scale this transformer to RunPod yet. The next useful experiment is data-side: add more
action-diverse SO-101 episodes or a second modality-compatible SO-101 dataset, retain
episode/source-aware splits, and rerun H1. Promote the architecture to H5 only if it both beats
persistence and develops a material action-ablation gap. A task-relevant object or end-effector
metric remains the other prerequisite before larger video-native models.
