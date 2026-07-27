# SO-101 visual-latent spike — 2026-07-27

## Outcome

The first one-camera visual world-model recipe completed end to end on Apple MPS. It resolved the
SO-101 dataset and URDF through both WarmHub registries, selectively downloaded the laptop-camera
AV1 streams, enforced exact row-to-frame alignment, cached frozen DINOv2-S spatial features, trained
an action-conditioned latent predictor and small RGB decoder, and wrote actual/predicted images
beside animated actual/predicted robots in Rerun.

The result is a genuine visual-latent dynamics proof: held-out predictions beat both latent
persistence and a mean-action ablation. It is not yet a sharp video generator or long-horizon
planning model.

## Reproduce

```bash
uv sync --extra train --extra lerobot --extra vision
uv run rwm train so101-dinov2-visual-poc --run-dir runs/so101-visual
uv run rerun runs/so101-visual/evaluation.rrd
```

The command writes configuration, discovery, source preflight, data and feature-cache receipts,
episode split, normalization, smoke test, checkpoints, metrics, a PNG rollout preview, the pinned
robot receipt, and `evaluation.rrd`. `runs/` is gitignored.

## Reviewed inputs and model

- Dataset: `Dataset/huggingface/nashmo/so101@v1`
- Upstream dataset commit: `7e8ba026d3040061561c9fb15976f33a4bc1d275`
- Camera: `observation.images.laptop`, 640x480 AV1 at 30 FPS
- Robot: `Robot/the-robot-studio/so-arm101@v1`
- Upstream robot commit: `63eede5a636e548eb8f2854e558bd343c21db9f7`
- Encoder: official `facebook/dinov2-small`, Apache-2.0, pinned to
  `ed25f3a31f01632728cabb09d1542f84ab7b0056`
- WarmHub encoder status: exact `dinov2` match absent in both registries; explicit registry gap
- Encoder view: 224x224; 16x16 patch grid pooled to 4x4; 384 dimensions per token
- Context: three visual frames plus current normalized six-dimensional state and action
- Outputs: next 16x384 latent grid, next state, and a learned 64x64 RGB decode
- Trainable parameters: 1,997,737; DINOv2 remains frozen and outside the checkpoint
- Decoder: bilinear resize followed by 3x3 convolutions; no transposed-convolution upsampling

The cache stores float16 L2-normalized tokens and uint8 RGB. RGB is derived by denormalizing the
exact DINOv2 processor tensor and then resizing it, so the decoder target has the same crop and
orientation as the encoder view.

## Observed result

| Measurement | Result |
| --- | ---: |
| Device | MPS, PyTorch 2.11.0 |
| Split | 3 train / 1 validation / 1 test episodes |
| Visual windows | 3,058 train / 1,026 validation / 1,024 test |
| Smoke test | 0.3946 → 0.2187 total loss in 30 steps |
| Full training | 1,000 steps in about 76 seconds |
| One-step latent cosine error | 0.0569 |
| Persistence latent error | 0.0684 |
| Improvement over persistence | 16.9% |
| Training-mean action ablation error | 0.0713 |
| Improvement from action | 20.2% |
| Rollout latent error, horizon 1 / 5 / 10 | 0.0570 / 0.2285 / 0.2737 |
| Predicted RGB MAE | 0.1160 on normalized RGB |
| Ground-truth-latent decoder MAE | 0.1157 on normalized RGB |
| State MAE | 0.2769 raw mixed dataset units |
| Rerun | 30-step open-loop RGB/error streams plus animated actual/predicted robots |

## What worked

- The modular payload contract selected one camera and five episodes without downloading the second
  camera.
- PyAV decoded the upstream AV1 streams on macOS; every selected video frame count exactly matched
  its LeRobot state/action rows.
- Frozen feature extraction and all trainable optimization ran on MPS.
- A 4x4 pooled spatial cache retained enough signal for the action-conditioned predictor to beat
  both required baselines while keeping each episode cache around 24 MB.
- The action ablation is materially worse than the real-action model, evidence that the predictor
  is not merely copying recent visual context.
- Rerun combines actual, predicted, and absolute-error images with the validated five-joint robot
  animation. The PNG preview makes visual failure visible without opening the full recording.

## What did not work well

1. The first implementation cached a full-frame square resize as the decoder target while DINOv2
   encoded a center crop. Latent metrics did not reveal the mismatch. Visual inspection did. Cache
   version 3 now derives the RGB target from the exact processor tensor and fingerprints the
   encoder, revision, input size, pool grid, and RGB output size before reuse.
2. Correcting preprocessing and replacing transposed-convolution upsampling with resize-convolution
   reduced held-out pixel MAE to 0.1160 and removed visible banding, but the result remains blurry.
   Decoder reconstruction from the ground-truth latent is itself 0.1157, showing that the 4x4
   bottleneck and lightweight decoder still dominate pixel fidelity.
3. Ten-step latent error is about five times one-step error. More one-step optimization did not
   solve compounding rollout drift; the next dynamics experiment needs an unrolled multi-step
   objective.
4. The visual recipe uses one fixed camera and one task. It does not establish viewpoint robustness,
   object-centric accuracy, task success, contact prediction, or transfer to a different SO-101
   dataset.
5. DINOv2 is not currently a first-class record in either WarmHub registry. The pinned official
   fallback is explicit and reproducible, but registry sourcing is incomplete.

## Spec refinements from the spike

- Treat encoder preprocessing as part of the cache schema, not an incidental library detail.
- Version feature caches and reject stale frame counts.
- Require a training-mean action ablation in addition to persistence.
- Separate decoder reconstruction error from predicted-latent pixel error.
- Generate a compact actual/predicted preview as well as the Rerun recording.
- Keep video materialization in the format adapter, feature extraction in a reusable cache layer,
  and model training independent of both.
- Do not justify a paid GPU merely because the model is visual: this full local iteration trains in
  under a minute after caching.

## Follow-up

The controlled representation/rollout ablation is complete:

1. 8x8 pooling was rejected because it increased cache size and worsened decoder reconstruction;
2. a five-step open-loop objective improved held-out horizon-5 and horizon-10 errors while modestly
   worsening one-step and state metrics;
3. persistence, action ablation, and ground-truth-latent decoder baselines were retained.

See
[`so101-visual-rollout-ablation-2026-07-27.md`](so101-visual-rollout-ablation-2026-07-27.md).
The next visual experiment should add an object- or task-relevant metric before using pixel fidelity
as a planning proxy.
