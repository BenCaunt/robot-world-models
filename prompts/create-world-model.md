# Prompt: create a robot world model

Use the `robot-world-models` repository to help me design, train, and evaluate a world model for my
robot.

Follow `AGENTS.md` and `SPEC.md`. Work with me gate by gate; do not select paid compute or begin a
large download before I approve the data plan.

1. Ask for the robot, task, prediction target, horizon, available modalities, local hardware,
   license constraints, and success criteria.
2. Ask whether I expect a homogeneous or heterogeneous mixture. Explain the practical tradeoff using
   the candidate data you discover.
3. Run `uv run rwm warmhub discover "<query>"`. Query both `bencaunt/robot-models` and
   `bencaunt/robot-datasets`; traverse Arcs directionally and preserve `RecordedWith` match evidence.
4. Build a compatibility table of candidate datasets. Separate registry facts, upstream facts,
   inferences, and unknowns.
5. Recommend the smallest experiment that can test the idea. Show rejected datasets and why.
6. Before download, preflight the exact upstream revision and required files. Catalog presence is
   not proof that payload bytes are still public. After I approve the data plan, reuse existing
   source and format adapters. If an adapter is missing, follow `prompts/add-dataset.md` and leave a
   reusable contribution.
7. Train a bounded proof of concept locally with uv on MPS, CUDA, or CPU. Start with an overfit smoke
   test and compare against a naive baseline.
8. For visual models, isolate representation changes from dynamics-objective changes. Establish a
   decoder-oracle/image/storage baseline before increasing token resolution, then test multi-step
   training against the unchanged representation. Keep the declared and executable horizons equal.
9. Evaluate with Rerun. Produce an `.rrd` containing observations, predictions, metrics,
   provenance, and the WarmHub-resolved URDF. Animate validated mapping entries, display any
   explicitly unmapped joints, and prove motion by checking temporal `Transform3D` rows plus two
   visibly different timeline steps. A static URDF import is not a completed motion evaluation.
10. Review the local result with me. Only if remote compute is justified, follow
   `prompts/runpod-gpu.md`.
11. End with reproducible commands, artifact paths, caveats, and the modular contribution that future
    agents can reuse.

For visual recipes, use the exact encoder preprocessing output as the source for both cached
features and decoder RGB targets. Record resize, crop, orientation, normalization, model revision,
cache version, and per-episode frame counts. Compare latent prediction against both persistence and
a training-mean action ablation. Report ground-truth-latent decoder reconstruction separately from
predicted-latent pixel error, and inspect the generated actual/predicted preview before calling the
visual result useful. Do not compare raw latent cosine values across different pooling grids as if
they shared the same representation; use within-grid baselines and decoder/image metrics instead.
For an open-loop objective, feed predictions—not future ground truth—through the unrolled steps and
report the one-step tradeoff alongside longer-horizon gains.

For an implemented homogeneous recipe, prefer `uv run rwm train <recipe-id> --run-dir
runs/<run-id>` over recipe-specific scripts. Treat feature-to-URDF semantics as a dataset-to-robot
mapping capability and skip physical metrics while it remains provisional.
Validated partial mappings may animate only their covered joints; never infer a missing actuator
conversion just to make every mesh move.

Never ask me to paste an API key into chat. Never treat possession of a key as approval to spend.
