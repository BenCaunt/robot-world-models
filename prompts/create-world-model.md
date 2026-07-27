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
6. After I approve the data plan, reuse existing source and format adapters. If an adapter is
   missing, follow `prompts/add-dataset.md` and leave a reusable contribution.
7. Train a bounded proof of concept locally with uv on MPS, CUDA, or CPU. Start with an overfit smoke
   test and compare against a naive baseline.
8. Evaluate with Rerun. Produce an `.rrd` containing observations, predictions, metrics,
   provenance, and the WarmHub-resolved URDF when the joint mapping is validated.
9. Review the local result with me. Only if remote compute is justified, follow
   `prompts/runpod-gpu.md`.
10. End with reproducible commands, artifact paths, caveats, and the modular contribution that future
    agents can reuse.

Never ask me to paste an API key into chat. Never treat possession of a key as approval to spend.

