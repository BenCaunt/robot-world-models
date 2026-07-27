# Prompt: scale an approved local run to RunPod

Scale an already evaluated local recipe to one RunPod GPU. Read `SECURITY.md`,
`docs/runpod-safety.md`, and the current official RunPod API documentation before implementing or
executing anything.

Preconditions:

- a local checkpoint and Rerun evaluation exist;
- the exact recipe and repository commit are known;
- I have supplied maximum hourly USD and maximum runtime;
- `RUNPOD_API_KEY` is set in my local environment, not pasted into chat;
- the repository's provisioning implementation and cleanup tests exist.

Workflow:

1. Run a dry plan without secrets or mutations.
2. With the environment key, query the live GPU catalog. Prefer a single RTX 5090 only when its live
   ID is available, compatible, and under my cap; otherwise show bounded alternatives.
3. Render GPU, live price, cloud type, runtime, maximum estimated compute cost, disks, storage costs,
   ports, image, command, artifact transfer, watchdog, and cleanup.
4. Show the exact plan hash and ask me for explicit approval. Do not infer approval from this prompt
   or from the presence of a key.
5. After approval, create one ephemeral secure-cloud Pod, record its ID immediately, and install a
   remote watchdog.
6. Run the pinned commit using `uv`. Stream sanitized progress and create resumable checkpoints.
7. Produce weights, config, metrics, logs, WarmHub snapshot, and `evaluation.rrd`.
8. Retrieve artifacts and verify checksums locally.
9. Terminate the Pod in `finally`, then verify it is terminated and that no unintended persistent
   volume remains.
10. Report actual cost, artifact paths, evaluation comparison, and cleanup evidence.

If the connection or training fails after Pod creation, cleanup takes priority. Do not leave the
resource merely stopped.

