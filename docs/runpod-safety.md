# RunPod safety contract

The first implementation target is an ephemeral, single-GPU training job—not a persistent
development machine.

## Secret entry

Ask the user to set a restricted key in their own shell:

```bash
read -s RUNPOD_API_KEY
export RUNPOD_API_KEY
```

Do not accept a token flag. Do not echo the variable. Do not write it to `.env` automatically.

## Required state machine

```text
draft -> live-priced -> user-approved -> creating -> running
      -> artifacts-retrieved -> checksums-verified -> terminating -> terminated
```

Any failure after `creating` enters cleanup. Cleanup is idempotent.

## Required plan fields

- live GPU ID, name, availability, and hourly price;
- cloud type;
- GPU count;
- image and CUDA compatibility;
- maximum runtime and maximum estimated compute cost;
- disk sizes and persistent-storage cost;
- ports and public IP behavior;
- repository commit and locked command;
- source and artifact transfer;
- remote watchdog deadline;
- local cleanup command.

An RTX 5090 may be preferred, but code must resolve its current catalog ID and price at planning
time. If unavailable or over budget, return alternatives and wait for the user.

## Defaults

- one GPU;
- secure cloud;
- no persistent volume;
- no public service ports;
- no notebook server;
- explicit hourly and runtime caps;
- checkpoint before deadline;
- terminate after verified retrieval.

RunPod documents that stopped Pods can still incur volume-storage charges and network-volume Pods
cannot be stopped. The implementation therefore verifies termination and separately verifies that
it did not create an unintended persistent volume.

## Implementation acceptance tests

- dry-run requires no API key and performs no writes;
- live plan can list GPU types without exposing the key;
- plan serialization contains no secret material;
- create is impossible without a fresh approval token tied to the plan hash;
- Pod ID is persisted before polling;
- timeout invokes cleanup;
- training failure invokes cleanup;
- artifact failure still invokes cleanup while preserving the Pod ID for recovery reporting;
- repeated cleanup is safe;
- terminal verification detects a still-running or stopped-but-billable resource.

Current status: contract only. Paid provisioning is intentionally absent from v0.1.

