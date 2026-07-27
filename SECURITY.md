# Security and cost safety

## Secrets

- Never paste `WH_TOKEN`, `RUNPOD_API_KEY`, Hugging Face tokens, SSH private keys, or cloud
  credentials into an agent conversation.
- Never pass secret values on command lines, where they can enter shell history or process listings.
- Read secrets from environment variables or a platform secret manager.
- Never serialize, print, test-snapshot, or include secret values in a generated plan.
- Use restricted, least-privilege, short-lived keys whenever the provider supports them.

## RunPod

Paid infrastructure creation is an important action and always requires explicit user approval of
the rendered plan. The plan must show:

- live GPU type and hourly price;
- GPU count;
- secure/community cloud choice;
- maximum runtime and estimated maximum compute cost;
- disk and persistent-storage choices;
- exposed ports;
- artifact retrieval path;
- stop and termination behavior.

Defaults are one GPU, secure cloud, no public service ports, no persistent volume, and termination
after artifacts have been verified locally. A stopped Pod can continue to incur storage charges, so
the normal terminal state is terminated, not merely stopped.

Provisioning code must use `try/finally`, install a remote watchdog, record the Pod ID immediately,
and provide an idempotent cleanup command. Never choose a GPU by display name alone or hard-code a
price; query the live catalog and enforce the user's cap.

## Data and models

Registry discovery does not grant redistribution rights. Respect every upstream license and use
restriction. Do not upload private robot data or checkpoints to public services without explicit
approval.

