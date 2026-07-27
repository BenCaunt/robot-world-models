# Architecture

WarmHub is the control plane; upstream hosts remain the payload plane.

```mermaid
flowchart LR
    U["User model intent"] --> D["WarmHub discovery"]
    RM["bencaunt/robot-models"] --> D
    RD["bencaunt/robot-datasets"] --> D
    D --> P["Approved data plan"]
    P --> C["Dataset and robot manifests"]
    C --> S["Source adapters"]
    S --> F["Format adapters"]
    F --> E["Canonical episodes"]
    E --> T["Local POC training"]
    T --> R["Rerun evaluation"]
    RM --> R
    R --> A{"Remote run useful?"}
    A -- "No" --> PR["Reusable contribution"]
    A -- "Yes, approved" --> RP["Cost-bounded RunPod job"]
    RP --> R
    R --> PR
```

## Boundaries

| Layer | Owns | Does not own |
| --- | --- | --- |
| WarmHub discovery | identity, relationships, evidence, provenance, URLs | bulk payload bytes |
| Dataset manifest | declarative mapping and compatibility | transport implementation |
| Source adapter | download, authentication, resume, checksum | dataset schema |
| Format adapter | source layout to canonical episodes | recipe-specific training |
| Recipe | mixture, modalities, model and evaluation contract | one-off extraction logic |
| Compute adapter | device or remote lifecycle | model semantics |
| Rerun evaluator | inspectable inputs, predictions, metrics, robot geometry | training decisions |

## Initial package boundaries

```text
robot_world_models.contracts  manifest types and validation
robot_world_models.catalog    deterministic catalog discovery and generation
robot_world_models.warmhub    read-only typed wrapper around the wh CLI
robot_world_models.devices    MPS/CUDA/CPU selection
robot_world_models.runpod     plan and policy only in v0.1
robot_world_models.models     small reference models
robot_world_models.eval       mandatory Rerun receipts
```

Adapters will be introduced behind protocols after the first bounded SO-101 data slice proves their
required interface.

