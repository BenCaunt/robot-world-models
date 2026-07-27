# Architecture

WarmHub is the control plane; upstream hosts remain the payload plane.

```mermaid
flowchart LR
    U["User model intent"] --> D["WarmHub discovery"]
    RM["bencaunt/robot-models"] --> D
    RD["bencaunt/robot-datasets"] --> D
    D --> P["Approved data plan"]
    P --> C["Dataset, robot, and mapping manifests"]
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
| Dataset-robot mapping | feature/joint names, affine transforms, units, coverage, evidence, validation status | global robot identity |
| Source adapter | download, authentication, resume, checksum | dataset schema |
| Format adapter | source layout, payload selection, and canonical episodes | recipe-specific training |
| Visual cache | video decoding, exact frame alignment, encoder transform, reusable features | source transport or model training |
| Recipe | mixture, modalities, model and evaluation contract | one-off extraction logic |
| Compute adapter | device or remote lifecycle | model semantics |
| Rerun evaluator | inspectable inputs, predictions, metrics, robot geometry | training decisions |

## Initial package boundaries

```text
robot_world_models.contracts  manifest types and validation
robot_world_models.catalog    deterministic catalog discovery and generation
robot_world_models.warmhub    read-only typed wrapper around the wh CLI
robot_world_models.adapters   versioned source and format adapters
robot_world_models.devices    MPS/CUDA/CPU selection
robot_world_models.runpod     plan and policy only in v0.1
robot_world_models.models     small reference models
robot_world_models.eval       mandatory Rerun receipts
robot_world_models.training   receipt-producing local recipe runner
robot_world_models.visual_data  video decoding and frozen-feature cache contract
robot_world_models.visual_training  visual recipe composition, training, and rollout evaluation
```

The first bounded SO-101 spike established two reusable boundaries: source adapters selectively
materialize immutable upstream paths, while format adapters own storage-version parsing and
canonical episode validation. LeRobot library compatibility is not used as a proxy for dataset
storage compatibility.

The visual spike adds a third boundary: a visual-feature cache owns the relationship between one
decoded camera frame, the frozen encoder's exact processed view, its spatial latent tokens, and the
RGB reconstruction target. The dynamics model consumes that cache without importing a source or
format adapter. This lets a newly contributed dataset reuse visual training as soon as its adapter
can provide aligned camera paths and canonical episodes.
