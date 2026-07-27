# Contributing

Contributions should leave a reusable capability behind instead of a one-off script.

## Dataset contribution

1. Search `catalog/catalog.json` and both WarmHub registries first.
2. Add or update a manifest in `catalog/datasets/`.
3. Reuse an existing `source.adapter` and `format.adapter` whenever possible.
4. Add code only for a genuinely new source protocol or data layout.
5. Add a bounded, redistributable fixture under `tests/fixtures/` when implementation code changes.
6. Run:

   ```bash
   uv run rwm catalog validate
   uv run rwm catalog build
   uv run ruff check .
   uv run pytest
   ```

Every manifest must retain WarmHub wrefs, upstream provenance, license, modality declarations, and
robot-link evidence. A provisional local mapping is allowed, but it must be labeled provisional and
must not masquerade as a registry fact.

## Recipe contribution

A recipe must declare:

- homogeneous or heterogeneous mixing;
- required and optional modalities;
- normalization, missing-modality, and sampling policies;
- dataset and robot manifests;
- local compute envelope;
- evaluation metrics;
- Rerun output;
- remote budget and teardown requirements if remote compute is supported.

## Pull requests

Explain the reusable capability added, list registry facts used, and attach a small evaluation
receipt. Do not commit datasets, model checkpoints, secrets, or `.rrd` files.

