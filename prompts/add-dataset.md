# Prompt: add a dataset capability

Add support for a robot-learning dataset without creating one-off download or extraction code.

1. Search `catalog/catalog.json`.
2. Search `bencaunt/robot-datasets` through WarmHub and record its Dataset wref, profile,
   `DatasetModality` Arcs, `RecordedWith` evidence, Assessments, license, and upstream revision.
3. If the Dataset is absent from WarmHub, make the gap explicit. Create a provisional manifest, but
   do not present provisional fields as registry facts.
4. Compare its transport with existing source adapters and its layout with existing format adapters.
5. Preflight the exact upstream revision, access/gating state, required metadata, episode payload
   paths, and estimated bytes. Distinguish total repository bytes from the exact approved transfer.
   If access fails, return to discovery instead of bypassing WarmHub.
6. Prefer a dataset manifest using existing adapters.
7. Add a new source adapter only for a genuinely new protocol or authentication method.
8. Add a new format adapter only for a genuinely new storage or episode schema/version. Do not
   force an upstream conversion merely because the installed training library dropped an old format.
9. If the repository is a collection, declare its complete reviewed member set, exclusions, and
   selection evidence. Put the experiment's exact member roots and fail-closed byte ceiling in the
   recipe; do not write raw rows, videos, feature caches, or checkpoints to WarmHub.
10. Add the smallest license-safe fixture that exercises metadata, one episode boundary, timestamps,
   modalities, and error handling.
11. Add contract tests for canonical episode output, selective materialization, checksums, resume
    behavior, and malformed input.
12. Run:

    ```bash
    uv run rwm catalog validate
    uv run rwm catalog build
    uv run ruff check .
    uv run pytest
    ```

Prepare the change so it can be opened as a focused pull request. Do not commit dataset payloads,
credentials, checkpoints, or generated evaluation recordings.
