# Prompt: add a dataset capability

Add support for a robot-learning dataset without creating one-off download or extraction code.

1. Search `catalog/catalog.json`.
2. Search `bencaunt/robot-datasets` through WarmHub and record its Dataset wref, profile,
   `DatasetModality` Arcs, `RecordedWith` evidence, Assessments, license, and upstream revision.
3. If the Dataset is absent from WarmHub, make the gap explicit. Create a provisional manifest, but
   do not present provisional fields as registry facts.
4. Compare its transport with existing source adapters and its layout with existing format adapters.
5. Prefer a dataset manifest using existing adapters.
6. Add a new source adapter only for a genuinely new protocol or authentication method.
7. Add a new format adapter only for a genuinely new storage or episode schema.
8. Add the smallest license-safe fixture that exercises metadata, one episode boundary, timestamps,
   modalities, and error handling.
9. Add contract tests for canonical episode output, checksums, resume behavior, and malformed input.
10. Run:

    ```bash
    uv run rwm catalog validate
    uv run rwm catalog build
    uv run ruff check .
    uv run pytest
    ```

Prepare the change so it can be opened as a focused pull request. Do not commit dataset payloads,
credentials, checkpoints, or generated evaluation recordings.

