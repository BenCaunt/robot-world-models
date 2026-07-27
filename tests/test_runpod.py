from __future__ import annotations

import json

from robot_world_models.runpod import make_draft_plan


def test_plan_is_non_mutating_and_secret_free(monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "should-never-appear")

    plan = make_draft_plan(
        preferred_gpu="NVIDIA GeForce RTX 5090",
        max_hourly_usd=2.0,
        max_runtime_minutes=90,
    )
    serialized = json.dumps(plan.model_dump(mode="json"), sort_keys=True)

    assert plan.maximum_compute_usd == 3.0
    assert plan.mutation_allowed is False
    assert plan.api_key_source == "environment:RUNPOD_API_KEY"
    assert plan.plan_sha256
    assert "should-never-appear" not in serialized


def test_plan_hash_changes_with_budget() -> None:
    first = make_draft_plan(
        preferred_gpu="NVIDIA GeForce RTX 5090",
        max_hourly_usd=2.0,
        max_runtime_minutes=60,
    )
    second = make_draft_plan(
        preferred_gpu="NVIDIA GeForce RTX 5090",
        max_hourly_usd=2.0,
        max_runtime_minutes=61,
    )

    assert first.plan_sha256 != second.plan_sha256

