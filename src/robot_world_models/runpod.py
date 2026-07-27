from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RunPodDraftPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["draft-live-catalog-required"] = "draft-live-catalog-required"
    provider: Literal["runpod"] = "runpod"
    preferred_gpu: str
    gpu_count: Literal[1] = 1
    cloud_type: Literal["secure"] = "secure"
    max_hourly_usd: float = Field(gt=0)
    max_runtime_minutes: int = Field(gt=0)
    maximum_compute_usd: float = Field(gt=0)
    persistent_volume: Literal[False] = False
    public_service_ports: Literal[False] = False
    api_key_source: Literal["environment:RUNPOD_API_KEY"] = "environment:RUNPOD_API_KEY"
    mutation_allowed: Literal[False] = False
    required_before_create: list[str]
    plan_sha256: str = ""


def make_draft_plan(
    *,
    preferred_gpu: str,
    max_hourly_usd: float,
    max_runtime_minutes: int,
) -> RunPodDraftPlan:
    plan = RunPodDraftPlan(
        preferred_gpu=preferred_gpu,
        max_hourly_usd=max_hourly_usd,
        max_runtime_minutes=max_runtime_minutes,
        maximum_compute_usd=max_hourly_usd * max_runtime_minutes / 60,
        required_before_create=[
            "query live GPU catalog and resolve ID, price, availability, and CUDA compatibility",
            "render image, disks, ports, command, artifact transfer, watchdog, and cleanup",
            "bind explicit user approval to the final plan hash",
            "pass provisioning and idempotent teardown tests",
        ],
    )
    canonical = json.dumps(
        plan.model_dump(exclude={"plan_sha256"}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return plan.model_copy(update={"plan_sha256": hashlib.sha256(canonical).hexdigest()})

