from __future__ import annotations

from typing import Any


def device_report() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "torchInstalled": False,
            "selected": None,
            "available": [],
            "next": "uv sync --extra train",
        }

    available: list[str] = []
    if torch.backends.mps.is_available():
        available.append("mps")
    if torch.cuda.is_available():
        available.append("cuda")
    available.append("cpu")
    return {
        "torchInstalled": True,
        "torchVersion": torch.__version__,
        "available": available,
        "selected": available[0],
    }


def select_device(preference: list[str]) -> str:
    report = device_report()
    available = set(report["available"])
    for device in preference:
        if device in available:
            return device
    raise RuntimeError(f"none of the requested devices are available: {preference}")

