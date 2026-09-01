from __future__ import annotations

from collections.abc import Callable
from typing import Any

GPUClassifierFactory = Callable[[dict[str, Any]], Any]

GPU_CLASSIFIER_REGISTRY: dict[str, GPUClassifierFactory] = {}


def register_gpu_classifier(name: str) -> Callable[[GPUClassifierFactory], GPUClassifierFactory]:
    def decorator(fn: GPUClassifierFactory) -> GPUClassifierFactory:
        GPU_CLASSIFIER_REGISTRY[name] = fn
        return fn

    return decorator
