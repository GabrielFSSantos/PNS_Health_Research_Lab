"""Resolução de runtime device/backend auto para perfis."""

from __future__ import annotations

import os
from typing import Any, Literal

from pns_health_research.core.gpu import cuda_available, gpu_ready_for_training

ModuleName = Literal["preprocessing", "clustering", "classification"]


def _resolve_auto_workers(value: Any) -> int:
    if value in (None, "auto"):
        return max(1, min(4, os.cpu_count() or 1))
    return int(value)


def _resolve_device(value: Any) -> str:
    if value in (None, "auto"):
        return "cuda" if gpu_ready_for_training() else "cpu"
    if str(value) == "cuda" and not gpu_ready_for_training():
        return "cpu"
    return str(value)


def _resolve_clustering_backend(value: Any) -> str:
    if value in (None, "auto"):
        if cuda_available():
            try:
                import cuml  # noqa: F401

                return "cuml"
            except ImportError:
                pass
        return "sklearn"
    return str(value)


def _classification_gpu_available() -> bool:
    if not gpu_ready_for_training():
        return False
    try:
        import cuml  # noqa: F401

        return True
    except ImportError:
        return False


def _resolve_classification_backend(value: Any) -> str:
    if value in (None, "auto"):
        return "gpu" if _classification_gpu_available() else "sklearn"
    if str(value) == "gpu" and not _classification_gpu_available():
        return "sklearn"
    return str(value)


def resolve_runtime_options(
    runtime: dict[str, Any],
    *,
    module: ModuleName,
) -> dict[str, Any]:
    """Converte valores auto do perfil em opções concretas."""
    out: dict[str, Any] = {}
    device = runtime.get("device", "cpu")
    if module == "classification":
        out["device"] = _resolve_device(device)
        backend = runtime.get("classification_backend", runtime.get("backend", "sklearn"))
        out["classification_backend"] = _resolve_classification_backend(backend)
        out["parallel"] = bool(runtime.get("parallel", False))
        out["max_workers"] = _resolve_auto_workers(runtime.get("max_workers"))
    if module == "clustering":
        backend = runtime.get("clustering_backend", runtime.get("backend", "sklearn"))
        out["clustering_backend"] = _resolve_clustering_backend(backend)
    return out


def apply_runtime_to_experiment_config(cfg: Any, runtime: dict[str, Any], module: ModuleName) -> Any:
    """Aplica runtime resolvido a um ExperimentConfig já carregado."""
    resolved = resolve_runtime_options(runtime, module=module)
    if module == "classification":
        if resolved.get("device"):
            cfg.classification.device = resolved["device"]  # type: ignore[assignment]
        if resolved.get("parallel") is not None:
            cfg.classification.parallel = resolved["parallel"]
        if resolved.get("max_workers") is not None:
            cfg.classification.max_workers = resolved["max_workers"]
    if module == "clustering" and resolved.get("clustering_backend"):
        cfg.clustering.backend = resolved["clustering_backend"]  # type: ignore[assignment]
    return cfg
