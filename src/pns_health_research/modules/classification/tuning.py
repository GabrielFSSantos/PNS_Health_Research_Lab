from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pns_health_research.config.models import ExperimentConfig
from pns_health_research.core.paths import CONFIGS_DIR, PROJECT_ROOT


def _registry_entry(model_name: str):
    from pns_health_research.config.registry import ConfigurationError, load_models_registry

    try:
        registry = load_models_registry()
    except ConfigurationError:
        return None
    return registry.models.get(model_name)


def load_classifier_yaml(
    model_name: str,
    configs_dir: Path | None = None,
    *,
    config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    if config and model_name in config.model_registry:
        entry = config.model_registry[model_name]
        payload: dict[str, Any] = {}
        if entry.param_grid:
            payload["param_grid"] = entry.param_grid
        if entry.n_jobs is not None:
            payload["n_jobs"] = entry.n_jobs
        if payload:
            return payload
        if not entry.config_file:
            return {}

    path = _resolve_classifier_path(model_name, configs_dir, config=config)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _resolve_classifier_path(
    model_name: str,
    configs_dir: Path | None = None,
    *,
    config: ExperimentConfig | None = None,
) -> Path:
    if config and model_name in config.model_registry:
        rel = config.model_registry[model_name].config_file
        if rel:
            path = Path(rel)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            return path

    if configs_dir is None:
        configs_dir = CONFIGS_DIR
    return configs_dir / "classification" / f"{model_name}.yaml"


def load_param_grid(
    model_name: str,
    configs_dir: Path | None = None,
    *,
    config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    if config and model_name in config.model_registry:
        entry = config.model_registry[model_name]
        if not entry.gridsearch_enabled:
            return {}
        if entry.param_grid:
            return dict(entry.param_grid)
    entry = _registry_entry(model_name)
    if entry is not None:
        if not entry.gridsearch.enabled:
            return {}
        if entry.param_grid:
            return dict(entry.param_grid)
    data = load_classifier_yaml(model_name, configs_dir, config=config)
    return data.get("param_grid", {}) or {}


def load_model_n_jobs(
    model_name: str,
    configs_dir: Path | None = None,
    *,
    config: ExperimentConfig | None = None,
) -> int | None:
    if config and model_name in config.model_registry:
        n_jobs = config.model_registry[model_name].n_jobs
        if n_jobs is not None:
            return int(n_jobs)
    entry = _registry_entry(model_name)
    if entry is not None and entry.n_jobs is not None:
        return int(entry.n_jobs)
    data = load_classifier_yaml(model_name, configs_dir, config=config)
    n_jobs = data.get("n_jobs")
    return int(n_jobs) if n_jobs is not None else None


def effective_n_jobs(
    model_name: str,
    *,
    parallel: bool = False,
    max_workers: int = 4,
    config: ExperimentConfig | None = None,
) -> int:
    """CPU jobs per model; avoids max_workers × n_jobs=-1 contention."""
    yaml_jobs = load_model_n_jobs(model_name, config=config)
    if yaml_jobs is not None:
        return max(1, yaml_jobs)
    import os

    cpu_count = os.cpu_count() or 4
    if parallel:
        return max(1, cpu_count // max(1, max_workers))
    return cpu_count
