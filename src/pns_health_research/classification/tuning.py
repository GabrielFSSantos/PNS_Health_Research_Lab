from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_classifier_yaml(model_name: str, configs_dir: Path | None = None) -> dict[str, Any]:
    if configs_dir is None:
        from pns_health_research.core.paths import CONFIGS_DIR

        configs_dir = CONFIGS_DIR
    path = configs_dir / "classifiers" / f"{model_name}.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_param_grid(model_name: str, configs_dir: Path | None = None) -> dict[str, Any]:
    data = load_classifier_yaml(model_name, configs_dir)
    return data.get("param_grid", {}) or {}


def load_model_n_jobs(model_name: str, configs_dir: Path | None = None) -> int | None:
    data = load_classifier_yaml(model_name, configs_dir)
    n_jobs = data.get("n_jobs")
    return int(n_jobs) if n_jobs is not None else None


def effective_n_jobs(
    model_name: str,
    *,
    parallel: bool = False,
    max_workers: int = 4,
) -> int:
    """CPU jobs per model; avoids max_workers × n_jobs=-1 contention."""
    yaml_jobs = load_model_n_jobs(model_name)
    if yaml_jobs is not None:
        return max(1, yaml_jobs)
    cpu_count = os.cpu_count() or 4
    if parallel:
        return max(1, cpu_count // max(1, max_workers))
    return cpu_count
