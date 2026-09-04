"""Resolve perfis unificados (configs/profiles/) em ExperimentConfig por módulo."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml

from pns_health_research.config.loader import load_config
from pns_health_research.config.registry import deep_merge
from pns_health_research.core.paths import CONFIGS_DIR, PROJECT_ROOT

ModuleName = Literal["preprocessing", "clustering", "classification"]

_ALL_STAGES = (
    "preprocessing",
    "clustering",
    "classification",
    "cross_analysis",
    "statistics",
)


def profile_path(profile: str) -> Path:
    return CONFIGS_DIR / "profiles" / f"{profile.strip().lower()}.yaml"


def module_entry_path(profile: str, module: ModuleName) -> str:
    return f"configs/profiles/{profile.strip().lower()}.yaml#{module}"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _is_override_document(doc: dict[str, Any]) -> bool:
    return "extends_profile" in doc and "profile" not in doc


def load_profile_document(
    profile: str,
    *,
    override_path: Path | str | None = None,
) -> dict[str, Any]:
    path = profile_path(profile)
    if not path.is_file():
        raise FileNotFoundError(f"Perfil não encontrado: {path}")
    doc = _load_yaml(path)

    if override_path is not None:
        override = _load_yaml(Path(override_path))
        if _is_override_document(override):
            base_profile = str(override.pop("extends_profile", profile))
            base_doc = load_profile_document(base_profile)
            doc = deep_merge(base_doc, override)
        else:
            doc = deep_merge(doc, override)
    return doc


def _stage_flags(active_module: ModuleName) -> dict[str, bool]:
    return {name: name == active_module for name in _ALL_STAGES}


def _configuration_files(module: ModuleName, mod: dict[str, Any]) -> dict[str, Any]:
    datasets = "configs/shared/datasets.yaml"
    if module == "preprocessing":
        return {
            "datasets": datasets,
            "preprocessing": mod["pipeline"],
            "clustering": False,
            "models": False,
        }
    if module == "clustering":
        return {
            "datasets": datasets,
            "preprocessing": False,
            "clustering": mod["pipeline"],
            "models": False,
        }
    return {
        "datasets": datasets,
        "preprocessing": False,
        "clustering": False,
        "models": "configs/shared/models.yaml",
    }


def _apply_runtime_to_raw(raw: dict[str, Any], runtime: dict[str, Any], module: ModuleName) -> None:
    from pns_health_research.config.runtime import resolve_runtime_options

    runtime_merged = dict(runtime)
    if module == "classification":
        clf = raw.get("classification") or {}
        if clf.get("backend") in (None, "auto"):
            runtime_merged.setdefault("classification_backend", "auto")
        elif "backend" in clf:
            runtime_merged["classification_backend"] = clf["backend"]
        if clf.get("device") in (None, "auto"):
            runtime_merged.setdefault("device", "auto")
        elif "device" in clf:
            runtime_merged["device"] = clf["device"]

    resolved = resolve_runtime_options(runtime_merged, module=module)
    if module == "clustering" and resolved.get("clustering_backend"):
        clustering = raw.setdefault("clustering", {})
        clustering["backend"] = resolved["clustering_backend"]
    if module == "classification":
        clf = raw.setdefault("classification", {})
        if resolved.get("device"):
            clf["device"] = resolved["device"]
        if resolved.get("classification_backend"):
            clf["backend"] = resolved["classification_backend"]
        elif clf.get("backend") in (None, "auto"):
            clf["backend"] = resolved.get("classification_backend", "sklearn")
        if resolved.get("parallel") is not None:
            clf["parallel"] = resolved["parallel"]
        if resolved.get("max_workers") is not None:
            clf["max_workers"] = resolved["max_workers"]


def build_module_raw(
    profile: str,
    module: ModuleName,
    *,
    override_path: Path | str | None = None,
) -> dict[str, Any]:
    doc = load_profile_document(profile, override_path=override_path)
    mod = dict((doc.get("modules") or {}).get(module) or {})
    runtime = dict(doc.get("runtime") or {})

    raw: dict[str, Any] = {
        "schema_version": "2.0",
        "experiment": {
            "name": mod.pop("experiment_name", f"{profile}_{module}"),
            "profile": mod.pop("methodological_profile", doc.get("profile", profile)),
        },
        "configuration_files": _configuration_files(module, mod),
        "stages": _stage_flags(module),
        "baseline": {"enabled": False},
    }

    if module == "preprocessing":
        if mod.pop("download_if_missing", True):
            raw["data"] = {"download_if_missing": True}
    elif module == "clustering":
        clustering_overlay = {
            k: v
            for k, v in mod.items()
            if k != "pipeline"
        }
        if clustering_overlay:
            raw["clustering"] = clustering_overlay
    elif module == "classification":
        clf_overlay = {k: v for k, v in mod.items() if k not in ("pipeline", "primary_metric")}
        if clf_overlay:
            raw["classification"] = clf_overlay
        metric = mod.get("primary_metric")
        if metric:
            raw["evaluation"] = {"primary_metric": metric}

    _apply_runtime_to_raw(raw, runtime, module)
    return raw


def resolve_module_config(
    profile: str,
    module: ModuleName,
    *,
    override_path: Path | str | None = None,
    cli_models: list[str] | None = None,
    cli_dataset: str | None = None,
):
    """Carrega ExperimentConfig para um módulo do perfil unificado."""
    cache_dir = CONFIGS_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{profile.strip().lower()}_{module}"
    if override_path is not None:
        cache_key += f"_{Path(override_path).stem}"
    cache_file = cache_dir / f"{cache_key}.yaml"

    raw = build_module_raw(profile, module, override_path=override_path)
    raw["extends"] = "configs/_internal/schema_root.yaml"

    with open(cache_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)

    rel = cache_file.relative_to(PROJECT_ROOT) if cache_file.is_relative_to(PROJECT_ROOT) else cache_file
    return load_config(rel, cli_models=cli_models, cli_dataset=cli_dataset)


def resolve_config_for_run(
    *,
    profile: str | None,
    module: ModuleName,
    config: Path | str,
    override_is_explicit: bool = False,
) -> tuple[Any, str]:
    """Retorna (ExperimentConfig, entry_config_path para manifest)."""
    config_path = Path(config)
    if profile:
        override = config_path if override_is_explicit and _path_is_override(config_path) else None
        cfg = resolve_module_config(profile, module, override_path=override)
        return cfg, module_entry_path(profile, module)

    cfg = load_config(config_path)
    return cfg, str(config_path)


def _path_is_override(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return _is_override_document(_load_yaml(path))
    except OSError:
        return False
