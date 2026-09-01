"""Carregamento e resolução de registries YAML (datasets, preprocessing, models, clustering)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pns_health_research.core.paths import CONFIGS_DIR, PROJECT_ROOT

SUPPORTED_MODELS_SCHEMA = "2.0"
SUPPORTED_CLUSTERING_SCHEMA = "1.0"
SUPPORTED_DATASETS_SCHEMA = "2.0"
SUPPORTED_PREPROCESSING_SCHEMA = "2.0"
IMPLEMENTED_REDUCERS = frozenset({"pca"})
IMPLEMENTED_ESTIMATORS = frozenset({"kmeans"})


class ConfigurationError(ValueError):
    """Erro de leitura ou validação de registry YAML."""


@dataclass(frozen=True)
class GridsearchConfig:
    enabled: bool = True
    scoring: str = "f1_macro"


@dataclass(frozen=True)
class ModelRegistryEntry:
    key: str
    enabled: bool
    order: int
    display_name: str
    param_grid: dict[str, Any]
    n_jobs: int | None
    gridsearch: GridsearchConfig
    config_file: Path | None = None


@dataclass(frozen=True)
class ModelsRegistry:
    schema_version: str
    defaults: dict[str, Any]
    models: dict[str, ModelRegistryEntry]

    def enabled_models(self) -> list[ModelRegistryEntry]:
        return sorted(
            (m for m in self.models.values() if m.enabled),
            key=lambda m: (m.order, m.key),
        )


@dataclass(frozen=True)
class DatasetEntry:
    key: str
    enabled: bool
    order: int
    year: int
    display_name: str
    status: str
    raw_path: str
    survey_weight_column: str
    record_id_columns: list[str]
    columns: dict[str, str]
    download_backend: str | None
    message: str | None = None
    download: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetsRegistry:
    schema_version: str
    defaults: dict[str, Any]
    datasets: dict[str, DatasetEntry]

    def default_key(self) -> str:
        enabled = sorted(
            (
                d
                for d in self.datasets.values()
                if d.enabled and d.status == "supported"
            ),
            key=lambda d: (d.order, d.key),
        )
        if not enabled:
            raise ConfigurationError("Nenhum dataset enabled e supported em datasets.yaml")
        return enabled[0].key

    def get(self, key: str) -> DatasetEntry:
        if key not in self.datasets:
            raise ConfigurationError(
                f"Dataset {key!r} não encontrado. Disponíveis: {sorted(self.datasets)}"
            )
        return self.datasets[key]

    def by_year(self, year: int) -> DatasetEntry:
        matches = [
            d
            for d in self.datasets.values()
            if d.year == year and d.status == "supported"
        ]
        if not matches:
            years = sorted({d.year for d in self.datasets.values()})
            raise ConfigurationError(
                f"Ano {year} não está no catálogo (ou não é supported). Edições: {years}"
            )
        return sorted(matches, key=lambda d: d.order)[0]


def resolve_path(path: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Arquivo de configuração não encontrado: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"YAML inválido (esperado mapping): {path}")
    return data


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{field} deve ser booleano, recebido: {value!r}")
    return value


def load_models_registry(path: str | Path | None = None) -> ModelsRegistry:
    from pns_health_research.classification.registry import CLASSIFIER_REGISTRY

    if path is None:
        path = CONFIGS_DIR / "models.yaml"
    resolved = resolve_path(path)
    raw = _load_yaml(resolved)

    schema = str(raw.get("schema_version", ""))
    if schema != SUPPORTED_MODELS_SCHEMA:
        raise ConfigurationError(
            f"schema_version em {resolved} deve ser {SUPPORTED_MODELS_SCHEMA!r}, "
            f"recebido {schema!r}"
        )

    defaults = raw.get("defaults") or {}
    default_gs = defaults.get("gridsearch") or {}
    models_raw = raw.get("models") or {}
    if not models_raw:
        raise ConfigurationError(f"Nenhum modelo definido em {resolved}")

    entries: dict[str, ModelRegistryEntry] = {}
    for key, spec in models_raw.items():
        if not isinstance(spec, dict):
            raise ConfigurationError(f"models.{key} deve ser um mapping")
        if key not in CLASSIFIER_REGISTRY:
            if spec.get("enabled", False):
                raise ConfigurationError(
                    f"models.{key}.enabled=true, mas '{key}' não está implementado. "
                    f"Disponíveis: {sorted(CLASSIFIER_REGISTRY)}"
                )
            continue

        gs_spec = spec.get("gridsearch") or default_gs
        config_file = spec.get("config_file")
        entries[key] = ModelRegistryEntry(
            key=key,
            enabled=_require_bool(spec.get("enabled", False), f"models.{key}.enabled"),
            order=int(spec.get("order", 999)),
            display_name=str(spec.get("display_name", key)),
            param_grid=dict(spec.get("param_grid") or {}),
            n_jobs=int(spec["n_jobs"]) if spec.get("n_jobs") is not None else None,
            config_file=resolve_path(config_file) if config_file else None,
            gridsearch=GridsearchConfig(
                enabled=_require_bool(
                    gs_spec.get("enabled", True),
                    f"models.{key}.gridsearch.enabled",
                ),
                scoring=str(gs_spec.get("scoring", default_gs.get("scoring", "f1_macro"))),
            ),
        )

    return ModelsRegistry(schema_version=schema, defaults=defaults, models=entries)


def load_datasets_registry(path: str | Path | None = None) -> DatasetsRegistry:
    if path is None:
        path = CONFIGS_DIR / "datasets.yaml"
    resolved = resolve_path(path)
    raw = _load_yaml(resolved)

    schema = str(raw.get("schema_version", ""))
    if schema != SUPPORTED_DATASETS_SCHEMA:
        raise ConfigurationError(
            f"schema_version em {resolved} deve ser {SUPPORTED_DATASETS_SCHEMA!r}, "
            f"recebido {schema!r}"
        )

    defaults = raw.get("defaults") or {}
    default_backend = (defaults.get("download") or {}).get("backend")
    datasets_raw = raw.get("datasets") or {}
    if not datasets_raw:
        raise ConfigurationError(f"Nenhum dataset definido em {resolved}")

    entries: dict[str, DatasetEntry] = {}
    for key, spec in datasets_raw.items():
        if not isinstance(spec, dict):
            raise ConfigurationError(f"datasets.{key} deve ser um mapping")
        download = dict(spec.get("download") or {})
        backend = download.get("backend", default_backend)
        entries[key] = DatasetEntry(
            key=key,
            enabled=bool(spec.get("enabled", False)),
            order=int(spec.get("order", 999)),
            year=int(spec["year"]),
            display_name=str(spec.get("display_name", key)),
            status=str(spec.get("status", "supported")),
            raw_path=str(spec.get("raw_path", f"data/raw/{key}.parquet")),
            survey_weight_column=str(spec.get("survey_weight_column", "V00291")),
            record_id_columns=list(spec.get("record_id_columns") or []),
            columns=dict(spec.get("columns") or {}),
            download_backend=str(backend) if backend else None,
            message=spec.get("message"),
            download=download,
        )

    return DatasetsRegistry(schema_version=schema, defaults=defaults, datasets=entries)


def load_preprocessing_registry(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = CONFIGS_DIR / "preprocessing.yaml"
    resolved = resolve_path(path)
    raw = _load_yaml(resolved)

    schema = str(raw.get("schema_version", ""))
    if schema != SUPPORTED_PREPROCESSING_SCHEMA:
        raise ConfigurationError(
            f"schema_version em {resolved} deve ser {SUPPORTED_PREPROCESSING_SCHEMA!r}, "
            f"recebido {schema!r}"
        )

    raw.pop("schema_version", None)
    return raw


def load_clustering_registry(path: str | Path | None = None) -> dict[str, Any]:
    from pns_health_research.clustering.registry import CLUSTERER_REGISTRY

    if path is None:
        path = CONFIGS_DIR / "clustering.yaml"
    resolved = resolve_path(path)
    raw = _load_yaml(resolved)

    schema = str(raw.get("schema_version", ""))
    if schema != SUPPORTED_CLUSTERING_SCHEMA:
        raise ConfigurationError(
            f"schema_version em {resolved} deve ser {SUPPORTED_CLUSTERING_SCHEMA!r}, "
            f"recebido {schema!r}"
        )

    if not _require_bool(raw.get("enabled", True), "clustering.enabled"):
        return {"enabled": False}

    reducer = raw.get("reducer") or {}
    estimator = raw.get("estimator") or {}

    reducer_name = str(reducer.get("name", "pca"))
    estimator_name = str(estimator.get("name", "kmeans"))

    if reducer.get("enabled", True) and reducer_name not in IMPLEMENTED_REDUCERS:
        raise ConfigurationError(
            f"reducer.name={reducer_name!r} não implementado. "
            f"Disponíveis: {sorted(IMPLEMENTED_REDUCERS)}"
        )
    if estimator.get("enabled", True) and estimator_name not in CLUSTERER_REGISTRY:
        raise ConfigurationError(
            f"estimator.name={estimator_name!r} não implementado. "
            f"Disponíveis: {sorted(CLUSTERER_REGISTRY)}"
        )

    return raw


def resolve_enabled_models(
    registry: ModelsRegistry,
    *,
    explicit_models: list[str] | None = None,
    cli_models: list[str] | None = None,
) -> list[str]:
    """Retorna lista final de modelos ordenada (enabled ∩ explicit ∩ cli)."""
    enabled_keys = {m.key for m in registry.enabled_models()}

    if explicit_models:
        candidates = [m for m in explicit_models if m in enabled_keys]
        unknown = [m for m in explicit_models if m not in registry.models]
        if unknown:
            raise ConfigurationError(
                f"Modelos desconhecidos no experimento: {unknown}. "
                f"Registrados: {sorted(registry.models)}"
            )
        disabled = [m for m in explicit_models if m in registry.models and m not in enabled_keys]
        if disabled:
            raise ConfigurationError(
                f"Modelos listados no experimento mas disabled no registry: {disabled}"
            )
    else:
        candidates = [m.key for m in registry.enabled_models()]

    if cli_models:
        cli_set = set(cli_models)
        unknown_cli = cli_set - set(registry.models)
        if unknown_cli:
            raise ConfigurationError(
                f"Modelos desconhecidos via --models: {sorted(unknown_cli)}"
            )
        candidates = [m for m in candidates if m in cli_set]

    if not candidates:
        raise ConfigurationError("Nenhum classificador ativo após resolução do registry")

    order = {m.key: m.order for m in registry.enabled_models()}
    return sorted(candidates, key=lambda k: (order.get(k, 999), k))


def clustering_registry_to_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Converte registry de clustering para dict compatível com ClusteringConfig."""
    reducer = dict(raw.get("reducer") or {})
    estimator = dict(raw.get("estimator") or {})

    return {
        "enabled": raw.get("enabled", True),
        "backend": raw.get("backend", "sklearn"),
        "device": raw.get("device", "cpu"),
        "comorbidity_columns": raw.get("comorbidity_columns", []),
        "reducer": reducer,
        "estimator": estimator,
    }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def build_model_registry_map(registry: ModelsRegistry) -> dict[str, dict[str, Any]]:
    """Mapa serializável para ExperimentConfig.model_registry."""

    def _rel(path: Path | None) -> str | None:
        if path is None:
            return None
        if path.is_relative_to(PROJECT_ROOT):
            return str(path.relative_to(PROJECT_ROOT))
        return str(path)

    return {
        key: {
            "param_grid": entry.param_grid,
            "n_jobs": entry.n_jobs,
            "config_file": _rel(entry.config_file),
            "gridsearch_enabled": entry.gridsearch.enabled,
            "gridsearch_scoring": entry.gridsearch.scoring,
            "display_name": entry.display_name,
            "enabled": entry.enabled,
            "order": entry.order,
        }
        for key, entry in registry.models.items()
    }
