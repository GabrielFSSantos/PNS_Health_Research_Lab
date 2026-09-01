from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ExperimentMeta(BaseModel):
    name: str
    seed: int = 42
    profile: str = "custom"


class RunMeta(BaseModel):
    """Metadados amigáveis da run (não substituem run_id)."""

    name: str | None = None
    description: str | None = None
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    preserve: bool = False
    type: Literal["experiment", "clustering", "classification", "preprocessing"] | None = None


class DataConfig(BaseModel):
    dataset: str = "pns_2019"
    year: int = 2019
    source: str | None = None
    record_id_columns: list[str] = Field(
        default_factory=lambda: ["V0001", "UPA_PNS", "V0006_PNS", "C00301"]
    )
    download_if_missing: bool = True


class StageToggle(BaseModel):
    enabled: bool = True


class AdultFilterStage(BaseModel):
    enabled: bool = True
    min_age: int = 18


class ImputeMissingStage(BaseModel):
    enabled: bool = True
    binary: int = 0
    categorical: int = -1


class NearmissStage(BaseModel):
    enabled: bool = False
    version: int = 1


class PreprocessingStages(BaseModel):
    select_columns: StageToggle = Field(default_factory=StageToggle)
    adult_filter: AdultFilterStage = Field(default_factory=AdultFilterStage)
    derive_features: StageToggle = Field(default_factory=StageToggle)
    merge_internacao_into_servico: StageToggle = Field(default_factory=StageToggle)
    one_hot_and_numeric: StageToggle = Field(default_factory=StageToggle)
    impute_missing: ImputeMissingStage = Field(default_factory=ImputeMissingStage)
    impute_age_by_target: StageToggle = Field(default_factory=lambda: StageToggle(enabled=False))
    nearmiss: NearmissStage = Field(default_factory=NearmissStage)
    target_encoding_before_split: StageToggle = Field(
        default_factory=lambda: StageToggle(enabled=False)
    )


class WeightsConfig(BaseModel):
    legacy_formula: bool = False


class ClassificationFeaturesConfig(BaseModel):
    target_column: str = "Serviço de Saúde"
    weight_column: str = "Peso Amostral"
    exclude_columns: list[str] = Field(
        default_factory=lambda: [
            "Serviço de Saúde",
            "Peso Amostral",
            "Cluster",
            "survey_weight",
            "ml_sample_weight",
        ]
    )
    target_encode_columns: list[str] = Field(
        default_factory=lambda: ["Zona", "Sexo", "Raça", "Estado Civil", "Cluster"]
    )


class PreprocessingConfig(BaseModel):
    stages: PreprocessingStages = Field(default_factory=PreprocessingStages)
    weights: WeightsConfig = Field(default_factory=WeightsConfig)
    derived: dict[str, StageToggle] = Field(
        default_factory=lambda: {
            "imc": StageToggle(enabled=True),
            "faixas_idade": StageToggle(enabled=True),
            "grupos_comorbidade": StageToggle(enabled=True),
        }
    )
    classification_features: ClassificationFeaturesConfig = Field(
        default_factory=ClassificationFeaturesConfig
    )
    columns_map: dict[str, str] = Field(default_factory=dict)

    @property
    def min_age(self) -> int:
        return self.stages.adult_filter.min_age

    @property
    def categorical_missing_value(self) -> int:
        return self.stages.impute_missing.categorical

    @property
    def binary_missing_value(self) -> int:
        return self.stages.impute_missing.binary

    @property
    def legacy_weight_formula(self) -> bool:
        return self.weights.legacy_formula

    @property
    def merge_internacao_into_servico(self) -> bool:
        return self.stages.merge_internacao_into_servico.enabled

    @property
    def target_encoding_before_split(self) -> bool:
        return self.stages.target_encoding_before_split.enabled

    @property
    def nearmiss_before_split(self) -> bool:
        return self.stages.nearmiss.enabled

    @property
    def nearmiss_version(self) -> int:
        return self.stages.nearmiss.version

    @property
    def impute_age_by_target(self) -> bool:
        return self.stages.impute_age_by_target.enabled

    def derived_enabled(self, key: str) -> bool:
        if not self.stages.derive_features.enabled:
            return False
        spec = self.derived.get(key)
        if spec is None:
            return True
        return spec.enabled


class ClusteringReducerConfig(BaseModel):
    name: str = "pca"
    enabled: bool = True
    n_components: int = 8
    apply_scaler: bool = False
    scaler_type: Literal["none", "standard", "minmax"] = "none"


class ClusteringEstimatorConfig(BaseModel):
    name: str = "kmeans"
    enabled: bool = True
    n_clusters: int = 6
    selection_method: Literal["fixed", "elbow", "kneedle", "silhouette"] = "fixed"
    init: str = "k-means++"
    n_init: int = 50
    max_iter: int = 1000
    elbow_n_init: int = 10
    elbow_max_k: int = 15
    random_state: int = 42


class ClusteringConfig(BaseModel):
    enabled: bool = True
    backend: Literal["sklearn", "cuml"] = "sklearn"
    device: Literal["cpu", "cuda"] = "cpu"
    comorbidity_columns: list[str] = Field(default_factory=list)
    reducer: ClusteringReducerConfig = Field(default_factory=ClusteringReducerConfig)
    estimator: ClusteringEstimatorConfig = Field(default_factory=ClusteringEstimatorConfig)


class ConfigurationFiles(BaseModel):
    models: str | None = "configs/models.yaml"
    clustering: str | None = "configs/clustering.yaml"
    datasets: str | None = "configs/datasets.yaml"
    preprocessing: str | None = "configs/preprocessing.yaml"


class ModelRegistryEntryConfig(BaseModel):
    param_grid: dict[str, Any] = Field(default_factory=dict)
    n_jobs: int | None = None
    gridsearch_enabled: bool = True
    gridsearch_scoring: str = "f1_macro"
    display_name: str = ""
    enabled: bool = True
    order: int = 999
    config_file: str | None = None


class ClassificationConfig(BaseModel):
    target: str = "healthcare_service_usage"
    legacy_cv: bool = False
    n_splits: int = 5
    inner_test_size: float = 1 / 3
    use_sample_weight_in_fit: bool = True
    parallel: bool = False
    max_workers: int = 4
    device: Literal["cpu", "cuda"] = "cpu"
    backend: Literal["sklearn", "gpu"] = "sklearn"
    input_run: str | None = None
    features_config: str | None = None
    feature_columns: list[str] | None = None
    exclude_columns: list[str] = Field(default_factory=list)
    target_encode_columns: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)


class EvaluationConfig(BaseModel):
    primary_metric: str = "f1_positive"


class BaselineConfig(BaseModel):
    enabled: bool = False
    tolerances: dict[str, Any] = Field(default_factory=dict)
    tolerances_path: str | None = None


class ExperimentConfig(BaseModel):
    experiment: ExperimentMeta
    run: RunMeta = Field(default_factory=RunMeta)
    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    configuration_files: ConfigurationFiles | None = None
    model_registry: dict[str, ModelRegistryEntryConfig] = Field(default_factory=dict)
    stages: list[str] = Field(
        default_factory=lambda: [
            "preprocessing",
            "clustering",
            "classification",
            "cross_analysis",
            "statistics",
        ]
    )


def _stages_mapping_to_list(raw_stages: Any) -> list[str]:
    if isinstance(raw_stages, list):
        return [str(s) for s in raw_stages]
    if not isinstance(raw_stages, dict):
        return [
            "preprocessing",
            "clustering",
            "classification",
            "cross_analysis",
            "statistics",
        ]
    order = [
        "preprocessing",
        "clustering",
        "classification",
        "cross_analysis",
        "statistics",
    ]
    enabled: list[str] = []
    for name in order:
        spec = raw_stages.get(name)
        if spec is False:
            continue
        if spec is True or spec is None:
            if name in raw_stages:
                enabled.append(name)
            continue
        if isinstance(spec, dict) and spec.get("enabled", True):
            enabled.append(name)
    extra = [k for k in raw_stages if k not in order]
    for name in extra:
        spec = raw_stages[name]
        if spec is True or (isinstance(spec, dict) and spec.get("enabled", True)):
            enabled.append(name)
    return enabled


def load_config(
    path: str | Path,
    *,
    cli_models: list[str] | None = None,
    cli_dataset: str | None = None,
) -> ExperimentConfig:
    from pns_health_research.config.registry import (
        ConfigurationError,
        build_model_registry_map,
        clustering_registry_to_config,
        deep_merge,
        load_clustering_registry,
        load_datasets_registry,
        load_models_registry,
        load_preprocessing_registry,
        resolve_enabled_models,
        resolve_path,
    )

    config_path = Path(path).resolve()
    with open(config_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    extends = raw.pop("extends", None)
    if extends:
        base_path = resolve_path(extends)
        with open(base_path, encoding="utf-8") as f:
            base: dict[str, Any] = yaml.safe_load(f) or {}
        raw = deep_merge(base, raw)

    cfg_files = raw.get("configuration_files") or {}
    has_registry = bool(cfg_files) or "configuration_files" in raw

    if has_registry:
        models_registry_path = cfg_files.get("models", "configs/models.yaml")
        if models_registry_path is not False:
            models_registry = load_models_registry(models_registry_path)
            raw["model_registry"] = build_model_registry_map(models_registry)

            classification = raw.get("classification") or {}
            explicit_models = classification.get("models")
            if not explicit_models:
                classification["models"] = resolve_enabled_models(
                    models_registry,
                    cli_models=cli_models,
                )
            else:
                classification["models"] = resolve_enabled_models(
                    models_registry,
                    explicit_models=list(explicit_models),
                    cli_models=cli_models,
                )
            raw["classification"] = classification

        clustering_registry_path = cfg_files.get("clustering", "configs/clustering.yaml")
        if clustering_registry_path is not False:
            clustering_raw = load_clustering_registry(clustering_registry_path)
            registry_clustering = clustering_registry_to_config(clustering_raw)
            experiment_clustering = raw.get("clustering") or {}
            raw["clustering"] = deep_merge(registry_clustering, experiment_clustering)

        datasets_path = cfg_files.get("datasets", "configs/datasets.yaml")
        if datasets_path is not False:
            datasets = load_datasets_registry(datasets_path)
            data = raw.get("data") or {}
            dataset_key = cli_dataset or data.get("dataset") or datasets.default_key()
            edition = datasets.get(dataset_key)
            if edition.status != "supported":
                raise ConfigurationError(
                    edition.message or f"Dataset {dataset_key!r} não é supported"
                )
            data["dataset"] = dataset_key
            data["year"] = edition.year
            if not data.get("record_id_columns"):
                data["record_id_columns"] = edition.record_id_columns
            if "download_if_missing" not in data:
                data["download_if_missing"] = datasets.defaults.get("download", {}).get(
                    "if_missing", True
                )
            raw["data"] = data

            preprocessing = raw.get("preprocessing") or {}
            preprocessing["columns_map"] = edition.columns
            raw["preprocessing"] = preprocessing

        preprocessing_path = cfg_files.get("preprocessing", "configs/preprocessing.yaml")
        if preprocessing_path is not False:
            pp_registry = load_preprocessing_registry(preprocessing_path)
            experiment_pp = raw.get("preprocessing") or {}
            raw["preprocessing"] = deep_merge(pp_registry, experiment_pp)

        raw["configuration_files"] = cfg_files
    elif cli_models:
        classification = raw.get("classification") or {}
        models = list(classification.get("models") or [])
        filtered = [m for m in models if m in cli_models] if models else list(cli_models)
        classification["models"] = filtered
        raw["classification"] = classification

    if "stages" in raw:
        raw["stages"] = _stages_mapping_to_list(raw["stages"])
    clustering_cfg = raw.get("clustering") or {}
    if clustering_cfg.get("enabled") is False:
        raw["stages"] = [s for s in (raw.get("stages") or []) if s != "clustering"]

    return ExperimentConfig.model_validate(raw)
