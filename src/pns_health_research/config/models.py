from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ExperimentMeta(BaseModel):
    name: str
    seed: int = 42
    profile: Literal["legacy_reproduction", "research_v2"] = "legacy_reproduction"


class DataConfig(BaseModel):
    source: str = "data/raw/pns2019.parquet"
    record_id_columns: list[str] = Field(
        default_factory=lambda: ["V0001", "UPA_PNS", "V0006_PNS", "C00301"]
    )


class PreprocessingConfig(BaseModel):
    min_age: int = 18
    categorical_missing_value: int = -1
    binary_missing_value: int = 0
    legacy_weight_formula: bool = True
    merge_internacao_into_servico: bool = True
    target_encoding_before_split: bool = True  # legacy leakage
    nearmiss_before_split: bool = True
    nearmiss_version: int = 1
    impute_age_by_target: bool = True
    columns_config: str = "configs/data/pns2019_columns.yaml"
    binary_features_config: str = "configs/features/legacy_tcc_binary.yaml"
    categorical_features_config: str = "configs/features/legacy_tcc_categorical.yaml"


class ClusteringReducerConfig(BaseModel):
    name: Literal["pca"] = "pca"
    n_components: int = 8
    apply_scaler: bool = False
    scaler_type: Literal["none", "standard", "minmax"] = "none"


class ClusteringEstimatorConfig(BaseModel):
    name: Literal["kmeans"] = "kmeans"
    n_clusters: int = 6
    selection_method: Literal["fixed", "elbow", "kneedle", "silhouette"] = "fixed"
    init: str = "k-means++"
    n_init: int = 50
    max_iter: int = 1000
    elbow_n_init: int = 10
    elbow_max_k: int = 15
    random_state: int = 42


class ClusteringConfig(BaseModel):
    backend: Literal["sklearn", "cuml"] = "sklearn"
    device: Literal["cpu", "cuda"] = "cpu"
    comorbidity_columns: list[str] = Field(default_factory=list)
    reducer: ClusteringReducerConfig = Field(default_factory=ClusteringReducerConfig)
    estimator: ClusteringEstimatorConfig = Field(default_factory=ClusteringEstimatorConfig)


class ClassificationModelConfig(BaseModel):
    name: str
    enabled: bool = True


class ClassificationConfig(BaseModel):
    target: str = "healthcare_service_usage"
    legacy_cv: bool = True
    n_splits: int = 5
    inner_test_size: float = 1 / 3
    use_sample_weight_in_fit: bool = False
    parallel: bool = False
    max_workers: int = 4
    device: Literal["cpu", "cuda"] = "cpu"
    backend: Literal["sklearn", "gpu"] = "sklearn"
    input_run: str | None = None
    features_config: str = "configs/features/classification_default.yaml"
    feature_columns: list[str] | None = None
    exclude_columns: list[str] = Field(default_factory=list)
    target_encode_columns: list[str] = Field(default_factory=list)
    models: list[str] = Field(
        default_factory=lambda: ["random_forest", "svc", "mlp", "xgboost"]
    )


class BaselineConfig(BaseModel):
    enabled: bool = True
    tolerances_path: str = "configs/baseline/tolerances.yaml"


class ExperimentConfig(BaseModel):
    experiment: ExperimentMeta
    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    stages: list[str] = Field(
        default_factory=lambda: [
            "preprocessing",
            "clustering",
            "classification",
            "cross_analysis",
            "statistics",
        ]
    )


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)
