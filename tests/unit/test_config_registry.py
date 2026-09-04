"""Testes do registry de configuração modular."""

from __future__ import annotations

import pytest

from pns_health_research.config.loader import load_config
from pns_health_research.config.registry import (
    ConfigurationError,
    load_models_registry,
    resolve_enabled_models,
)
from pns_health_research.core.paths import PROJECT_ROOT


def test_load_models_registry_enabled_only_implemented():
    registry = load_models_registry(PROJECT_ROOT / "configs/shared/models.yaml")
    enabled = [m.key for m in registry.enabled_models()]
    assert "random_forest" in enabled
    assert "xgboost" in enabled
    assert "svc" not in enabled


def test_models_registry_inline_param_grid_no_config_file():
    registry = load_models_registry(PROJECT_ROOT / "configs/shared/models.yaml")
    rf = registry.models["random_forest"]
    assert rf.config_file is None
    assert "n_estimators" in rf.param_grid
    assert rf.n_jobs == 2


def test_resolve_enabled_models_explicit_intersection():
    registry = load_models_registry(PROJECT_ROOT / "configs/shared/models.yaml")
    models = resolve_enabled_models(
        registry,
        explicit_models=["random_forest", "xgboost", "mlp"],
    )
    assert models == ["random_forest", "xgboost", "mlp"]


def test_resolve_enabled_models_rejects_disabled():
    registry = load_models_registry(PROJECT_ROOT / "configs/shared/models.yaml")
    with pytest.raises(ConfigurationError, match="disabled"):
        resolve_enabled_models(registry, explicit_models=["svc"])


def test_load_config_with_configuration_files_resolves_models():
    cfg = load_config(PROJECT_ROOT / "configs/_internal/schema_root.yaml")
    assert cfg.classification.models == ["random_forest", "xgboost", "mlp"]
    assert "random_forest" in cfg.model_registry
    assert cfg.model_registry["random_forest"].config_file is None
    assert cfg.model_registry["random_forest"].param_grid["n_estimators"] == [50, 100]


def test_load_config_cli_models_filter():
    cfg = load_config(
        PROJECT_ROOT / "configs/_internal/schema_root.yaml",
        cli_models=["xgboost"],
    )
    assert cfg.classification.models == ["xgboost"]


def test_load_config_cli_dataset():
    cfg = load_config(
        PROJECT_ROOT / "configs/_internal/schema_root.yaml",
        cli_dataset="pns_2019",
    )
    assert cfg.data.dataset == "pns_2019"
    assert cfg.data.year == 2019
    assert cfg.preprocessing.columns_map["J014"] == "Serviço de Saúde"


def test_load_legacy_config_with_registry():
    cfg = load_config(PROJECT_ROOT / "configs/_internal/tcc_overlay.yaml")
    assert cfg.experiment.profile == "legacy_reproduction"
    assert cfg.classification.models == ["random_forest", "xgboost", "mlp"]
    assert cfg.clustering.reducer.apply_scaler is False
    assert cfg.baseline.tolerances["preprocessing"]["rows_out"] == 87678


def test_load_param_grid_respects_gridsearch_disabled():
    from pns_health_research.modules.classification.tuning import load_param_grid
    from pns_health_research.config.models import (
        ClassificationConfig,
        ExperimentConfig,
        ExperimentMeta,
        ModelRegistryEntryConfig,
    )

    cfg = ExperimentConfig(
        experiment=ExperimentMeta(name="grid_off"),
        model_registry={
            "xgboost": ModelRegistryEntryConfig(
                param_grid={"n_estimators": [50, 100]},
                gridsearch_enabled=False,
            )
        },
        classification=ClassificationConfig(models=["xgboost"]),
    )
    assert load_param_grid("xgboost", config=cfg) == {}


def test_load_param_grid_inline():
    from pns_health_research.modules.classification.tuning import load_param_grid

    cfg = load_config(PROJECT_ROOT / "configs/_internal/schema_root.yaml")
    grid = load_param_grid("random_forest", config=cfg)
    assert "n_estimators" in grid
    assert "max_depth" in grid
