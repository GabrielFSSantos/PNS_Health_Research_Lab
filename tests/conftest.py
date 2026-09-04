from __future__ import annotations

from pathlib import Path

import pytest

from pns_health_research.config.models import ExperimentConfig
from tests.fixtures.synthetic_pns import SYNTHETIC_PNS_ROWS, make_synthetic_pns


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "regression: E2E PNS real (lento)")
    config.addinivalue_line(
        "markers",
        "regression_smoke: valida artefatos materializados ou load_config (rápido)",
    )


def apply_fast_classification(cfg: ExperimentConfig) -> ExperimentConfig:
    """Reduz classificação para testes rápidos (1 modelo, 2 folds, sem gridsearch)."""
    cfg.classification.models = ["random_forest"]
    cfg.classification.n_splits = 2
    cfg.classification.parallel = False
    cfg.classification.max_workers = 1
    cfg.classification.legacy_cv = True
    cfg.classification.use_sample_weight_in_fit = False
    cfg.preprocessing.stages.nearmiss.enabled = False
    for entry in cfg.model_registry.values():
        entry.gridsearch_enabled = False
        entry.param_grid = {}
    return cfg


def apply_fast_regression(cfg: ExperimentConfig) -> ExperimentConfig:
    """Reduz classificação para regressão E2E (1 modelo, sem gridsearch, sem paralelo)."""
    cfg.classification.models = ["random_forest"]
    cfg.classification.n_splits = 2
    cfg.classification.parallel = False
    cfg.classification.max_workers = 1
    cfg.preprocessing.stages.nearmiss.enabled = False
    for entry in cfg.model_registry.values():
        entry.gridsearch_enabled = False
        entry.param_grid = {}
    return cfg


@pytest.fixture
def synthetic_raw_df():
    return make_synthetic_pns(SYNTHETIC_PNS_ROWS)


@pytest.fixture
def synthetic_raw_path(tmp_path: Path, synthetic_raw_df):
    path = tmp_path / "raw.parquet"
    synthetic_raw_df.to_parquet(path)
    return path


@pytest.fixture
def monkeypatch_bases_dir(tmp_path: Path, monkeypatch):
    bases = tmp_path / "bases"
    monkeypatch.setattr(
        "pns_health_research.persistence.base_store.BASES_DIR",
        bases,
    )
    return bases
