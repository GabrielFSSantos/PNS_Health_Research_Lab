"""Fixtures compartilhadas para regressão E2E (PNS real, uma passagem por perfil)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pns_health_research.config.profile_resolver import resolve_module_config
from pns_health_research.core.paths import PROJECT_ROOT
from pns_health_research.modules.clustering.pipeline_runner import run_clustering_pipeline
from pns_health_research.persistence import base_store
from pns_health_research.persistence.base_store import write_clustering_base, write_preprocessing_base
from pns_health_research.modules.preprocessing.pipelines import run_preprocessing

PROFILES = ("legacy", "pns")


@dataclass
class PreprocessArtifacts:
    result: Any
    config: Any


@dataclass
class ClusteringArtifacts:
    result: Any
    config: Any


def _require_real_pns() -> Path:
    raw = PROJECT_ROOT / "data/raw/pns2019.parquet"
    if not raw.exists():
        pytest.skip("PNS real ausente")
    return raw


@pytest.fixture(scope="session")
def regression_bases_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _require_real_pns()
    return tmp_path_factory.mktemp("regression_bases")


@pytest.fixture(scope="session")
def regression_bases_session(regression_bases_dir: Path):
    """Monta bases legacy e pns uma vez por sessão de regressão."""
    original = base_store.BASES_DIR
    base_store.BASES_DIR = regression_bases_dir
    cache: dict[str, dict[str, Any]] = {"preprocess": {}, "clustering": {}}

    for profile in PROFILES:
        pp_cfg = resolve_module_config(profile, "preprocessing")
        pp_result = run_preprocessing(pp_cfg)
        write_preprocessing_base(
            pp_result,
            pp_cfg,
            base_name=profile,
            profile_kind=profile,
        )
        cache["preprocess"][profile] = PreprocessArtifacts(result=pp_result, config=pp_cfg)

        cl_cfg = resolve_module_config(profile, "clustering")
        cl_result = run_clustering_pipeline(cl_cfg, base_name=profile, source_base=profile)
        write_clustering_base(
            base_name=profile,
            result=cl_result,
            config=cl_cfg,
            source_base=profile,
            profile_kind=profile,
        )
        cache["clustering"][profile] = ClusteringArtifacts(result=cl_result, config=cl_cfg)

    yield cache
    base_store.BASES_DIR = original


@pytest.fixture
def regression_bases(regression_bases_session, regression_bases_dir: Path, monkeypatch):
    monkeypatch.setattr(
        "pns_health_research.persistence.base_store.BASES_DIR",
        regression_bases_dir,
    )
    return regression_bases_session
