"""Testes do pipeline modular schema 3.0."""

from __future__ import annotations

import pytest

from pns_health_research.config.profile_resolver import resolve_module_config
from pns_health_research.config.models import PipelineStage, PreprocessingConfig
from pns_health_research.core.paths import PROJECT_ROOT
from pns_health_research.modules.preprocessing.pipeline_adapter import validate_pipeline
from pns_health_research.modules.preprocessing.pipelines import run_preprocessing


def test_pipeline_dependency_validation():
    pp = PreprocessingConfig(
        schema_version="3.0",
        pipeline=[
            PipelineStage(id="select_columns", enabled=True),
            PipelineStage(id="adult_filter", enabled=True),
            PipelineStage(id="encode_binary", enabled=True),
            PipelineStage(id="gestational_reinforce", enabled=True),
            PipelineStage(id="attach_weights", enabled=False),
            PipelineStage(id="drop_design_columns", enabled=True),
        ],
    )
    with pytest.raises(ValueError, match="drop_design_columns"):
        validate_pipeline(pp.resolved_pipeline())


def test_valid_preprocessing_pipeline_stages_recorded(synthetic_raw_path):
    cfg = resolve_module_config("pns", "preprocessing")
    result = run_preprocessing(cfg, raw_path=str(synthetic_raw_path))

    executed = result.processing_stats.get("pipeline_executed", [])
    assert executed
    assert executed[0]["id"] == "select_columns"
    assert "servico_j014" in result.categorical_df.columns
    assert "servico_composto" in result.categorical_df.columns
    stage_ids = [s["id"] for s in executed]
    assert "derive_binary" in stage_ids
    assert "encode_categorical" in stage_ids


def test_derive_binary_disabled_skips_imc_columns():
    import pandas as pd

    from pns_health_research.modules.preprocessing.derived import apply_derived_binary

    df = pd.DataFrame({"Peso": [70.0], "Altura": [170.0], "Idade": [30]})
    pp = PreprocessingConfig(
        schema_version="3.0",
        pipeline=[PipelineStage(id="derive_binary", enabled=False)],
    )
    pp.stages.derive_features.enabled = False
    out = apply_derived_binary(df, config=pp)
    assert "Baixo Peso" not in out.columns
    assert "Peso" in out.columns
