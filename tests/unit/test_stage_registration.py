"""Testes da API de registro de estágios."""

from __future__ import annotations

import pytest

from pns_health_research.modules.preprocessing.contract import PreprocessingContext
from pns_health_research.modules.preprocessing.stages.registry import (
    STAGE_REGISTRY,
    register_stage,
    run_registered_stages,
)
from pns_health_research.config.models import PipelineStage, PreprocessingConfig


def test_builtin_stages_registered():
    expected = {
        "select_columns",
        "adult_filter",
        "encode_binary",
        "gestational_reinforce",
        "attach_weights",
        "drop_design_columns",
        "derive_binary",
        "encode_categorical",
        "build_targets",
        "missing_audit",
        "impute_missing",
        "finalize_outputs",
    }
    assert expected.issubset(set(STAGE_REGISTRY.keys()))


def test_register_stage_decorator():
    import pandas as pd

    called: list[str] = []

    @register_stage("test_custom_stage")
    def _custom(ctx: PreprocessingContext, params: dict) -> None:
        called.append("ok")

    assert "test_custom_stage" in STAGE_REGISTRY
    STAGE_REGISTRY["test_custom_stage"].run(
        PreprocessingContext(
            pp=PreprocessingConfig(),
            experiment_profile="t",
            raw=pd.DataFrame({"a": [1]}),
        ),
        {},
    )
    assert called == ["ok"]
    del STAGE_REGISTRY["test_custom_stage"]


def test_unknown_stage_raises():
    pp = PreprocessingConfig(
        schema_version="3.0",
        pipeline=[PipelineStage(id="nonexistent_stage", enabled=True)],
    )
    ctx = PreprocessingContext(
        pp=pp,
        experiment_profile="test",
        raw=__import__("pandas").DataFrame({"a": [1]}),
    )
    with pytest.raises(ValueError, match="desconhecido"):
        run_registered_stages(ctx)
