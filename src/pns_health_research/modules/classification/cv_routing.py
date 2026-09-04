"""Roteamento do protocolo de validação cruzada na classificação."""

from __future__ import annotations

import pandas as pd

from pns_health_research.modules.classification.validation import (
    ModelResult,
    run_legacy_cv,
    run_research_v2_cv,
)
from pns_health_research.config.models import ExperimentConfig


def run_model_cv(
    x: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    model_name: str,
    config: ExperimentConfig,
) -> ModelResult:
    """Executa CV do modelo conforme legacy_cv / profile / backend."""
    if config.classification.legacy_cv:
        return run_legacy_cv(x, y, weights, model_name, config)
    if (
        config.experiment.profile == "research_v2"
        and config.classification.backend == "gpu"
    ):
        from pns_health_research.modules.classification.gpu.pipelines import run_gpu_cv

        return run_gpu_cv(x, y, weights, model_name, config)
    if config.experiment.profile == "research_v2":
        return run_research_v2_cv(x, y, weights, model_name, config)
    return run_research_v2_cv(x, y, weights, model_name, config)
