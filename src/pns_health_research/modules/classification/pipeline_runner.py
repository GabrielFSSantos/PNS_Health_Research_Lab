"""Orquestrador modular de classificação (lê clustering/ da base)."""

from __future__ import annotations

import pandas as pd

from pns_health_research.modules.classification.pipelines import (
    ClassificationStageResult,
    run_classification_stage,
)
from pns_health_research.config.models import ExperimentConfig
from pns_health_research.persistence.base_store import resolve_clustering_source


def run_classification_pipeline(
    config: ExperimentConfig,
    *,
    base_name: str,
    source_base: str | None = None,
) -> ClassificationStageResult:
    src = source_base or config.classification.source_base or base_name
    try:
        ref = resolve_clustering_source(src)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Agrupamento da base '{src}' não encontrado. "
            f"Execute: ./scripts/run_clustering.sh --profile {src}"
        ) from exc

    df = pd.read_parquet(ref.categorical_path)
    if "Cluster" not in df.columns:
        raise ValueError(
            f"clustered_categorical em {ref.categorical_path} não contém coluna 'Cluster'. "
            f"Execute o agrupamento antes da classificação."
        )
    return run_classification_stage(df, config)
