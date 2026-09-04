"""Orquestrador do pipeline de agrupamento (schema 2.0)."""

from __future__ import annotations

from pns_health_research.modules.clustering.contract import ClusteringContext
from pns_health_research.modules.clustering.pipeline_adapter import validate_pipeline
from pns_health_research.modules.clustering.results import ClusteringStageResult
from pns_health_research.modules.clustering.stages.registry import (
    build_stage_result,
    run_registered_stages,
)
from pns_health_research.config.models import ExperimentConfig


def run_clustering_pipeline(
    config: ExperimentConfig,
    *,
    base_name: str,
    source_base: str | None = None,
) -> ClusteringStageResult:
    cc = config.clustering
    validate_pipeline(cc.resolved_pipeline())

    src = source_base or cc.source_base or base_name

    ctx = ClusteringContext(
        config=config,
        cc=cc,
        source_base=src,
        base_name=base_name,
    )
    run_registered_stages(ctx)
    return build_stage_result(ctx)
