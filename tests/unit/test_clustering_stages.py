"""Testes do pipeline modular de clustering."""

from __future__ import annotations

import pytest

from pns_health_research.config.models import ClusteringConfig, PipelineStage
from pns_health_research.modules.clustering.contract import ClusteringContext
from pns_health_research.modules.clustering.stages.registry import STAGE_REGISTRY, run_registered_stages
from pns_health_research.config.models import ExperimentConfig, ExperimentMeta


def test_builtin_clustering_stages():
    expected = {
        "load_preprocessing",
        "select_features",
        "pca_reducer",
        "elbow_selection",
        "kmeans_estimator",
        "attach_cluster_labels",
        "evaluate_metrics",
        "finalize_outputs",
    }
    assert expected.issubset(set(STAGE_REGISTRY.keys()))


def test_unknown_clustering_stage_raises():
    cc = ClusteringConfig(
        schema_version="2.0",
        pipeline=[PipelineStage(id="unknown_stage", enabled=True)],
    )
    cfg = ExperimentConfig(experiment=ExperimentMeta(name="t", profile="t"))
    cfg.clustering = cc
    ctx = ClusteringContext(
        config=cfg,
        cc=cc,
        source_base="pns",
        base_name="pns",
    )
    with pytest.raises(ValueError, match="desconhecido"):
        run_registered_stages(ctx)
