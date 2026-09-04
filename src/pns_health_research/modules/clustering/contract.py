"""Contrato de estado e resultado do pipeline de agrupamento."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pns_health_research.config.models import ClusteringConfig, ExperimentConfig
from pns_health_research.modules.clustering.pca import PCAResult


@dataclass
class ClusteringContext:
    config: ExperimentConfig
    cc: ClusteringConfig
    source_base: str
    base_name: str
    binary_df: pd.DataFrame | None = None
    categorical_df: pd.DataFrame | None = None
    feature_df: pd.DataFrame | None = None
    feature_names: list[str] = field(default_factory=list)
    transformed: np.ndarray | None = None
    pca_result: PCAResult | None = None
    elbow_results: pd.DataFrame | None = None
    n_clusters: int = 6
    labels: np.ndarray | None = None
    cluster_result: Any = None
    binary_out: pd.DataFrame | None = None
    categorical_out: pd.DataFrame | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    pipeline_executed: list[dict[str, Any]] = field(default_factory=list)
    processing_stats: dict[str, Any] = field(default_factory=dict)
