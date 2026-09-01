from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from pns_health_research.clustering.kmeans import ClusteringResult, run_kmeans_clustering
from pns_health_research.config.models import ExperimentConfig

ClustererFn = Callable[
    [np.ndarray, pd.DataFrame, ExperimentConfig],
    ClusteringResult,
]

CLUSTERER_REGISTRY: dict[str, ClustererFn] = {}


def register_clusterer(name: str) -> Callable[[ClustererFn], ClustererFn]:
    def decorator(fn: ClustererFn) -> ClustererFn:
        CLUSTERER_REGISTRY[name] = fn
        return fn

    return decorator


@register_clusterer("kmeans")
def _kmeans_clusterer(
    x: np.ndarray,
    feature_df: pd.DataFrame,
    config: ExperimentConfig,
) -> ClusteringResult:
    cc = config.clustering.estimator
    return run_kmeans_clustering(
        x,
        feature_df,
        n_clusters=cc.n_clusters,
        n_init=cc.n_init,
        max_iter=cc.max_iter,
        random_state=cc.random_state,
    )


def get_clusterer(name: str) -> ClustererFn:
    if name not in CLUSTERER_REGISTRY:
        raise KeyError(
            f"Clusterer '{name}' não registrado. Disponíveis: {list(CLUSTERER_REGISTRY)}"
        )
    return CLUSTERER_REGISTRY[name]
