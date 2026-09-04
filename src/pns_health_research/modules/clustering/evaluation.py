"""Métricas de avaliação de clustering."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def compute_clustering_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    wcss: float | None = None,
    n_clusters: int | None = None,
) -> dict[str, Any]:
    n_points = x.shape[0]
    n_k = n_clusters or len(set(labels))
    sil = float(silhouette_score(x, labels)) if len(set(labels)) > 1 else 0.0
    inertia = wcss if wcss is not None else 0.0
    unique, counts = np.unique(labels, return_counts=True)
    sizes = {int(k): int(v) for k, v in zip(unique, counts, strict=False)}
    out: dict[str, Any] = {
        "n_clusters": n_k,
        "silhouette": sil,
        "wcss": inertia,
        "mean_error_per_point": inertia / n_points if n_points else 0.0,
        "cluster_sizes": sizes,
    }
    if len(set(labels)) > 1 and x.shape[0] > n_k:
        try:
            out["davies_bouldin"] = float(davies_bouldin_score(x, labels))
            out["calinski_harabasz"] = float(calinski_harabasz_score(x, labels))
        except ValueError:
            pass
    return out


def compare_cluster_labels(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    return {
        "adjusted_rand_index": float(adjusted_rand_score(reference, candidate)),
        "normalized_mutual_info": float(normalized_mutual_info_score(reference, candidate)),
    }
