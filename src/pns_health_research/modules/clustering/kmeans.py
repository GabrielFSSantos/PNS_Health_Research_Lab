from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class ClusteringMetrics:
    n_clusters: int
    silhouette: float
    wcss: float
    mean_error_per_point: float
    cluster_sizes: dict[int, int]


@dataclass
class ClusteringResult:
    labels: np.ndarray
    metrics: ClusteringMetrics
    centroids: np.ndarray
    profiles: pd.DataFrame
    model: KMeans
    elbow_table: pd.DataFrame | None = None


def run_elbow(
    x: np.ndarray,
    max_k: int,
    *,
    n_init: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    rows = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=random_state)
        km.fit(x)
        rows.append({"k": k, "wcss": km.inertia_})
    return pd.DataFrame(rows)


def kneedle_k(elbow_df: pd.DataFrame, value_col: str = "wcss") -> int:
    x = elbow_df["k"].to_numpy(dtype=float)
    y = elbow_df[value_col].to_numpy(dtype=float)
    if len(x) < 2:
        return int(x[0]) if len(x) else 2
    # use EV-style: invert wcss for distance to line
    y_norm = (y.max() - y) / (y.max() - y.min() + 1e-12)
    x1, y1 = x[0], y_norm[0]
    x2, y2 = x[-1], y_norm[-1]
    num = np.abs((y2 - y1) * x - (x2 - x1) * y_norm + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    dist = num / den
    return int(x[np.argmax(dist)])


def run_kmeans_clustering(
    x: np.ndarray,
    feature_df: pd.DataFrame,
    *,
    n_clusters: int = 6,
    n_init: int = 50,
    max_iter: int = 1000,
    random_state: int = 42,
    sample_weight: np.ndarray | None = None,
) -> ClusteringResult:
    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = km.fit_predict(x, sample_weight=sample_weight)
    sil = float(silhouette_score(x, labels))
    wcss = float(km.inertia_)
    n_points = x.shape[0]
    profiles = feature_df.copy()
    profiles["Cluster"] = labels
    profile_means = profiles.groupby("Cluster").mean(numeric_only=True)
    sizes = {int(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()}
    metrics = ClusteringMetrics(
        n_clusters=n_clusters,
        silhouette=sil,
        wcss=wcss,
        mean_error_per_point=wcss / n_points,
        cluster_sizes=sizes,
    )
    return ClusteringResult(
        labels=labels,
        metrics=metrics,
        centroids=km.cluster_centers_,
        profiles=profile_means,
        model=km,
    )
