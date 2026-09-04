"""Artefatos de visualização do agrupamento (dashboard)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pns_health_research.modules.analysis.cluster_service_usage import (
    bubble_health_service_table,
    cluster_profile_table,
    service_usage_by_cluster,
)
from pns_health_research.modules.clustering.results import ClusteringStageResult


def persist_clustering_viz(
    result: ClusteringStageResult,
    out: Path,
    *,
    seed: int = 42,
) -> None:
    """Grava tabelas e métricas extras consumidas pelo dashboard."""
    tables_dir = out / "tables"
    metrics_dir = out / "metrics"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not result.elbow_results.empty:
        result.elbow_results.to_parquet(tables_dir / "elbow_results.parquet", index=False)
    if not result.pca_variance.empty:
        result.pca_variance.to_parquet(tables_dir / "pca_explained_variance.parquet", index=False)
        result.pca_variance.to_parquet(tables_dir / "pca_variance.parquet", index=False)
    if not result.pca_loadings.empty:
        result.pca_loadings.to_parquet(tables_dir / "pca_loadings.parquet", index=False)
    if not result.profiles.empty:
        profiles_table = cluster_profile_table(result.profiles)
        profiles_table.to_parquet(tables_dir / "cluster_profiles.parquet", index=False)
        profiles_table.to_parquet(tables_dir / "cluster_profiles_table.parquet", index=False)

    result.binary_with_cluster.to_parquet(tables_dir / "clustered_binary.parquet", index=False)
    result.categorical_with_cluster.to_parquet(
        tables_dir / "clustered_categorical.parquet", index=False
    )

    cross = service_usage_by_cluster(result.binary_with_cluster)
    if not cross.empty:
        cross.to_parquet(tables_dir / "cross_analysis.parquet", index=False)
    bubble = bubble_health_service_table(result.binary_with_cluster)
    if not bubble.empty:
        bubble.to_parquet(tables_dir / "bubble_health_service.parquet", index=False)

    if result.silhouette_summary:
        (metrics_dir / "silhouette_summary.json").write_text(
            __import__("json").dumps(result.silhouette_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if result.pca_transformed is not None:
        n = result.pca_transformed.shape[0]
        sample_size = min(5000, n)
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_size, replace=False)
        scatter = pd.DataFrame(
            {
                "pc1": result.pca_transformed[idx, 0],
                "pc2": (
                    result.pca_transformed[idx, 1]
                    if result.pca_transformed.shape[1] > 1
                    else 0
                ),
                "cluster": result.labels[idx],
            }
        )
        scatter.to_parquet(tables_dir / "pca_scatter_sample.parquet", index=False)

    if result.centroids is not None:
        centroids_df = pd.DataFrame(
            result.centroids,
            columns=[f"pc{i + 1}" for i in range(result.centroids.shape[1])],
        )
        centroids_df.insert(0, "cluster", range(len(centroids_df)))
        centroids_df.to_parquet(tables_dir / "centroids.parquet", index=False)

    if result.silhouette_samples is not None and result.pca_transformed is not None:
        n = len(result.silhouette_samples)
        sample_size = min(2000, n)
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_size, replace=False)
        sil_df = pd.DataFrame(
            {
                "silhouette": result.silhouette_samples[idx],
                "cluster": result.labels[idx],
            }
        )
        sil_df.to_parquet(tables_dir / "silhouette_samples.parquet", index=False)
