"""Backend GPU opcional para clustering (cuML PCA + KMeans)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from pns_health_research.clustering.kmeans import run_elbow
from pns_health_research.clustering.results import ClusteringStageResult
from pns_health_research.config.models import ExperimentConfig
from pns_health_research.data.schema import CLUSTERING_COMORBIDITIES

logger = logging.getLogger("pns_health_research")


def _cuml_available() -> bool:
    try:
        import cuml  # noqa: F401

        return True
    except ImportError:
        return False


def run_cuml_clustering_stage(
    binary_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
    config: ExperimentConfig,
) -> ClusteringStageResult:
    if not _cuml_available():
        raise ImportError(
            "cuML não instalado. Use clustering.backend=sklearn ou instale extra [gpu]."
        )

    from cuml.cluster import KMeans
    from cuml.decomposition import PCA

    cc = config.clustering
    comorb_cols = cc.comorbidity_columns or CLUSTERING_COMORBIDITIES
    comorb_cols = [c for c in comorb_cols if c in binary_df.columns]
    x_df = binary_df[comorb_cols].fillna(0).astype(np.float32)

    pca = PCA(n_components=cc.reducer.n_components, random_state=cc.estimator.random_state)
    transformed = np.asarray(pca.fit_transform(x_df))

    elbow = run_elbow(
        transformed,
        cc.estimator.elbow_max_k,
        n_init=cc.estimator.elbow_n_init,
        random_state=cc.estimator.random_state,
    )

    kmeans = KMeans(
        n_clusters=cc.estimator.n_clusters,
        init=cc.estimator.init,
        n_init=cc.estimator.n_init,
        max_iter=cc.estimator.max_iter,
        random_state=cc.estimator.random_state,
    )
    labels = np.asarray(kmeans.fit_predict(transformed)).ravel()

    sil = float(silhouette_score(transformed, labels))
    centers = np.asarray(kmeans.cluster_centers_)
    wcss = float(np.sum((transformed - centers[labels]) ** 2))
    sizes = {int(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()}

    bin_out = binary_df.copy()
    cat_out = categorical_df.copy()
    bin_out["Cluster"] = labels
    cat_out["Cluster"] = labels

    profiles = x_df.copy()
    profiles["Cluster"] = labels
    profile_means = profiles.groupby("Cluster").mean(numeric_only=True)

    evr = np.asarray(pca.explained_variance_ratio_)
    pca_var = pd.DataFrame(
        {
            "component": range(1, len(evr) + 1),
            "explained_variance_ratio": evr,
            "cumulative_explained_variance": np.cumsum(evr),
        }
    )

    metrics = {
        "algorithm": "cuml_kmeans",
        "n_clusters": cc.estimator.n_clusters,
        "silhouette": sil,
        "wcss": wcss,
        "mean_error_per_point": wcss / len(labels),
        "cluster_sizes": sizes,
        "pca_n_components": cc.reducer.n_components,
        "backend": "cuml",
    }

    logger.info("Clustering cuML concluído (métricas não comparáveis ao baseline sklearn TCC)")

    return ClusteringStageResult(
        labels=labels,
        binary_with_cluster=bin_out,
        categorical_with_cluster=cat_out,
        metrics=metrics,
        elbow_results=elbow,
        pca_variance=pca_var,
        profiles=profile_means,
    )
