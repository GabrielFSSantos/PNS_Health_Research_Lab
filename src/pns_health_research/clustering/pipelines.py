from __future__ import annotations

import pandas as pd

from pns_health_research.clustering.kmeans import ClusteringResult, run_elbow, run_kmeans_clustering
from pns_health_research.clustering.results import ClusteringStageResult
from pns_health_research.config.models import ExperimentConfig
from pns_health_research.data.schema import CLUSTERING_COMORBIDITIES
from pns_health_research.dimensionality.pca import fit_pca


def run_clustering_stage(
    binary_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
    config: ExperimentConfig,
) -> ClusteringStageResult:
    if config.clustering.backend == "cuml":
        from pns_health_research.clustering.gpu.pipelines import run_cuml_clustering_stage

        return run_cuml_clustering_stage(binary_df, categorical_df, config)

    cc = config.clustering
    comorb_cols = cc.comorbidity_columns or CLUSTERING_COMORBIDITIES
    comorb_cols = [c for c in comorb_cols if c in binary_df.columns]
    x_df = binary_df[comorb_cols].fillna(0)

    transformed, pca_res, _ = fit_pca(
        x_df,
        cc.reducer.n_components,
        apply_scaler=cc.reducer.apply_scaler,
        scaler_type=cc.reducer.scaler_type,
        random_state=cc.estimator.random_state,
    )

    elbow = run_elbow(
        transformed,
        cc.estimator.elbow_max_k,
        n_init=cc.estimator.elbow_n_init,
        random_state=cc.estimator.random_state,
    )

    n_k = cc.estimator.n_clusters
    result: ClusteringResult = run_kmeans_clustering(
        transformed,
        x_df,
        n_clusters=n_k,
        n_init=cc.estimator.n_init,
        max_iter=cc.estimator.max_iter,
        random_state=cc.estimator.random_state,
    )

    bin_out = binary_df.copy()
    cat_out = categorical_df.copy()
    bin_out["Cluster"] = result.labels
    cat_out["Cluster"] = result.labels

    pca_var = pd.DataFrame(
        {
            "component": range(1, len(pca_res.explained_variance_ratio) + 1),
            "explained_variance_ratio": pca_res.explained_variance_ratio,
            "cumulative_explained_variance": pca_res.cumulative_variance,
        }
    )

    metrics = {
        "algorithm": "kmeans",
        "n_clusters": result.metrics.n_clusters,
        "silhouette": result.metrics.silhouette,
        "wcss": result.metrics.wcss,
        "mean_error_per_point": result.metrics.mean_error_per_point,
        "cluster_sizes": result.metrics.cluster_sizes,
        "pca_n_components": cc.reducer.n_components,
    }

    return ClusteringStageResult(
        labels=result.labels,
        binary_with_cluster=bin_out,
        categorical_with_cluster=cat_out,
        metrics=metrics,
        elbow_results=elbow,
        pca_variance=pca_var,
        profiles=result.profiles,
    )
