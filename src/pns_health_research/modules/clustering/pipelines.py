from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

from pns_health_research.modules.clustering.evaluation import compute_clustering_metrics
from pns_health_research.modules.clustering.kmeans import kneedle_k, run_elbow
from pns_health_research.modules.clustering.registry import get_clusterer
from pns_health_research.modules.clustering.results import ClusteringStageResult
from pns_health_research.config.models import ExperimentConfig
from pns_health_research.data.schema import CLUSTERING_COMORBIDITIES
from pns_health_research.modules.clustering.pca import PCAResult, fit_pca


def run_clustering_stage(
    binary_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
    config: ExperimentConfig,
) -> ClusteringStageResult:
    if config.clustering.backend == "cuml":
        from pns_health_research.modules.clustering.gpu.pipelines import run_cuml_clustering_stage

        return run_cuml_clustering_stage(binary_df, categorical_df, config)

    cc = config.clustering
    comorb_cols = cc.comorbidity_columns or CLUSTERING_COMORBIDITIES
    comorb_cols = [c for c in comorb_cols if c in binary_df.columns]
    x_df = binary_df[comorb_cols].fillna(0)

    if cc.reducer.enabled:
        transformed, pca_res, _ = fit_pca(
            x_df,
            cc.reducer.n_components,
            apply_scaler=cc.reducer.apply_scaler,
            scaler_type=cc.reducer.scaler_type,
            random_state=cc.estimator.random_state,
        )
    else:
        transformed = x_df.to_numpy(dtype=float)
        pca_res = PCAResult(
            components=transformed,
            explained_variance_ratio=np.array([]),
            cumulative_variance=np.array([]),
            loadings=pd.DataFrame(index=x_df.columns),
        )

    elbow = run_elbow(
        transformed,
        cc.estimator.elbow_max_k,
        n_init=cc.estimator.elbow_n_init,
        random_state=cc.estimator.random_state,
    )

    n_clusters = cc.estimator.n_clusters
    if cc.estimator.selection_method in ("elbow", "kneedle"):
        n_clusters = max(2, kneedle_k(elbow))
        config = config.model_copy(
            update={
                "clustering": cc.model_copy(
                    update={
                        "estimator": cc.estimator.model_copy(update={"n_clusters": n_clusters})
                    }
                )
            }
        )
        cc = config.clustering

    clusterer = get_clusterer(cc.estimator.name)
    result = clusterer(transformed, x_df, config)

    eval_metrics = compute_clustering_metrics(
        transformed,
        result.labels,
        wcss=result.metrics.wcss,
        n_clusters=result.metrics.n_clusters,
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
        "algorithm": cc.estimator.name,
        "backend": cc.backend,
        **eval_metrics,
        "pca_n_components": cc.reducer.n_components if cc.reducer.enabled else 0,
        "selection_method": cc.estimator.selection_method,
    }

    sil_samples = (
        silhouette_samples(transformed, result.labels)
        if len(set(result.labels)) > 1
        else np.array([])
    )
    sil_summary: dict = {}
    sil_sample_arr: np.ndarray | None = None
    if len(sil_samples):
        sil_summary = {
            "mean": float(np.mean(sil_samples)),
            "std": float(np.std(sil_samples)),
            "per_cluster_mean": {
                int(k): float(np.mean(sil_samples[result.labels == k]))
                for k in sorted(set(result.labels))
            },
        }
        sil_sample_arr = sil_samples

    loadings_df = pca_res.loadings.copy()
    loadings_df = loadings_df.reset_index(names="feature")

    return ClusteringStageResult(
        labels=result.labels,
        binary_with_cluster=bin_out,
        categorical_with_cluster=cat_out,
        metrics=metrics,
        elbow_results=elbow,
        pca_variance=pca_var,
        profiles=result.profiles,
        pca_loadings=loadings_df,
        pca_transformed=transformed,
        feature_names=comorb_cols,
        centroids=result.centroids if hasattr(result, "centroids") else None,
        silhouette_summary=sil_summary,
        silhouette_samples=sil_sample_arr,
    )
