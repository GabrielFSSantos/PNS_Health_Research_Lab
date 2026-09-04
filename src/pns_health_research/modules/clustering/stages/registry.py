"""Registry e estágios do pipeline de agrupamento."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pns_health_research.modules.clustering.contract import ClusteringContext
from pns_health_research.modules.clustering.evaluation import compute_clustering_metrics
from pns_health_research.modules.clustering.kmeans import kneedle_k, run_elbow
from pns_health_research.modules.clustering.registry import get_clusterer
from pns_health_research.modules.clustering.results import ClusteringStageResult
from pns_health_research.data.schema import CLUSTERING_COMORBIDITIES
from pns_health_research.modules.clustering.pca import PCAResult, fit_pca
from pns_health_research.persistence.base_store import resolve_preprocessing_source
from sklearn.metrics import silhouette_samples

StageFn = Callable[[ClusteringContext, dict[str, Any]], None]

STAGE_REGISTRY: dict[str, "StageSpec"] = {}


@dataclass(frozen=True)
class StageSpec:
    id: str
    run: StageFn
    requires: tuple[str, ...] = ()


def register_stage(
    stage_id: str,
    *,
    requires: tuple[str, ...] = (),
) -> Callable[[StageFn], StageFn]:
    def decorator(fn: StageFn) -> StageFn:
        STAGE_REGISTRY[stage_id] = StageSpec(stage_id, fn, requires)
        return fn

    return decorator


@register_stage("load_preprocessing")
def _run_load_preprocessing(ctx: ClusteringContext, params: dict[str, Any]) -> None:
    _ = params
    ref = resolve_preprocessing_source(ctx.source_base)
    ctx.binary_df = pd.read_parquet(ref.binary_path)
    ctx.categorical_df = pd.read_parquet(ref.categorical_path)
    ctx.processing_stats["source_base"] = ctx.source_base
    ctx.processing_stats["preprocessing_path"] = str(ref.path)


@register_stage("select_features", requires=("load_preprocessing",))
def _run_select_features(ctx: ClusteringContext, params: dict[str, Any]) -> None:
    _ = params
    assert ctx.binary_df is not None
    comorb_cols = ctx.cc.comorbidity_columns or CLUSTERING_COMORBIDITIES
    comorb_cols = [c for c in comorb_cols if c in ctx.binary_df.columns]
    ctx.feature_names = comorb_cols
    ctx.feature_df = ctx.binary_df[comorb_cols].fillna(0)


@register_stage("pca_reducer", requires=("select_features",))
def _run_pca_reducer(ctx: ClusteringContext, params: dict[str, Any]) -> None:
    assert ctx.feature_df is not None
    n_components = int(params.get("n_components", ctx.cc.reducer.n_components))
    apply_scaler = bool(params.get("apply_scaler", ctx.cc.reducer.apply_scaler))
    scaler_type = str(params.get("scaler_type", ctx.cc.reducer.scaler_type))
    random_state = int(params.get("random_state", ctx.cc.estimator.random_state))

    transformed, pca_res, _ = fit_pca(
        ctx.feature_df,
        n_components,
        apply_scaler=apply_scaler,
        scaler_type=scaler_type,
        random_state=random_state,
    )
    ctx.transformed = transformed
    ctx.pca_result = pca_res


@register_stage("elbow_selection", requires=("pca_reducer",))
def _run_elbow_selection(ctx: ClusteringContext, params: dict[str, Any]) -> None:
    assert ctx.transformed is not None
    elbow_max_k = int(params.get("elbow_max_k", ctx.cc.estimator.elbow_max_k))
    elbow_n_init = int(params.get("elbow_n_init", ctx.cc.estimator.elbow_n_init))
    random_state = int(params.get("random_state", ctx.cc.estimator.random_state))
    selection_method = str(params.get("selection_method", ctx.cc.estimator.selection_method))

    ctx.elbow_results = run_elbow(
        ctx.transformed,
        elbow_max_k,
        n_init=elbow_n_init,
        random_state=random_state,
    )
    if selection_method in ("elbow", "kneedle"):
        ctx.n_clusters = max(2, kneedle_k(ctx.elbow_results))


@register_stage("kmeans_estimator", requires=("select_features",))
def _run_kmeans_estimator(ctx: ClusteringContext, params: dict[str, Any]) -> None:
    if ctx.transformed is None:
        assert ctx.feature_df is not None
        ctx.transformed = ctx.feature_df.to_numpy(dtype=float)
        ctx.pca_result = PCAResult(
            components=ctx.transformed,
            explained_variance_ratio=np.array([]),
            cumulative_variance=np.array([]),
            loadings=pd.DataFrame(index=ctx.feature_df.columns),
        )

    n_clusters = int(params.get("n_clusters", ctx.n_clusters))
    selection_method = str(params.get("selection_method", ctx.cc.estimator.selection_method))
    if selection_method in ("elbow", "kneedle") and ctx.elbow_results is not None:
        n_clusters = ctx.n_clusters

    estimator_name = str(params.get("name", ctx.cc.estimator.name))
    clusterer = get_clusterer(estimator_name)

    cc_copy = ctx.config.model_copy(
        update={
            "clustering": ctx.cc.model_copy(
                update={
                    "estimator": ctx.cc.estimator.model_copy(update={"n_clusters": n_clusters})
                }
            )
        }
    )
    assert ctx.feature_df is not None and ctx.transformed is not None
    ctx.cluster_result = clusterer(ctx.transformed, ctx.feature_df, cc_copy)
    ctx.labels = ctx.cluster_result.labels
    ctx.n_clusters = n_clusters


@register_stage("attach_cluster_labels", requires=("kmeans_estimator",))
def _run_attach_cluster_labels(ctx: ClusteringContext, params: dict[str, Any]) -> None:
    _ = params
    assert (
        ctx.binary_df is not None
        and ctx.categorical_df is not None
        and ctx.labels is not None
    )
    ctx.binary_out = ctx.binary_df.copy()
    ctx.categorical_out = ctx.categorical_df.copy()
    ctx.binary_out["Cluster"] = ctx.labels
    ctx.categorical_out["Cluster"] = ctx.labels


@register_stage("evaluate_metrics", requires=("attach_cluster_labels",))
def _run_evaluate_metrics(ctx: ClusteringContext, params: dict[str, Any]) -> None:
    _ = params
    assert ctx.transformed is not None and ctx.labels is not None
    eval_metrics = compute_clustering_metrics(
        ctx.transformed,
        ctx.labels,
        wcss=ctx.cluster_result.metrics.wcss if ctx.cluster_result else None,
        n_clusters=ctx.n_clusters,
    )
    ctx.metrics.update(eval_metrics)
    ctx.metrics["algorithm"] = ctx.cc.estimator.name
    ctx.metrics["backend"] = ctx.cc.backend
    ctx.metrics["n_clusters"] = ctx.n_clusters
    ctx.metrics["selection_method"] = ctx.cc.estimator.selection_method
    ctx.metrics["pca_n_components"] = (
        ctx.cc.reducer.n_components if ctx.cc.reducer.enabled else 0
    )

    if len(set(ctx.labels)) > 1:
        sil = silhouette_samples(ctx.transformed, ctx.labels)
        ctx.metrics["silhouette_summary"] = {
            "mean": float(np.mean(sil)),
            "std": float(np.std(sil)),
        }
        ctx.processing_stats["silhouette_samples"] = sil


@register_stage("finalize_outputs", requires=("evaluate_metrics",))
def _run_finalize_outputs(ctx: ClusteringContext, params: dict[str, Any]) -> None:
    _ = params
    ctx.metrics["pipeline_executed"] = ctx.pipeline_executed
    ctx.metrics["source_base"] = ctx.source_base
    ctx.metrics["base_name"] = ctx.base_name
    if ctx.binary_out is not None:
        ctx.metrics["rows_out"] = len(ctx.binary_out)


def run_registered_stages(ctx: ClusteringContext) -> None:
    from pns_health_research.core.run_progress import RunProgress

    pipeline = ctx.cc.resolved_pipeline()
    enabled_ids = {step.id for step in pipeline if step.enabled}
    stage_stats: dict[str, Any] = {}
    progress = RunProgress("agrupamento", title="Agrupamento (clustering)")
    progress.banner(
        base=ctx.base_name,
        source_base=ctx.source_base,
        backend=ctx.cc.backend,
    )

    for idx, step in enumerate(pipeline, start=1):
        spec = STAGE_REGISTRY.get(step.id)
        if spec is None:
            raise ValueError(f"Estágio de clustering desconhecido: {step.id!r}")

        record = {"id": step.id, "enabled": step.enabled, "params": dict(step.params)}
        ctx.pipeline_executed.append(record)

        if not step.enabled:
            stage_stats[step.id] = {"enabled": False, "skipped": True}
            progress.step_skip(idx, len(pipeline), step.id)
            continue

        for req in spec.requires:
            if req not in enabled_ids:
                raise ValueError(
                    f"Estágio {step.id!r} requer {req!r} habilitado no pipeline"
                )

        t0 = time.perf_counter()
        progress.step_begin(idx, len(pipeline), step.id)
        spec.run(ctx, step.params)
        stage_stats[step.id] = {
            "enabled": True,
            "elapsed_seconds": round(time.perf_counter() - t0, 4),
        }
        extra = ""
        if step.id == "kmeans_estimator" and ctx.n_clusters:
            extra = f"k={ctx.n_clusters}"
        elif step.id == "evaluate_metrics" and ctx.metrics.get("silhouette") is not None:
            extra = f"silhouette={ctx.metrics['silhouette']:.4f}"
        progress.step_end(extra)

    ctx.processing_stats["stages"] = stage_stats
    if any(s.enabled for s in pipeline):
        k = ctx.n_clusters or ctx.metrics.get("n_clusters", "?")
        sil = ctx.metrics.get("silhouette")
        summary = f"k={k}"
        if sil is not None:
            summary += f", silhouette={sil:.4f}"
        progress.finish(summary)


def build_stage_result(ctx: ClusteringContext) -> ClusteringStageResult:
    assert (
        ctx.labels is not None
        and ctx.binary_out is not None
        and ctx.categorical_out is not None
        and ctx.feature_df is not None
        and ctx.transformed is not None
        and ctx.pca_result is not None
        and ctx.cluster_result is not None
    )

    pca_var = pd.DataFrame(
        {
            "component": range(1, len(ctx.pca_result.explained_variance_ratio) + 1),
            "explained_variance_ratio": ctx.pca_result.explained_variance_ratio,
            "cumulative_explained_variance": ctx.pca_result.cumulative_variance,
        }
    )
    loadings_df = ctx.pca_result.loadings.copy().reset_index(names="feature")
    elbow = ctx.elbow_results if ctx.elbow_results is not None else pd.DataFrame()

    sil_samples = ctx.processing_stats.get("silhouette_samples")
    sil_summary = ctx.metrics.get("silhouette_summary", {})

    return ClusteringStageResult(
        labels=ctx.labels,
        binary_with_cluster=ctx.binary_out,
        categorical_with_cluster=ctx.categorical_out,
        metrics=ctx.metrics,
        elbow_results=elbow,
        pca_variance=pca_var,
        profiles=ctx.cluster_result.profiles,
        pca_loadings=loadings_df,
        pca_transformed=ctx.transformed,
        feature_names=ctx.feature_names,
        centroids=ctx.cluster_result.centroids
        if hasattr(ctx.cluster_result, "centroids")
        else None,
        silhouette_summary=sil_summary if isinstance(sil_summary, dict) else {},
        silhouette_samples=sil_samples if isinstance(sil_samples, np.ndarray) else None,
    )
