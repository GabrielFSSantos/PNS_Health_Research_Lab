"""Conversão schema 1.x flat → schema 2.0 (pipeline ordenado) para clustering."""

from __future__ import annotations

from typing import Any

from pns_health_research.config.models import ClusteringConfig, PipelineStage

DEFAULT_CLUSTERING_PIPELINE: tuple[str, ...] = (
    "load_preprocessing",
    "select_features",
    "pca_reducer",
    "elbow_selection",
    "kmeans_estimator",
    "attach_cluster_labels",
    "evaluate_metrics",
    "finalize_outputs",
)

STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "select_features": ("load_preprocessing",),
    "pca_reducer": ("select_features",),
    "elbow_selection": ("pca_reducer",),
    "kmeans_estimator": ("select_features",),
    "attach_cluster_labels": ("kmeans_estimator",),
    "evaluate_metrics": ("attach_cluster_labels",),
    "finalize_outputs": ("evaluate_metrics",),
}


def _stage_params(stage_id: str, cc: ClusteringConfig) -> dict[str, Any]:
    if stage_id == "pca_reducer":
        return {
            "n_components": cc.reducer.n_components,
            "apply_scaler": cc.reducer.apply_scaler,
            "scaler_type": cc.reducer.scaler_type,
        }
    if stage_id == "elbow_selection":
        return {
            "elbow_max_k": cc.estimator.elbow_max_k,
            "elbow_n_init": cc.estimator.elbow_n_init,
            "selection_method": cc.estimator.selection_method,
        }
    if stage_id == "kmeans_estimator":
        return {
            "name": cc.estimator.name,
            "n_clusters": cc.estimator.n_clusters,
            "selection_method": cc.estimator.selection_method,
            "n_init": cc.estimator.n_init,
            "max_iter": cc.estimator.max_iter,
            "random_state": cc.estimator.random_state,
        }
    return {}


def _stage_enabled(stage_id: str, cc: ClusteringConfig) -> bool:
    mapping: dict[str, bool] = {
        "load_preprocessing": True,
        "select_features": True,
        "pca_reducer": cc.reducer.enabled,
        "elbow_selection": cc.estimator.selection_method in ("elbow", "kneedle"),
        "kmeans_estimator": cc.estimator.enabled,
        "attach_cluster_labels": True,
        "evaluate_metrics": True,
        "finalize_outputs": True,
    }
    return mapping.get(stage_id, True)


def flat_config_to_pipeline(cc: ClusteringConfig) -> list[PipelineStage]:
    steps: list[PipelineStage] = []
    for stage_id in DEFAULT_CLUSTERING_PIPELINE:
        steps.append(
            PipelineStage(
                id=stage_id,
                enabled=_stage_enabled(stage_id, cc),
                params=_stage_params(stage_id, cc),
            )
        )
    return steps


def apply_flat_clustering_overrides(
    pipeline: list[PipelineStage],
    cc: ClusteringConfig,
) -> list[PipelineStage]:
    """Mescla params do pipeline YAML com valores flat de reducer/estimator."""
    merged: list[PipelineStage] = []
    for step in pipeline:
        defaults = _stage_params(step.id, cc)
        params = {**defaults, **step.params}
        merged.append(
            PipelineStage(id=step.id, enabled=step.enabled, params=params)
        )
    return merged


def validate_pipeline(pipeline: list[PipelineStage]) -> None:
    enabled_ids = {step.id for step in pipeline if step.enabled}
    for step in pipeline:
        if not step.enabled:
            continue
        for req in STAGE_DEPENDENCIES.get(step.id, ()):
            if req not in enabled_ids:
                raise ValueError(
                    f"Estágio de clustering {step.id!r} requer {req!r} habilitado"
                )
