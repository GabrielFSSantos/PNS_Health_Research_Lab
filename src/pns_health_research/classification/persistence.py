"""Persistência de artifacts de classificação por modelo."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pns_health_research.classification.validation import ModelResult
from pns_health_research.persistence.artifact_store import ArtifactStore


def persist_model_artifacts(store: ArtifactStore, result: ModelResult) -> None:
    name = result.model_name
    model_dir = f"models/{name}"
    store.write_json(f"{model_dir}/search.json", result.search_config)
    store.write_json(f"{model_dir}/best_params.json", result.best_params)
    store.write_json(
        f"{model_dir}/best_score.json",
        {
            "best_score": result.best_score,
            "best_index": result.best_index,
            "scoring": result.search_config.get("scoring"),
        },
    )
    store.write_json(f"{model_dir}/final_params.json", result.final_params)

    if not result.cv_results.empty:
        store.write_parquet(f"tables/grid_search/{name}.parquet", result.cv_results)

    if result.fold_search_results:
        fold_rows = [
            {
                "fold": fs.fold,
                "best_score": fs.best_score,
                **{f"param_{k}": v for k, v in fs.best_params.items()},
            }
            for fs in result.fold_search_results
        ]
        store.write_parquet(
            f"tables/grid_search_folds/{name}.parquet",
            pd.DataFrame(fold_rows),
        )

    if not result.feature_importance.empty:
        store.write_parquet(
            f"tables/feature_importance/{name}.parquet",
            result.feature_importance,
        )
    if not result.permutation_importance.empty:
        store.write_parquet(
            f"tables/permutation_importance/{name}.parquet",
            result.permutation_importance,
        )


def persist_classification_aggregates(
    store: ArtifactStore,
    model_results: dict[str, ModelResult | dict[str, Any]],
) -> None:
    fold_payload: dict[str, list[dict[str, Any]]] = {}
    cm_payload: dict[str, list[list[int]]] = {}

    for model_name, res in model_results.items():
        if not isinstance(res, ModelResult):
            continue
        fold_payload[model_name] = [
            {"fold": fm.fold, **{k: v for k, v in fm.metrics.items() if k != "confusion_matrix"}}
            for fm in res.fold_metrics
        ]
        if res.confusion_matrix_sum:
            cm_payload[model_name] = res.confusion_matrix_sum

    if fold_payload:
        store.write_json("metrics/classification_folds.json", fold_payload)
    if cm_payload:
        store.write_json("metrics/confusion_matrices.json", cm_payload)


def predictions_summary(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {"n": 0}
    n = len(predictions)
    pos_rate = (
        float(predictions["predicted_label"].mean())
        if "predicted_label" in predictions
        else 0.0
    )
    summary: dict[str, Any] = {
        "n": n,
        "predicted_positive_rate": pos_rate,
    }
    if "proba_positive" in predictions.columns:
        proba = predictions["proba_positive"].dropna()
        if len(proba):
            counts, edges = np.histogram(proba, bins=5, range=(0.0, 1.0))
            summary["proba_histogram"] = {
                "bins": [float(x) for x in edges],
                "counts": [int(c) for c in counts],
            }
    return summary
