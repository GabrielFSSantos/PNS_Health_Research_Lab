"""Serviços de classificação."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from pns_health_research.web import query


@lru_cache(maxsize=64)
def list_models(run_id: str) -> tuple[str, ...]:
    metrics = query.get_run_metrics(run_id, "classification")
    return tuple(k for k, v in metrics.items() if isinstance(v, dict))


def load_grid_search(run_id: str, model: str) -> dict[str, Any]:
    return query.get_grid_search(run_id, model)


def grid_search_candidates_df(run_id: str, model: str) -> pd.DataFrame:
    data = load_grid_search(run_id, model)
    rows = data.get("candidates") or []
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_curves(run_id: str, model: str) -> dict[str, Any]:
    return query.get_curves(run_id, model)


def load_feature_importance(run_id: str, model: str) -> dict[str, pd.DataFrame]:
    data = query.get_feature_importance(run_id, model)
    native = pd.DataFrame(data.get("native") or [])
    perm = pd.DataFrame(data.get("permutation") or [])
    return {"native": native, "permutation": perm}


def load_classification_metrics(run_id: str) -> dict[str, Any]:
    return query.get_run_metrics(run_id, "classification")


def load_model_metrics(run_id: str, model: str) -> dict[str, Any]:
    metrics = load_classification_metrics(run_id)
    return metrics.get(model) or {}


def load_confusion_matrices(run_id: str) -> dict[str, Any]:
    return query.get_run_metrics(run_id, "confusion_matrices")


def load_folds(run_id: str) -> dict[str, Any]:
    return query.get_run_metrics(run_id, "classification_folds")


def load_predictions_summary(run_id: str) -> dict[str, Any]:
    return query.get_predictions_summary(run_id)


def normalize_prediction_summary(raw: dict[str, Any]) -> dict[str, Any]:
    """Normaliza chaves do artifact para a UI."""
    n = raw.get("n") or raw.get("n_samples")
    rate = raw.get("predicted_positive_rate")
    if rate is None:
        rate = raw.get("positive_rate")
    positives = None
    if n is not None and rate is not None:
        positives = int(round(float(n) * float(rate)))
    return {
        **raw,
        "n_samples": n,
        "n_positive": positives,
        "positive_rate": rate,
    }


def grid_search_table_df(run_id: str, model: str, best_params: dict[str, Any] | None = None) -> pd.DataFrame:
    df = grid_search_candidates_df(run_id, model)
    if df.empty:
        return df
    if best_params:
        df = df.copy()
        df["melhor"] = df.apply(
            lambda r: _params_match_candidate(r, best_params), axis=1
        )
    return df


def _params_match_candidate(row: pd.Series, best_params: dict[str, Any]) -> bool:
    for key, val in best_params.items():
        col = key if key in row.index else f"param_{key}"
        if col not in row.index:
            continue
        row_val = row[col]
        if pd.isna(row_val) and val is None:
            continue
        if str(row_val) != str(val):
            return False
    return True
