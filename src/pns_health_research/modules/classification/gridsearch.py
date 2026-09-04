"""Persistência e serialização de resultados de GridSearch."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV


def grid_search_config(
    param_grid: dict[str, Any],
    *,
    scoring: str,
    cv_folds: int,
    random_state: int,
    n_candidates: int | None = None,
) -> dict[str, Any]:
    n_combos = n_candidates
    if n_combos is None and param_grid:
        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v) if isinstance(v, list) else 1
    return {
        "algorithm": "grid",
        "scoring": scoring,
        "cv_folds": cv_folds,
        "random_state": random_state,
        "param_grid": param_grid,
        "n_candidates": n_combos or 0,
        "n_fits": (n_combos or 0) * cv_folds,
    }


def cv_results_to_dataframe(grid: GridSearchCV) -> pd.DataFrame:
    return pd.DataFrame(grid.cv_results_)


def serialize_grid_result(
    grid: GridSearchCV | None,
    param_grid: dict[str, Any],
    *,
    scoring: str,
    cv_folds: int,
    random_state: int,
    final_estimator: Any,
) -> dict[str, Any]:
    if grid is None:
        return {
            "search": grid_search_config(
                param_grid, scoring=scoring, cv_folds=cv_folds, random_state=random_state
            ),
            "best_params": {},
            "best_score": None,
            "best_index": None,
            "cv_results": pd.DataFrame(),
            "final_params": dict(final_estimator.get_params()) if final_estimator else {},
        }
    cv_df = cv_results_to_dataframe(grid)
    return {
        "search": grid_search_config(
            param_grid,
            scoring=scoring,
            cv_folds=cv_folds,
            random_state=random_state,
            n_candidates=len(cv_df),
        ),
        "best_params": dict(grid.best_params_),
        "best_score": float(grid.best_score_),
        "best_index": int(grid.best_index_),
        "cv_results": cv_df,
        "final_params": dict(final_estimator.get_params()),
    }
