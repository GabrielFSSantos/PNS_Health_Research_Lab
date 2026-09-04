"""Helpers para persistência de search config em paths sem GridSearchCV."""

from __future__ import annotations

from typing import Any


def gpu_search_config(
    model_name: str,
    param_grid: dict[str, Any],
    *,
    cv_folds: int,
    random_state: int,
) -> dict[str, Any]:
    return {
        "algorithm": "gpu_defaults",
        "scoring": "f1_macro",
        "cv_folds": cv_folds,
        "random_state": random_state,
        "param_grid": param_grid,
        "n_candidates": 0,
        "n_fits": 0,
        "note": f"GPU path for {model_name} uses default params without GridSearchCV",
    }
