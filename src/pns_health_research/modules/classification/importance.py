"""Extração de importância de features para classificadores."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def extract_native_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame | None:
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values(
            "importance", ascending=False
        )
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)
        return pd.DataFrame({"feature": feature_names, "importance": coef}).sort_values(
            "importance", ascending=False
        )
    return None


def compute_permutation_importance(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
    max_samples: int = 2000,
) -> pd.DataFrame:
    if len(x_test) > max_samples:
        x_sub = x_test.sample(n=max_samples, random_state=random_state)
        y_sub = y_test.loc[x_sub.index]
    else:
        x_sub, y_sub = x_test, y_test
    result = permutation_importance(
        model,
        x_sub,
        y_sub,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": list(x_test.columns),
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance", ascending=False)
