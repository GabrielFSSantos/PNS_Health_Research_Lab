from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class PCAResult:
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    loadings: pd.DataFrame


def fit_pca(
    x: pd.DataFrame,
    n_components: int,
    *,
    apply_scaler: bool = False,
    scaler_type: str = "none",
    random_state: int = 42,
) -> tuple[np.ndarray, PCAResult, object | None]:
    data = x.values.astype(float)
    scaler = None
    if apply_scaler:
        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        data = scaler.fit_transform(data)
    pca = PCA(n_components=n_components, random_state=random_state)
    transformed = pca.fit_transform(data)
    evr = pca.explained_variance_ratio_
    result = PCAResult(
        components=transformed,
        explained_variance_ratio=evr,
        cumulative_variance=np.cumsum(evr),
        loadings=pd.DataFrame(pca.components_.T, index=x.columns),
    )
    return transformed, result, scaler
