from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ClusteringStageResult:
    labels: np.ndarray
    binary_with_cluster: pd.DataFrame
    categorical_with_cluster: pd.DataFrame
    metrics: dict
    elbow_results: pd.DataFrame
    pca_variance: pd.DataFrame
    profiles: pd.DataFrame
    pca_loadings: pd.DataFrame = field(default_factory=pd.DataFrame)
    pca_transformed: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    centroids: np.ndarray | None = None
    silhouette_summary: dict = field(default_factory=dict)
    silhouette_samples: np.ndarray | None = None
