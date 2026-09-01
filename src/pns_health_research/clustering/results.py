from __future__ import annotations

from dataclasses import dataclass

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
