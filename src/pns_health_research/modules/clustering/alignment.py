from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def align_cluster_labels(
    reference_centroids: np.ndarray,
    candidate_labels: np.ndarray,
    candidate_centroids: np.ndarray,
) -> np.ndarray:
    """Hungarian matching of cluster IDs by centroid distance."""
    n = reference_centroids.shape[0]
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = np.linalg.norm(reference_centroids[i] - candidate_centroids[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {col_ind[k]: row_ind[k] for k in range(n)}
    return np.array([mapping.get(label, label) for label in candidate_labels])
