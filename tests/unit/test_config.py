import numpy as np

from pns_health_research.clustering.alignment import align_cluster_labels
from pns_health_research.config.loader import load_config
from pns_health_research.core.paths import PROJECT_ROOT


def test_load_legacy_config():
    cfg = load_config(PROJECT_ROOT / "configs/experiments/legacy_reproduction.yaml")
    assert cfg.experiment.profile == "legacy_reproduction"
    assert cfg.clustering.reducer.n_components == 8


def test_align_cluster_labels():
    ref = np.array([[0, 0], [1, 1]])
    cand_centroids = np.array([[1, 1], [0, 0]])
    labels = np.array([0, 1, 0, 1])
    aligned = align_cluster_labels(ref, labels, cand_centroids)
    assert set(aligned) <= {0, 1}
