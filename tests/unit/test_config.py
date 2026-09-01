import numpy as np

from pns_health_research.clustering.alignment import align_cluster_labels
from pns_health_research.config.loader import load_config
from pns_health_research.core.paths import PROJECT_ROOT


def test_load_legacy_config():
    cfg = load_config(PROJECT_ROOT / "configs/experiments/legacy.yaml")
    assert cfg.experiment.profile == "legacy_reproduction"
    assert cfg.data.year == 2019
    assert cfg.data.dataset == "pns_2019"
    assert cfg.clustering.reducer.n_components == 8
    assert cfg.preprocessing.nearmiss_before_split is True
    assert cfg.preprocessing.legacy_weight_formula is True


def test_load_experiment_config():
    cfg = load_config(PROJECT_ROOT / "configs/experiment.yaml")
    assert cfg.experiment.profile == "custom"
    assert "preprocessing" in cfg.stages
    assert cfg.preprocessing.nearmiss_before_split is False


def test_align_cluster_labels():
    ref = np.array([[0, 0], [1, 1]])
    cand_centroids = np.array([[1, 1], [0, 0]])
    labels = np.array([0, 1, 0, 1])
    aligned = align_cluster_labels(ref, labels, cand_centroids)
    assert set(aligned) <= {0, 1}
