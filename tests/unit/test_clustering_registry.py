
from pns_health_research.clustering.registry import CLUSTERER_REGISTRY, get_clusterer


def test_kmeans_registered():
    assert "kmeans" in CLUSTERER_REGISTRY
    fn = get_clusterer("kmeans")
    assert callable(fn)
