"""Testes de clustering dashboard."""

from __future__ import annotations

from pns_health_research.dashboard.services import clustering

RUN_ID = "pesquisa-original-legacy"


def test_load_elbow_not_empty():
    df = clustering.load_elbow(RUN_ID)
    assert not df.empty
    assert "k" in df.columns


def test_load_pca_variance():
    df = clustering.load_pca_variance(RUN_ID)
    assert not df.empty


def test_load_profiles():
    df = clustering.load_profiles(RUN_ID)
    assert not df.empty
    assert "Cluster" in df.columns or "cluster" in df.columns


def test_silhouette_summary():
    summary = clustering.load_silhouette_summary(RUN_ID)
    assert summary


def test_resolve_best_k_from_n_clusters():
    metrics = clustering.load_clustering_metrics(RUN_ID)
    k = clustering.resolve_best_k(metrics)
    assert k == 6
