"""Testes de comparação dashboard."""

from __future__ import annotations

from pns_health_research.dashboard.services import compare


def test_compare_clustering_single_run():
    result = compare.compare_clustering(["pesquisa-original-legacy"])
    assert "comparison" in result


def test_classification_compare_df():
    df = compare.classification_compare_df(["pesquisa-original-legacy"])
    assert not df.empty
    assert "model" in df.columns
