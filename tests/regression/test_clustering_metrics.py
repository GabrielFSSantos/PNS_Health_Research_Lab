"""Regressão de métricas de agrupamento com PNS real."""

from __future__ import annotations

import pytest


@pytest.mark.regression
def test_legacy_profile_clustering_metrics(regression_bases):
    metrics = regression_bases["clustering"]["legacy"].result.metrics
    assert metrics["n_clusters"] == 6
    assert metrics["rows_out"] == 87678
    assert abs(metrics["silhouette"] - 0.513881101838751) < 0.005
    assert abs(metrics["wcss"] - 37821.75) / 37821.75 < 0.01


@pytest.mark.regression
def test_pns_profile_clustering_metrics(regression_bases):
    metrics = regression_bases["clustering"]["pns"].result.metrics
    assert metrics["n_clusters"] == 6
    assert metrics["rows_out"] == 87678
    assert metrics["silhouette"] > 0
