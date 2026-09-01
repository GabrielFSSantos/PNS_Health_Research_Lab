"""Testes do catálogo de métricas do dashboard."""

from __future__ import annotations

from pns_health_research.dashboard.metrics.catalog import METRICS, get_metric, interpret_value

REQUIRED_METRIC_KEYS = [
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "f1_positive",
    "recall_positive",
    "precision_positive",
    "roc_auc",
    "pr_auc",
    "n_samples",
    "n_positive",
    "positive_rate",
]


def test_all_required_metrics_exist():
    for key in REQUIRED_METRIC_KEYS:
        assert key in METRICS, f"Métrica ausente no catálogo: {key}"


def test_metric_descriptions_are_rich():
    for key in REQUIRED_METRIC_KEYS:
        info = get_metric(key)
        assert info is not None
        assert len(info.description) > 80, f"Descrição curta demais: {key}"
        assert "**O que é" in info.description


def test_interpret_value_with_bands():
    text = interpret_value("f1_positive", 0.75)
    assert "bom" in text.lower()


def test_interpret_value_without_bands():
    assert interpret_value("n_samples", 1000) == ""
