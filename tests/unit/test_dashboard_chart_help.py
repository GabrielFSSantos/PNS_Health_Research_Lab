"""Testes das chaves de ajuda dos gráficos."""

from __future__ import annotations

from pns_health_research.dashboard.content.chart_help import CHART_HELP

REQUIRED_KEYS = [
    "elbow",
    "pca_variance",
    "pca_scatter",
    "pca_loadings",
    "silhouette",
    "radar_profiles",
    "heatmap_profiles",
    "bubble_usage",
    "cross_analysis_bar",
    "cluster_cross_heatmap",
    "cluster_size_bar",
    "funnel",
    "distribution",
    "roc",
    "pr",
    "calibration",
    "confusion_matrix",
    "importance",
    "importance_native",
    "importance_permutation",
    "survey_variable",
    "gridsearch",
    "metrics_bar",
    "compare_bar",
    "evolution",
]


def test_chart_help_keys_exist():
    for key in REQUIRED_KEYS:
        assert key in CHART_HELP, f"Chave ausente: {key}"
        assert CHART_HELP[key].strip(), f"Texto vazio: {key}"
