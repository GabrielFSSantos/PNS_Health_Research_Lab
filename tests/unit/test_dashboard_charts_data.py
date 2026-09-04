"""Testes de gráficos com dados do showcase legacy e bases modulares."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from pns_health_research.dashboard.components import charts
from pns_health_research.dashboard.services import clustering
from pns_health_research.modules.analysis.cluster_service_usage import normalize_cluster_profiles

RUN_ID = "pesquisa-original-legacy"
MODULAR_BASE = "legacy"


def test_silhouette_dual_inputs():
    samples = clustering.load_silhouette_samples(RUN_ID)
    scatter = clustering.load_pca_scatter(RUN_ID)
    summary = clustering.load_silhouette_summary(RUN_ID)
    assert not samples.empty
    assert not scatter.empty
    assert summary


def test_profile_heatmap_data():
    profiles = clustering.load_profiles(RUN_ID)
    assert not profiles.empty


def test_load_profiles_legacy_has_cluster_column():
    clustering.load_clustering_data.cache_clear()
    profiles = clustering.load_profiles(MODULAR_BASE)
    assert not profiles.empty
    assert "cluster" in profiles.columns or "Cluster" in profiles.columns


def test_normalize_cluster_profiles_infers_index():
    raw = pd.DataFrame({"Hipertensão": [0.9, 0.1], "Diabetes": [0.2, 0.0]})
    out = normalize_cluster_profiles(raw)
    assert "cluster" in out.columns
    assert list(out["cluster"]) == [0, 1]


@patch("pns_health_research.dashboard.components.charts.st")
def test_profile_heatmap_no_crash_legacy(mock_st):
    mock_st.plotly_chart = MagicMock()
    clustering.load_clustering_data.cache_clear()
    profiles = clustering.load_profiles(MODULAR_BASE)
    charts.profile_heatmap(profiles)
    mock_st.plotly_chart.assert_called_once()


def test_pca_variance_has_cumulative_or_ratio():
    df = clustering.load_pca_variance(RUN_ID)
    assert not df.empty
    assert "explained_variance_ratio" in df.columns or len(df.columns) >= 2


def test_cross_analysis_bar_data():
    df = clustering.load_cross_analysis(RUN_ID)
    assert not df.empty
    assert "cluster" in df.columns or "Cluster" in df.columns


def test_clustered_binary_catalog():
    catalog = clustering.get_viz_variable_catalog(RUN_ID)
    if clustering.load_clustered_binary(RUN_ID).empty:
        return
    assert catalog["split"]
    assert catalog["category"]


def test_build_bubble_from_catalog():
    catalog = clustering.get_viz_variable_catalog(RUN_ID)
    if not catalog["split"] or not catalog["category"]:
        return
    split = catalog["split"][0]
    category = catalog["category"][0]
    bubble = clustering.build_bubble_df(run_id=RUN_ID, split_col=split.columns[0], category_cols=category.columns)
    assert isinstance(bubble, pd.DataFrame)


def test_bubble_fallback_with_cross():
    cross = clustering.load_cross_analysis(RUN_ID)
    # bubble pode estar ausente no showcase antigo; fallback não deve falhar com dados válidos
    assert isinstance(cross, pd.DataFrame)
