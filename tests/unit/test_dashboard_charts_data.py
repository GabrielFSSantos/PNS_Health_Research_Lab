"""Testes de gráficos com dados do showcase legacy."""

from __future__ import annotations

import pandas as pd

from pns_health_research.dashboard.services import clustering

RUN_ID = "pesquisa-original-legacy"


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
