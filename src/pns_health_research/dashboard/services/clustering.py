"""Serviços de clustering."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from pns_health_research.analysis.cluster_service_usage import (
    bubble_cross_table,
    cross_rate_table,
    discover_viz_variables,
    rate_by_cluster,
)
from pns_health_research.web import query


@lru_cache(maxsize=64)
def load_clustering_data(run_id: str) -> dict:
    return query.get_clusters(run_id)


def load_elbow(run_id: str) -> pd.DataFrame:
    data = load_clustering_data(run_id)
    rows = data.get("tables", {}).get("elbow_results", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_pca_variance(run_id: str) -> pd.DataFrame:
    data = load_clustering_data(run_id)
    rows = data.get("tables", {}).get("pca_explained_variance", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_pca_loadings(run_id: str) -> pd.DataFrame:
    data = load_clustering_data(run_id)
    rows = data.get("tables", {}).get("pca_loadings", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_pca_scatter(run_id: str) -> pd.DataFrame:
    data = load_clustering_data(run_id)
    rows = data.get("tables", {}).get("pca_scatter_sample", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_profiles(run_id: str) -> pd.DataFrame:
    data = load_clustering_data(run_id)
    rows = data.get("tables", {}).get("cluster_profiles", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_cross_analysis(run_id: str) -> pd.DataFrame:
    data = load_clustering_data(run_id)
    rows = data.get("tables", {}).get("cross_analysis", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_bubble_health(run_id: str) -> pd.DataFrame:
    data = load_clustering_data(run_id)
    rows = data.get("tables", {}).get("bubble_health_service", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_clustered_binary(run_id: str) -> pd.DataFrame:
    data = load_clustering_data(run_id)
    rows = data.get("tables", {}).get("clustered_binary", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@lru_cache(maxsize=64)
def get_viz_variable_catalog(run_id: str) -> dict:
    df = load_clustered_binary(run_id)
    if df.empty:
        return {"split": [], "category": []}
    catalog = discover_viz_variables(df)
    return {
        "split": catalog["split"],
        "category": catalog["category"],
    }


def build_bubble_df(
    run_id: str,
    split_col: str,
    category_cols: tuple[str, ...],
) -> pd.DataFrame:
    df = load_clustered_binary(run_id)
    if df.empty:
        return pd.DataFrame()
    return bubble_cross_table(df, split_col, category_cols)


def build_rate_df(run_id: str, variable_col: str) -> pd.DataFrame:
    df = load_clustered_binary(run_id)
    if not df.empty and variable_col in df.columns:
        return rate_by_cluster(df, variable_col)
    profiles = load_profiles(run_id)
    if profiles.empty or variable_col not in profiles.columns:
        return pd.DataFrame()
    rows = []
    cluster_col = "Cluster" if "Cluster" in profiles.columns else profiles.columns[0]
    for _, row in profiles.iterrows():
        cluster = int(row[cluster_col])
        rows.append(
            {
                "cluster": cluster,
                "n": None,
                "rate": float(row[variable_col]),
                "service_usage_rate": float(row[variable_col]),
                "variable": variable_col,
                "weighted": False,
                "from_profiles": True,
            }
        )
    return pd.DataFrame(rows)


def build_cross_rate_df(
    run_id: str,
    primary_col: str,
    secondary_cols: tuple[str, ...],
) -> pd.DataFrame:
    df = load_clustered_binary(run_id)
    if df.empty:
        return pd.DataFrame()
    return cross_rate_table(df, primary_col, secondary_cols)


def load_silhouette_samples(run_id: str) -> pd.DataFrame:
    data = load_clustering_data(run_id)
    rows = data.get("silhouette_samples", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_silhouette_summary(run_id: str) -> dict:
    data = load_clustering_data(run_id)
    return data.get("silhouette_summary") or {}


def load_clustering_metrics(run_id: str) -> dict:
    data = load_clustering_data(run_id)
    return data.get("metrics") or {}


def resolve_best_k(metrics: dict) -> int | None:
    val = metrics.get("best_k") or metrics.get("n_clusters")
    return int(val) if val is not None else None
