"""Testes das funções genéricas de uso por cluster."""

from __future__ import annotations

import pandas as pd

from pns_health_research.analysis.cluster_service_usage import (
    bubble_cross_table,
    bubble_health_service_table,
    cross_rate_table,
    discover_viz_variables,
    rate_by_cluster,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Cluster": [0, 0, 0, 1, 1, 1],
            "Serviço de Saúde": [1, 1, 0, 0, 1, 1],
            "Saúde Bom": [1, 0, 0, 1, 0, 0],
            "Saúde Ruim": [0, 1, 0, 0, 1, 0],
            "Hipertensão": [1, 0, 1, 0, 1, 0],
        }
    )


def test_discover_viz_variables():
    catalog = discover_viz_variables(_sample_df())
    split_keys = {o.key for o in catalog["split"]}
    assert "Serviço de Saúde" in split_keys
    assert "Hipertensão" in split_keys
    category_keys = {o.key for o in catalog["category"]}
    assert "group:Saúde" in category_keys


def test_bubble_cross_table():
    df = _sample_df()
    out = bubble_cross_table(
        df,
        "Serviço de Saúde",
        ["Saúde Bom", "Saúde Ruim"],
    )
    assert not out.empty
    assert {"cluster", "split_label", "category", "count"}.issubset(out.columns)
    assert set(out["split_label"]) <= {"Sim", "Não"}


def test_bubble_health_service_table_legacy_columns():
    df = _sample_df()
    out = bubble_health_service_table(df)
    assert {"servico", "categoria_saude", "count"}.issubset(out.columns)


def test_rate_by_cluster():
    df = _sample_df()
    out = rate_by_cluster(df, "Serviço de Saúde")
    assert len(out) == 2
    assert "rate" in out.columns
    assert out.loc[out["cluster"] == 0, "rate"].iloc[0] == 2 / 3


def test_cross_rate_table():
    df = _sample_df()
    out = cross_rate_table(df, "Serviço de Saúde", ["Saúde Bom", "Saúde Ruim"])
    assert not out.empty
    assert "secondary_category" in out.columns
    assert "rate" in out.columns
