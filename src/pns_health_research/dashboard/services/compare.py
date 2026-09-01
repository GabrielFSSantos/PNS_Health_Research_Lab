"""Comparação entre runs."""

from __future__ import annotations

import pandas as pd

from pns_health_research.web import query


def compare_clustering(runs: list[str]) -> dict:
    return query.compare_clustering(runs)


def compare_classification(runs: list[str]) -> dict:
    return query.compare_classification(runs)


def classification_compare_df(runs: list[str]) -> pd.DataFrame:
    data = compare_classification(runs)
    rows = data.get("comparison") or []
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def clustering_compare_df(runs: list[str]) -> pd.DataFrame:
    data = compare_clustering(runs)
    rows = data.get("comparison") or []
    return pd.DataFrame(rows) if rows else pd.DataFrame()
