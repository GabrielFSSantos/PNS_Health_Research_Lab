"""Testes de utilitários de gráficos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pns_health_research.dashboard.components.chart_utils import (
    as_cluster_category,
    cluster_color_map,
    cluster_label,
    format_percent,
)


def test_cluster_label():
    assert cluster_label(3) == "Cluster 3"


def test_cluster_color_map_distinct():
    colors = cluster_color_map([0, 1, 2])
    assert len(set(colors.values())) == 3
    assert all(isinstance(k, int) for k in colors)


def test_cluster_color_map_numpy_int64_lookup():
    colors = cluster_color_map([0, 1, 2])
    assert colors[np.int64(0)] == colors[0]


def test_as_cluster_category():
    df = pd.DataFrame({"cluster": [0, 1, 2]})
    out = as_cluster_category(df)
    assert list(out["cluster_label"]) == ["Cluster 0", "Cluster 1", "Cluster 2"]


def test_format_percent():
    assert format_percent(0.792) == "79.2%"
