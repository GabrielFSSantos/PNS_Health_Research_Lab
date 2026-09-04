"""Utilitários compartilhados para gráficos Plotly."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pns_health_research.dashboard.config import CLUSTER_COLORS

_chart_key_stack: list[str] = []


def _next_auto_key() -> str:
    if "chart_key_counter" not in st.session_state:
        st.session_state.chart_key_counter = 0
    st.session_state.chart_key_counter += 1
    return f"chart_auto_{st.session_state.chart_key_counter}"


@contextmanager
def chart_context(chart_id: str) -> Iterator[None]:
    _chart_key_stack.append(chart_id)
    try:
        yield
    finally:
        _chart_key_stack.pop()


def current_chart_key() -> str:
    return _chart_key_stack[-1] if _chart_key_stack else _next_auto_key()


def cluster_color(i: int) -> str:
    return CLUSTER_COLORS[i % len(CLUSTER_COLORS)]


def cluster_color_map(clusters: list | pd.Series) -> dict[int, str]:
    unique = sorted({int(c) for c in clusters})
    return {c: cluster_color(i) for i, c in enumerate(unique)}


def cluster_label(c: int | float | str) -> str:
    return f"Cluster {int(c)}"


def resolve_cluster_column(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    if "cluster" in df.columns:
        return "cluster"
    if "Cluster" in df.columns:
        return "Cluster"
    return None


def as_cluster_category(df: pd.DataFrame, col: str = "cluster") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].astype(int)
        out["cluster_label"] = out[col].map(lambda c: cluster_label(c))
    return out


def apply_cluster_axis(fig: go.Figure, axis: str = "x", df: pd.DataFrame | None = None, col: str = "cluster") -> None:
    if df is not None and col in df.columns:
        categories = [cluster_label(c) for c in sorted(df[col].astype(int).unique())]
        fig.update_layout(**{f"{axis}axis": dict(type="category", categoryorder="array", categoryarray=categories)})
    else:
        fig.update_layout(**{f"{axis}axis": dict(type="category", dtick=1)})


def add_bar_value_labels(fig: go.Figure, fmt: str = "%{y:,.0f}", position: str = "outside") -> None:
    fig.update_traces(texttemplate=fmt, textposition=position)


def format_percent(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def format_count(value: int | float) -> str:
    return f"{int(value):,}".replace(",", ".")
