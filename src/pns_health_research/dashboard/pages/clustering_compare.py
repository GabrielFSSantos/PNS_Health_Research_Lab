"""Comparação de agrupamento entre execuções."""

from __future__ import annotations

import streamlit as st

from pns_health_research.dashboard.components import charts
from pns_health_research.dashboard.components.chart_block import chart_block
from pns_health_research.dashboard.components.filters import init_session_state
from pns_health_research.dashboard.components.insight_panel import render_insights
from pns_health_research.dashboard.components.layout import page_header, section_header
from pns_health_research.dashboard.components.states import empty_state
from pns_health_research.dashboard.content import pt_br
from pns_health_research.dashboard.insights import engine as insights_engine
from pns_health_research.dashboard.services import catalog, compare


def render() -> None:
    init_session_state()
    page_header(
        "Comparar agrupamento",
        "Métricas lado a lado com alinhamento Hungarian quando centróides existem.",
    )

    runs = catalog.list_runs()
    if len(runs) < 2:
        empty_state("É necessário ter pelo menos 2 execuções curadas para comparar.")
        return

    options = {r.display_name: r.query_id for r in runs}
    selected_labels = st.multiselect(
        "Execuções",
        list(options.keys()),
        default=list(options.keys())[: min(2, len(options))],
    )
    selected = [options[lbl] for lbl in selected_labels]
    if len(selected) < 2:
        st.warning("Selecione pelo menos 2 execuções.")
        return

    result = compare.compare_clustering(selected)
    render_insights(
        insights_engine.generate(
            {
                "page": "clustering_compare",
                "alignment_warning": result.get("alignment_warning"),
            }
        )
    )

    df = compare.clustering_compare_df(selected)
    if not df.empty:
        name_map = {r.query_id: r.display_name for r in runs}
        df = df.copy()
        if "name" in df.columns:
            df["name"] = df["run_id"].map(name_map).fillna(df["name"])
        section_header("Métricas")
        st.dataframe(df, width="stretch")
        if "silhouette" in df.columns and "name" in df.columns:
            chart_block(
                "compare_bar",
                "Coeficiente de silhueta por execução",
                lambda: charts.grouped_metric_bar(
                    df,
                    x="name",
                    y="silhouette",
                    color="run_id",
                    title="Silhueta por execução",
                ),
                chart_id="cluster_cmp_silhouette",
            )

    alignment = result.get("alignment") or {}
    if alignment.get("mapping"):
        with st.expander(pt_br.TECHNICAL_DETAILS):
            st.json(alignment["mapping"])
