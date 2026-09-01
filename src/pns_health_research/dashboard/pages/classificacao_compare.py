"""Comparação de classificação entre execuções."""

from __future__ import annotations

import streamlit as st

from pns_health_research.dashboard.components import charts
from pns_health_research.dashboard.components.chart_block import chart_block
from pns_health_research.dashboard.components.filters import init_session_state
from pns_health_research.dashboard.components.insight_panel import render_insights
from pns_health_research.dashboard.components.kpi_cards import kpi_row
from pns_health_research.dashboard.components.layout import page_header, section_header
from pns_health_research.dashboard.components.states import empty_state
from pns_health_research.dashboard.content.pt_br import model_label
from pns_health_research.dashboard.insights import engine as insights_engine
from pns_health_research.dashboard.services import catalog, compare


def render() -> None:
    init_session_state()
    page_header(
        "Comparar classificação",
        "Ranking por F1+, recall e ROC-AUC entre execuções de referência e modelos.",
    )

    runs = catalog.list_runs()
    if len(runs) < 1:
        empty_state("Nenhuma execução disponível.")
        return

    options = {r.display_name: r.query_id for r in runs}
    default = list(options.keys())[: min(3, len(options))]
    selected_labels = st.multiselect("Execuções", list(options.keys()), default=default)
    selected = [options[lbl] for lbl in selected_labels]
    if not selected:
        st.warning("Selecione ao menos uma execução.")
        return

    df = compare.classification_compare_df(selected)
    if df.empty:
        empty_state("Sem métricas de classificação para as execuções selecionadas.")
        return

    if "name" in df.columns:
        name_map = {r.query_id: r.display_name for r in runs}
        df = df.copy()
        df["name"] = df["run_id"].map(name_map).fillna(df["name"])
    if "model" in df.columns:
        df = df.copy()
        df["model_label"] = df["model"].map(model_label)

    best_f1 = None
    if "f1_positive" in df.columns:
        idx = df["f1_positive"].idxmax()
        row = df.loc[idx]
        m = row.get("model_label") or row.get("model")
        best_f1 = f"{m} ({row.get('name')}) — F1+ {row['f1_positive']:.1%}"

    render_insights(
        insights_engine.generate({"page": "classification_compare", "best_model": best_f1})
    )

    if "f1_positive" in df.columns:
        best_recall = df["recall_positive"].max() if "recall_positive" in df.columns else None
        best_auc = df["roc_auc"].max() if "roc_auc" in df.columns else None
        kpi_row(
            [
                ("Melhor F1+", f"{df['f1_positive'].max():.1%}", None),
                (
                    "Melhor recall+",
                    f"{best_recall:.1%}" if best_recall is not None else "—",
                    None,
                ),
                ("Melhor ROC-AUC", f"{best_auc:.3f}" if best_auc is not None else "—", None),
            ]
        )

    section_header("Tabela comparativa")
    show = df.copy()
    if "model_label" in show.columns:
        show = show.drop(columns=["model"], errors="ignore")
        show = show.rename(columns={"model_label": "modelo"})
    st.dataframe(show, width="stretch")

    if "f1_positive" in df.columns and "model" in df.columns:
        plot_df = df.copy()
        plot_df["model"] = plot_df["model"].map(model_label)
        chart_block(
            "compare_bar",
            "F1+ por modelo e execução",
            lambda: charts.grouped_metric_bar(
                plot_df,
                x="model",
                y="f1_positive",
                color="name",
                title="F1+ por modelo e execução",
                as_percent=True,
            ),
            chart_id="clf_cmp_f1",
        )
