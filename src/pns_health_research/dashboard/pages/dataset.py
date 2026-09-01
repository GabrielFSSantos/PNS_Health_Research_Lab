"""Conjunto de dados, dicionário e pré-processamento."""

from __future__ import annotations

import streamlit as st

from pns_health_research.dashboard.components import charts
from pns_health_research.dashboard.components.chart_block import chart_block
from pns_health_research.dashboard.components.filters import init_session_state, run_selector
from pns_health_research.dashboard.components.insight_panel import render_insights
from pns_health_research.dashboard.components.layout import page_header
from pns_health_research.dashboard.components.states import empty_state
from pns_health_research.dashboard.components.survey_explorer import render_survey_variable_explorer
from pns_health_research.dashboard.content import pt_br
from pns_health_research.dashboard.insights import engine as insights_engine
from pns_health_research.dashboard.services import dataset

MIN_DISTRIBUTION_VARS = 5


def render() -> None:
    init_session_state()
    page_header(
        "Conjunto de dados",
        "Variáveis da pesquisa, pré-processamento e dicionário técnico.",
    )

    run_id = run_selector("Execução de referência")
    if not run_id:
        empty_state("Nenhuma execução disponível no catálogo.")
        return

    tab_survey, tab_post, tab_dict = st.tabs(
        [pt_br.TAB_SURVEY, pt_br.TAB_POST_PREPROCESS, pt_br.TAB_DICTIONARY]
    )

    fingerprint = dataset.load_fingerprint(run_id)
    distributions = dataset.load_distributions(run_id)
    funnel = dataset.load_funnel(run_id)
    survey_snapshot = dataset.load_survey_snapshot(run_id)

    with tab_survey:
        render_insights(
            insights_engine.generate(
                {"page": "dataset", "has_fingerprint": bool(fingerprint), "tab": "survey"}
            )
        )
        chart_block(
            "survey_variable",
            "Explorador de variáveis da pesquisa",
            lambda: render_survey_variable_explorer(survey_snapshot),
            chart_id="dataset_survey_explorer",
        )

    with tab_post:
        render_insights(
            insights_engine.generate(
                {"page": "dataset", "has_fingerprint": bool(fingerprint), "tab": "processed"}
            )
        )
        if fingerprint:
            c1, c2, c3 = st.columns(3)
            c1.metric("Linhas", dataset.format_fingerprint_metric(fingerprint.get("rows")))
            c2.metric("Colunas", dataset.format_fingerprint_metric(fingerprint.get("columns")))
            c3.metric("Hash", str(fingerprint.get("hash", ""))[:12] + "…")
        if distributions and len(distributions) < MIN_DISTRIBUTION_VARS:
            st.warning(
                "Distribuições parciais neste showcase — execute "
                "`python scripts/backfill_showcase_distributions.py` ou reexporte o showcase."
            )
        chart_block(
            "distribution",
            "Distribuições após pré-processamento",
            lambda: charts.distribution_plot(distributions),
            chart_id="dataset_distribution",
        )
        chart_block(
            "funnel",
            "Funil de pré-processamento",
            lambda: charts.preprocess_funnel(funnel),
            chart_id="dataset_funnel",
        )
        info = dataset.load_dataset_info(run_id)
        prep = info.get("preprocessing")
        if prep:
            with st.expander(pt_br.TECHNICAL_DETAILS):
                st.json(prep)

    with tab_dict:
        df = dataset.load_dictionary_df()
        if df.empty:
            empty_state("Dicionário vazio.")
        else:
            search = st.text_input("Buscar coluna", "")
            show = df
            if search and "internal_name" in df.columns:
                mask = df["internal_name"].str.contains(search, case=False, na=False)
                if "friendly_name" in df.columns:
                    mask |= df["friendly_name"].str.contains(search, case=False, na=False)
                show = df[mask]
            st.dataframe(show, width="stretch", height=400)
            if "internal_name" in show.columns and len(show):
                chosen = st.selectbox("Detalhe da coluna", show["internal_name"].tolist())
                col = dataset.load_dictionary_column(chosen)
                with st.expander(f"Coluna: {chosen}", expanded=True):
                    st.json(col)
