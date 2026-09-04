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
from pns_health_research.dashboard.services import catalog, dataset

MIN_DISTRIBUTION_VARS = 5


def _run_context_caption(run_id: str) -> str | None:
    for run in catalog.list_runs():
        if run.query_id == run_id:
            parts: list[str] = []
            if run.layer:
                parts.append(run.layer)
            if run.profile_kind:
                parts.append(f"perfil `{run.profile_kind}`")
            if run.subtitle:
                parts.append(run.subtitle)
            return " · ".join(parts) if parts else None
    return None


def _preprocess_command(run_id: str) -> str:
    if run_id in ("legacy", "pns"):
        return f"./scripts/run_experiment.sh --profile {run_id}"
    return "./scripts/run_experiment.sh --profile pns"


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
    prep_summary = dataset.load_preprocessing_summary(run_id)

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
        ctx = _run_context_caption(run_id)
        if ctx:
            st.caption(ctx)

        render_insights(
            insights_engine.generate(
                {"page": "dataset", "has_fingerprint": bool(fingerprint), "tab": "processed"}
            )
        )

        if not fingerprint and not funnel.get("stages"):
            empty_state(
                f"Pré-processamento não materializado para esta execução. "
                f"Execute `{_preprocess_command(run_id)}`."
            )
        else:
            if fingerprint:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Linhas", dataset.format_fingerprint_metric(fingerprint.get("rows")))
                c2.metric(
                    "Colunas binárias",
                    dataset.format_fingerprint_metric(fingerprint.get("columns")),
                )
                c3.metric(
                    "Colunas categóricas",
                    dataset.format_fingerprint_metric(fingerprint.get("categorical_columns")),
                )
                hash_val = str(fingerprint.get("hash", ""))
                c4.metric("Hash", hash_val[:12] + "…" if hash_val else "—")

            if distributions and len(distributions) < MIN_DISTRIBUTION_VARS:
                st.warning(
                    "Distribuições parciais — execute "
                    f"`{_preprocess_command(run_id)}` "
                    "para regenerar `data/bases/<nome>/preprocessing/`."
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
            prep = info.get("preprocessing") or prep_summary
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
