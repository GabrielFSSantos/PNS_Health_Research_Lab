"""PNS Health Research Lab — Dashboard Streamlit."""

from __future__ import annotations

import streamlit as st

from pns_health_research.dashboard.config import THEME_CSS
from pns_health_research.dashboard.content import pt_br
from pns_health_research.dashboard.pages import (
    classificacao,
    classificacao_compare,
    clustering,
    clustering_compare,
    dataset,
    evolucao,
    research,
    visao_geral,
)


def _load_css() -> None:
    if THEME_CSS.is_file():
        st.markdown(
            f"<style>{THEME_CSS.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="PNS Health Research Lab",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _load_css()

    with st.sidebar:
        st.markdown(
            '<p class="pns-sidebar-title">PNS Health Research Lab</p>',
            unsafe_allow_html=True,
        )

    pages = [
        st.Page(visao_geral.render, title="Visão Geral", icon="🏠", url_path="visao-geral", default=True),
        st.Page(dataset.render, title=pt_br.NAV_DATASET, icon="📊", url_path="dataset"),
        st.Page(clustering.render, title=pt_br.NAV_CLUSTERING, icon="🔬", url_path="clustering"),
        st.Page(
            clustering_compare.render,
            title=pt_br.NAV_CLUSTERING_CMP,
            icon="⚖️",
            url_path="clustering-compare",
        ),
        st.Page(classificacao.render, title=pt_br.NAV_CLASSIFICATION, icon="🤖", url_path="classificacao"),
        st.Page(
            classificacao_compare.render,
            title=pt_br.NAV_CLASSIFICATION_CMP,
            icon="📈",
            url_path="classificacao-compare",
        ),
        st.Page(evolucao.render, title=pt_br.NAV_EVOLUTION, icon="📅", url_path="evolucao"),
        st.Page(research.render, title=pt_br.NAV_RESEARCH, icon="📚", url_path="research"),
    ]

    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()
