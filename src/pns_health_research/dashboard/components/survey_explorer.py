"""Explorador de variáveis da pesquisa (pré-one-hot)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from pns_health_research.dashboard.components.chart_utils import current_chart_key


def render_survey_variable_explorer(snapshot: dict[str, Any]) -> None:
    variables = snapshot.get("variables") or {}
    if not variables:
        st.info(
            "Snapshot da pesquisa indisponível nesta execução. "
            "Reexecute `./scripts/run_preprocess.sh --profile pns` (ou `legacy`) "
            "para gerar `survey_snapshot.json` em `data/bases/<nome>/preprocessing/metrics/`."
        )
        return

    names = sorted(variables.keys())
    selected = st.selectbox("Variável da pesquisa", names, format_func=lambda x: x)
    var = variables[selected]
    question = var.get("question") or "Pergunta não catalogada."
    code = var.get("code")
    st.markdown(f"**Pergunta:** {question}")
    if code:
        st.caption(f"Código IBGE: `{code}` · Tipo: {var.get('type', '—')}")

    value_labels = var.get("value_labels") or {}
    if value_labels:
        with st.expander("Respostas possíveis", expanded=False):
            for raw, label in value_labels.items():
                st.markdown(f"- **{raw}** — {label}")

    counts = var.get("counts") or {}
    missing = int(var.get("missing") or 0)
    total = int(var.get("total") or 0)
    if missing:
        st.caption(f"Ausentes na amostra filtrada: **{missing:,}** de **{total:,}**".replace(",", "."))

    if not counts:
        st.caption("Sem contagens para esta variável.")
        return

    df = pd.DataFrame(
        [{"resposta": k, "contagem": v} for k, v in counts.items()]
    ).sort_values("contagem", ascending=True)
    fig = px.bar(
        df,
        x="contagem",
        y="resposta",
        orientation="h",
        title=f"Distribuição — {selected}",
        text="contagem",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(height=max(320, 28 * len(df)), margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, width="stretch", key=current_chart_key())
