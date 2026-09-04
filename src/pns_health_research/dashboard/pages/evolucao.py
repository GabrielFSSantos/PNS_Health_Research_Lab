"""Evolução entre execuções de referência."""

from __future__ import annotations

import streamlit as st

from pns_health_research.dashboard.components import charts
from pns_health_research.dashboard.components.chart_block import chart_block
from pns_health_research.dashboard.components.filters import init_session_state
from pns_health_research.dashboard.components.insight_panel import render_insights
from pns_health_research.dashboard.components.layout import page_header, section_header
from pns_health_research.dashboard.config import SHOWCASE_DIR
from pns_health_research.dashboard.insights import engine as insights_engine
from pns_health_research.dashboard.services import catalog


def render() -> None:
    init_session_state()
    page_header(
        "Evolução",
        "Comparação da métrica primária (F1+) entre baseline do TCC e execuções candidatas.",
    )

    runs = catalog.list_runs()
    candidates = [r for r in runs if r.role == "candidate"]

    render_insights(
        insights_engine.generate(
            {
                "page": "evolution",
                "showcase_count": len(runs),
                "missing_candidates": [c.display_name for c in candidates if c.status != "completed"],
            }
        )
    )

    if len(runs) < 2:
        st.info(
            "Execute mais de uma run concluída para ver a evolução comparativa no gráfico."
        )

    labels: list[str] = []
    values: list[float] = []
    rows = []
    baseline_f1 = catalog.get_baseline_f1()

    for r in runs:
        pm = r.primary_metric or {}
        val = pm.get("value")
        role = "baseline" if r.role == "baseline" else "candidata"
        delta = None
        if val is not None and baseline_f1 is not None and r.role != "baseline":
            delta = float(val) - baseline_f1
        rows.append(
            {
                "execução": r.display_name,
                "papel": role,
                "F1+": f"{val:.1%}" if val is not None else "—",
                "Δ vs baseline": f"{delta:+.1%}" if delta is not None else "—",
                "status": r.status,
            }
        )
        if val is not None:
            labels.append(r.display_name)
            values.append(float(val))

    if values:
        chart_block(
            "evolution",
            "Evolução do F1+",
            lambda: charts.metric_bar(labels, values, title="F1+ — execuções de referência"),
            chart_id="evolution_f1",
        )

    section_header("Resumo")
    st.dataframe(rows, width="stretch")

    if SHOWCASE_DIR.exists():
        st.caption(
            f"Bases modulares: `data/bases/` · Showcases: `{SHOWCASE_DIR}`"
        )
