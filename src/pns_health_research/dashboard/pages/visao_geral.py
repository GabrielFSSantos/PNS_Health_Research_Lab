"""Visão geral da pesquisa PNS."""

from __future__ import annotations

import streamlit as st

from pns_health_research.dashboard.components.filters import init_session_state
from pns_health_research.dashboard.components.insight_panel import render_insights
from pns_health_research.dashboard.components.kpi_cards import kpi_row
from pns_health_research.dashboard.components.layout import page_header, section_header
from pns_health_research.dashboard.insights import engine as insights_engine
from pns_health_research.dashboard.services import catalog


def render() -> None:
    init_session_state()
    page_header(
        "Visão Geral",
        "Resumo das execuções de referência: baseline do TCC e candidatas com metodologia revisada.",
    )

    stats = catalog.dashboard_stats()
    kpi_row(
        [
            ("PNS", stats.get("pns_year", 2019), None),
            ("Execuções visíveis", stats.get("total_runs", 0), None),
            ("Baseline", "Legacy (TCC)", None),
            ("Melhor F1+", _fmt_f1(stats.get("best_classification")), None),
        ]
    )

    missing = []
    for r in catalog.list_runs():
        if r.status != "completed":
            missing.append(r.display_name)

    render_insights(
        insights_engine.generate(
            {
                "page": "overview",
                "total_runs": stats.get("total_runs"),
                "best_classification": _run_to_dict(stats.get("best_classification")),
                "artifacts_missing": missing,
            }
        )
    )

    section_header("Pipeline da pesquisa")
    st.markdown(
        """
1. **Microdados PNS 2019** — download e validação  
2. **Pré-processamento** — bases binária (agrupamento) e categórica (classificação)  
3. **Agrupamento (clustering)** — K-Means + PCA  
4. **Classificação** — modelos com validação cruzada  
5. **Artifacts** — métricas e tabelas versionadas por execução
"""
    )

    section_header("Execuções de referência")
    for r in stats.get("recent_runs") or []:
        pm = r.primary_metric or {}
        f1 = pm.get("value")
        f1_txt = f" — F1+ {f1:.1%}" if f1 is not None else ""
        role = "baseline" if r.role == "baseline" else "candidata"
        st.markdown(f"- **{r.display_name}** ({role}){f1_txt}")


def _fmt_f1(best) -> str:
    if best is None:
        return "—"
    if hasattr(best, "primary_metric"):
        pm = best.primary_metric or {}
        val = pm.get("value")
        return f"{val:.1%}" if val is not None else "—"
    pm = best.get("primary_metric") or {}
    val = pm.get("value")
    return f"{val:.1%}" if val is not None else "—"


def _run_to_dict(r) -> dict | None:
    if r is None:
        return None
    if hasattr(r, "display_name"):
        return {
            "name": r.display_name,
            "primary_metric": r.primary_metric,
        }
    return r
