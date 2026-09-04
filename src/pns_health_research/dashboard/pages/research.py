"""Trilha científica e documentação."""

from __future__ import annotations

import streamlit as st

from pns_health_research.dashboard.components.filters import init_session_state
from pns_health_research.dashboard.components.insight_panel import render_insights
from pns_health_research.dashboard.components.layout import page_header, section_header
from pns_health_research.dashboard.config import DOCS_DIR, PROJECT_ROOT
from pns_health_research.dashboard.insights import engine as insights_engine
from pns_health_research.dashboard.services import catalog


def render() -> None:
    init_session_state()
    page_header(
        "Pesquisa",
        "Trilha científica, documentação e execuções disponíveis no dashboard.",
    )

    runs = catalog.list_runs()

    render_insights(
        insights_engine.generate(
            {"page": "research", "total_runs": len(runs)}
        )
    )

    stats = catalog.dashboard_stats()
    st.metric("Execuções visíveis", stats.get("total_runs", 0))
    c1, c2 = st.columns(2)
    c1.metric("Bases modulares", stats.get("modular_bases", 0))
    c2.metric("Showcases", stats.get("showcase_count", 0))

    doc = DOCS_DIR / "documentacao.md"
    if doc.exists():
        st.markdown(
            f"**Como rodar novas execuções:** consulte "
            f"[documentacao.md]({doc.as_uri()}) ou execute `./scripts/run_experiment.sh`."
        )

    section_header("Trilha da pesquisa")
    steps = [
        ("1. Dados PNS", "Download e validação dos microdados 2019"),
        ("2. Pré-processamento", "Bases binária (agrupamento) e categórica (classificação)"),
        ("3. Agrupamento", "K-Means + PCA, perfis e análise de silhueta"),
        ("4. Classificação", "Validação cruzada, Grid Search, métricas F1+/recall"),
        ("5. Dashboard", "Visualização e comparação das runs concluídas"),
    ]
    for title, desc in steps:
        st.markdown(f"**{title}** — {desc}")

    section_header("Execuções disponíveis")
    if not runs:
        st.caption("Nenhuma execução concluída. Rode `./scripts/run_experiment.sh`.")
    for r in runs:
        role = "baseline" if r.role == "baseline" else "candidata"
        pm = r.primary_metric or {}
        f1 = pm.get("value")
        f1_txt = f" — F1+ {f1:.1%}" if f1 is not None else ""
        origem = {"base": "base modular", "showcase": "showcase", "runs": "run legada"}.get(
            r.source_type, r.source
        )
        layer = f" · camada {r.layer}" if r.layer else ""
        subtitle = f" — {r.subtitle}" if r.subtitle else ""
        st.markdown(f"- **{r.display_name}** ({role}, {origem}{layer}){f1_txt}")
        st.caption(f"`{r.query_id}`{subtitle}")

    section_header("Documentação")
    doc_links = [
        ("Índice da documentação", DOCS_DIR / "README.md"),
        ("Documentação técnica", DOCS_DIR / "documentacao.md"),
        ("Trajetória da pesquisa", DOCS_DIR / "trajetoria.md"),
        ("Inventário de gráficos", DOCS_DIR / "dashboard" / "chart_inventory.md"),
        ("Monografia (TCC)", DOCS_DIR / "research" / "monografia.pdf"),
    ]
    for label, path in doc_links:
        if path.exists():
            st.markdown(f"- [{label}]({path.as_uri()})")
        else:
            st.markdown(f"- {label} _(ausente: `{path.name}`)_")

    st.caption(f"Projeto: `{PROJECT_ROOT}`")
