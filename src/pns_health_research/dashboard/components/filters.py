"""Filtros e seletores."""

from __future__ import annotations

import streamlit as st

from pns_health_research.dashboard.services import catalog


def init_session_state() -> None:
    if "selected_run_id" not in st.session_state:
        runs = catalog.list_runs()
        if runs:
            st.session_state.selected_run_id = runs[0].query_id
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "random_forest"


def _run_option_label(run: catalog.RunSummary) -> str:
    prefix = f"[{run.layer}] " if run.layer else ""
    if run.showcase_label:
        return f"{prefix}{run.display_name}"
    short_id = run.run_id.split("_")[-1][:8] if "_" in run.run_id else run.run_id[:16]
    if run.display_name and run.display_name != run.run_id:
        return f"{prefix}{run.display_name} ({short_id})"
    return f"{prefix}{run.run_id}"


def run_selector(label: str = "Run", key: str = "run_selector") -> str | None:
    runs = catalog.list_runs()
    if not runs:
        return None
    options = {_run_option_label(r): r.query_id for r in runs}
    labels = list(options.keys())
    default = st.session_state.get("selected_run_id")
    default_idx = 0
    for i, lbl in enumerate(labels):
        if options[lbl] == default:
            default_idx = i
            break
    chosen = st.selectbox(label, labels, index=default_idx, key=key)
    run_id = options[chosen]
    st.session_state.selected_run_id = run_id
    return run_id


def model_selector(run_id: str, label: str = "Modelo", key: str = "model_selector") -> str:
    from pns_health_research.dashboard.content.pt_br import model_label
    from pns_health_research.dashboard.services import classification

    models = list(classification.list_models(run_id))
    if not models:
        st.caption("Nenhum modelo encontrado.")
        return st.session_state.selected_model

    models_display = {model_label(m): m for m in models}
    labels = list(models_display.keys())
    default = st.session_state.get("selected_model")
    default_key = model_label(default) if default in models else labels[0]
    idx = labels.index(default_key) if default_key in labels else 0
    chosen = st.selectbox(label, labels, index=idx, key=key)
    model = models_display[chosen]
    st.session_state.selected_model = model
    return model
