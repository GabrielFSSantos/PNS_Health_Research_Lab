"""Detalhe de classificação."""

from __future__ import annotations

import streamlit as st

from pns_health_research.dashboard.components import charts
from pns_health_research.dashboard.components.chart_block import chart_block
from pns_health_research.dashboard.components.filters import (
    init_session_state,
    model_selector,
    run_selector,
)
from pns_health_research.dashboard.components.insight_panel import render_insights
from pns_health_research.dashboard.components.layout import page_header
from pns_health_research.dashboard.components.metric_card import metric_row
from pns_health_research.dashboard.components.section_block import section_block
from pns_health_research.dashboard.components.states import empty_state, unavailable_data
from pns_health_research.dashboard.content import pt_br
from pns_health_research.dashboard.content.pt_br import model_label
from pns_health_research.dashboard.insights import engine as insights_engine
from pns_health_research.dashboard.services import catalog, classification


def render() -> None:
    init_session_state()
    page_header(
        "Classificação",
        "Desempenho, diagnóstico, treinamento e variáveis por modelo.",
    )

    run_id = run_selector("Execução")
    if not run_id:
        empty_state("Nenhuma execução disponível.")
        return

    model = model_selector(run_id)
    model_pt = model_label(model)
    gs = classification.load_grid_search(run_id, model)
    grid_missing = gs.get("status") == "missing"

    render_insights(
        insights_engine.generate(
            {"page": "classification", "grid_search_missing": grid_missing}
        )
    )

    tabs = st.tabs(
        [
            pt_br.TAB_PANORAMA,
            pt_br.TAB_PERFORMANCE,
            pt_br.TAB_DIAGNOSIS,
            pt_br.TAB_TRAINING,
            pt_br.TAB_VARIABLES,
        ]
    )

    model_metrics = classification.load_model_metrics(run_id, model)
    detail = catalog.get_detail(run_id)

    with tabs[0]:
        with section_block(f"Modelo em destaque: {model_pt}", "metrics_bar"):
            summary = classification.load_predictions_summary(run_id)
            if summary and isinstance(summary, dict) and summary.get(model):
                s = classification.normalize_prediction_summary(summary[model])
                metric_row(
                    [
                        ("n_samples", s.get("n_samples"), False),
                        ("n_positive", s.get("n_positive"), False),
                        ("positive_rate", s.get("positive_rate"), True),
                    ]
                )
            metric_row(
                [
                    ("f1_positive", model_metrics.get("f1_positive"), True),
                    ("recall_positive", model_metrics.get("recall_positive"), True),
                    ("roc_auc", model_metrics.get("roc_auc"), False),
                ]
            )

    with tabs[1]:
        labels = [k for k, v in model_metrics.items() if isinstance(v, (int, float))]
        values = [model_metrics[k] for k in labels]
        chart_block(
            "metrics_bar",
            f"Métricas — {model_pt}",
            lambda: charts.metric_bar(labels[:8], values[:8], title=f"Métricas — {model_pt}"),
            chart_id=f"clf_metrics_{model}",
        )
        curves = classification.load_curves(run_id, model)
        chart_block("roc", "Curva ROC", lambda: charts.roc_plot(curves), chart_id=f"clf_roc_{model}")
        chart_block("pr", "Curva precisão-recall", lambda: charts.pr_plot(curves), chart_id=f"clf_pr_{model}")
        chart_block(
            "calibration",
            "Calibração",
            lambda: charts.calibration_plot(curves),
            chart_id=f"clf_cal_{model}",
        )

    with tabs[2]:
        cms = classification.load_confusion_matrices(run_id)
        cm_data = cms.get(model) if isinstance(cms, dict) else None
        if cm_data and isinstance(cm_data, list):
            chart_block(
                "confusion_matrix",
                "Matriz de confusão",
                lambda: charts.confusion_matrix_plot(cm_data),
                chart_id=f"clf_cm_{model}",
            )
        else:
            st.caption("Matriz de confusão não disponível.")

    with tabs[3]:
        if grid_missing:
            unavailable_data(
                "Grid Search",
                "./scripts/run_experiment.sh --profile pns  # ou --profile legacy",
            )
        else:
            best_params = gs.get("best_params") or {}
            chart_block(
                "gridsearch",
                "Busca em grade (Grid Search)",
                lambda: charts.gridsearch_scatter(
                    classification.grid_search_candidates_df(run_id, model),
                    best_params=best_params,
                ),
                chart_id=f"clf_gs_{model}",
            )
            gs_table = classification.grid_search_table_df(run_id, model, best_params)
            if not gs_table.empty:
                display = gs_table.copy()
                if "melhor" in display.columns:
                    display = display.sort_values("melhor", ascending=False)
                st.dataframe(display, width="stretch")
            folds = gs.get("folds")
            if folds:
                st.caption("Detalhes por fold (validação cruzada):")
                st.dataframe(folds, width="stretch")

    with tabs[4]:
        imp = classification.load_feature_importance(run_id, model)
        chart_block(
            "importance_native",
            "Importância nativa",
            lambda: charts.importance_bar(imp["native"]),
            chart_id=f"clf_imp_native_{model}",
        )
        chart_block(
            "importance_permutation",
            "Importância por permutação",
            lambda: charts.importance_bar(imp["permutation"]),
            chart_id=f"clf_imp_perm_{model}",
        )
        pred = classification.load_predictions_summary(run_id)
        with st.expander("Resumo das predições (JSON)", expanded=False):
            if isinstance(pred, dict) and pred.get(model):
                st.json(pred[model])
            else:
                st.caption("Resumo de predições não disponível.")

    with st.expander(pt_br.TECHNICAL_DETAILS_DEV):
        st.json(detail.get("lineage") or {})
        if not grid_missing:
            st.json(gs.get("search") or {})
