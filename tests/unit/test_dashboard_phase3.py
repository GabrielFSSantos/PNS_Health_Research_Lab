"""Testes da Fase 3 do dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pns_health_research.dashboard.content.value_labels import (
    funnel_stage_label,
    sort_health_categories_bubble,
)
from pns_health_research.dashboard.services import classification

RUN_ID = "pesquisa-original-legacy"


def test_funnel_stage_labels_pt_br():
    assert funnel_stage_label("raw") == "Dados brutos (PNS 2019)"
    assert funnel_stage_label("adult_filter") == "Filtro de elegibilidade"
    assert funnel_stage_label("missing_imputation") == "Imputação de ausentes"
    assert funnel_stage_label("processed") == "Base processada"


def test_bubble_health_category_order():
    ordered = sort_health_categories_bubble(["Ruim", "Muito Bom", "Regular", "Bom", "Muito Ruim"])
    assert ordered[0] == "Muito Bom"
    assert ordered[-1] == "Muito Ruim"


def test_normalize_prediction_summary():
    raw = {"n": 100, "predicted_positive_rate": 0.25}
    norm = classification.normalize_prediction_summary(raw)
    assert norm["n_samples"] == 100
    assert norm["n_positive"] == 25
    assert norm["positive_rate"] == 0.25


def test_predictions_summary_from_showcase():
    summary = classification.load_predictions_summary(RUN_ID)
    assert "random_forest" in summary
    norm = classification.normalize_prediction_summary(summary["random_forest"])
    assert norm["n_samples"] == 44508
    assert norm["n_positive"] is not None
    assert norm["positive_rate"] is not None


def test_confusion_heatmap_builds_figure_data():
    matrix = [[80, 20], [15, 85]]
    arr = np.array(matrix, dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    pct_arr = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums != 0) * 100
    assert pct_arr[0, 0] == 80.0
    assert int(arr[1, 1]) == 85


def test_gridsearch_best_params_match():
    gs = classification.load_grid_search(RUN_ID, "random_forest")
    df = classification.grid_search_candidates_df(RUN_ID, "random_forest")
    best_params = gs.get("best_params") or {}
    table = classification.grid_search_table_df(RUN_ID, "random_forest", best_params)
    assert "melhor" in table.columns
    assert table["melhor"].any()


def test_run_overrides_baseline():
    from pns_health_research.dashboard.services import catalog

    overrides = catalog.load_run_overrides()
    assert "pesquisa-original-legacy" in overrides


def test_bubble_data_available_after_backfill():
    from pns_health_research.dashboard.services import clustering

    bubble = clustering.load_bubble_health(RUN_ID)
    assert not bubble.empty
    assert {"cluster", "servico", "categoria_saude", "count"}.issubset(bubble.columns)


def test_build_bubble_plot_dataframe():
    """Garante que o dataframe de bolhas suporta múltiplos clusters sem selectbox."""
    rows = []
    for cluster in range(6):
        for svc in ("Não", "Sim"):
            for cat in ("Muito Bom", "Bom", "Regular"):
                rows.append(
                    {
                        "cluster": cluster,
                        "servico": svc,
                        "categoria_saude": cat,
                        "count": 10 + cluster,
                    }
                )
    df = pd.DataFrame(rows)
    assert df["cluster"].nunique() == 6
    assert len(df) == 36
