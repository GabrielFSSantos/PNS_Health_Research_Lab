"""Testes de classificação dashboard."""

from __future__ import annotations

from pns_health_research.dashboard.services import classification

RUN_ID = "pesquisa-original-legacy"


def test_list_models():
    models = classification.list_models(RUN_ID)
    assert "random_forest" in models


def test_grid_search_present():
    gs = classification.load_grid_search(RUN_ID, "random_forest")
    assert gs.get("status") != "missing"
    assert "search" in gs


def test_curves_roc():
    curves = classification.load_curves(RUN_ID, "random_forest")
    assert "roc" in curves


def test_confusion_matrices():
    cms = classification.load_confusion_matrices(RUN_ID)
    assert isinstance(cms, dict)
