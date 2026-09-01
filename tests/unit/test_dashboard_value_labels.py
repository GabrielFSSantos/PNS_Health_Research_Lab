"""Testes de rótulos de valores."""

from __future__ import annotations

from pns_health_research.dashboard.content.value_labels import (
    funnel_stage_label,
    label_for_value,
    sort_health_categories,
    sort_health_categories_bubble,
)


def test_servico_saude_labels():
    assert label_for_value("Serviço de Saúde", "0") == "Não utilizou"
    assert label_for_value("Serviço de Saúde", "1") == "Utilizou"


def test_sort_health_categories():
    ordered = sort_health_categories(["Bom", "Muito Ruim", "Regular", "Ruim"])
    assert ordered[0] == "Muito Ruim"
    assert ordered[-1] == "Bom"


def test_funnel_labels():
    assert "PNS" in funnel_stage_label("raw")


def test_bubble_health_order():
    ordered = sort_health_categories_bubble(["Ruim", "Muito Bom", "Bom"])
    assert ordered[0] == "Muito Bom"
