"""Testes do snapshot de variáveis da pesquisa."""

from __future__ import annotations

import pandas as pd

from pns_health_research.modules.preprocessing.survey_snapshot import (
    build_survey_snapshot,
    load_survey_metadata,
)


def test_load_survey_metadata_has_servico():
    meta = load_survey_metadata()
    vars_ = meta.get("variables") or {}
    assert "Serviço de Saúde" in vars_
    assert vars_["Serviço de Saúde"].get("question")


def test_build_survey_snapshot_counts():
    df = pd.DataFrame(
        {
            "Sexo": ["Homem", "Mulher", "Homem"],
            "Serviço de Saúde": ["Sim", "Não", "Sim"],
            "Idade": [25, 40, 60],
        }
    )
    snap = build_survey_snapshot(df)
    assert snap["rows"] == 3
    assert "Serviço de Saúde" in snap["variables"]
    counts = snap["variables"]["Serviço de Saúde"]["counts"]
    assert counts.get("Sim") == 2
    assert counts.get("Não") == 1


def test_survey_snapshot_from_showcase():
    from pns_health_research.dashboard.services import dataset

    snap = dataset.load_survey_snapshot("pesquisa-original-legacy")
    assert snap.get("rows", 0) > 0
    assert "Serviço de Saúde" in (snap.get("variables") or {})
