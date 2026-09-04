"""Testes de encoding do bloco J/K/H (uso de serviços)."""

from __future__ import annotations

import pandas as pd

from pns_health_research.modules.preprocessing.service_utilization import encode_service_utilization


def test_encode_binary_yes_no():
    df = pd.DataFrame({"Urgência no Domicílio": ["Sim", "Não", None, "Ignorado"]})
    out = encode_service_utilization(df)
    assert list(out["Urgência no Domicílio"]) == [1, 0, -1, -1]


def test_encode_binary_unsure():
    df = pd.DataFrame({"Atendimento SUS": ["Sim", "Não", "Não sabe/Não lembra", None]})
    out = encode_service_utilization(df)
    assert list(out["Atendimento SUS"]) == [1, 0, -1, -1]


def test_encode_ordinal_partial():
    df = pd.DataFrame(
        {"Obteve Medicamentos": ["Sim, todos", "Sim, alguns", "Não, nenhum", None]}
    )
    out = encode_service_utilization(df)
    assert list(out["Obteve Medicamentos"]) == [1, 2, 3, -1]


def test_encode_count():
    df = pd.DataFrame({"Consultas 12m": [3, None, "5"]})
    out = encode_service_utilization(df)
    assert list(out["Consultas 12m"]) == [3, -1, 5]


def test_encode_skips_already_numeric_binary():
    df = pd.DataFrame({"Urgência no Domicílio": [1, 0, -1]})
    out = encode_service_utilization(df)
    assert list(out["Urgência no Domicílio"]) == [1, 0, -1]

    df = pd.DataFrame(
        {
            "Última Consulta Médica": ["Até 1 ano", "Nunca foi ao médico", None],
            "Última Consulta H": ["Até 15 dias", "Há mais de 1 ano", None],
        }
    )
    out = encode_service_utilization(df)
    assert list(out["Última Consulta Médica"]) == [1, 5, -1]
    assert list(out["Última Consulta H"]) == [1, 5, -1]
