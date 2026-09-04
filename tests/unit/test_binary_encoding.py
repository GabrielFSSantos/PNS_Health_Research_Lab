import pandas as pd
import pytest

from pns_health_research.modules.preprocessing.features import _yes_no_binary, attach_weights


def test_yes_no_binary_legacy_zero():
    s = pd.Series(["Sim", "Não", None, "Ignorado"])
    out = _yes_no_binary(s, policy="legacy_zero")
    assert list(out) == [1, 0, 0, 0]


def test_yes_no_binary_explicit_missing():
    s = pd.Series(["Sim", "Não", None, "Ignorado"])
    out = _yes_no_binary(s, policy="explicit_missing", missing_value=-1)
    assert list(out) == [1, 0, -1, -1]


def test_attach_weights_uses_peso_amostral():
    df = pd.DataFrame(
        {
            "UF": ["São Paulo"],
            "Extrato": [1],
            "Peso (V0091)": [1],
            "Ordem do Domicílio": [1],
            "Número de Ordem": [1],
            "Peso Amostral": [1234.5],
        }
    )
    out = attach_weights(df, legacy_formula=False)
    assert float(out["survey_weight"].iloc[0]) == pytest.approx(1234.5)
