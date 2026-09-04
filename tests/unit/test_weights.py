import pandas as pd

from pns_health_research.modules.preprocessing.features import (
    compute_legacy_survey_weight,
    normalize_ml_weights,
)


def test_normalize_ml_weights_mean_one():
    w = pd.Series([2.0, 4.0, 6.0])
    out = normalize_ml_weights(w)
    assert abs(out.mean() - 1.0) < 1e-9


def test_legacy_weight_positive():
    df = pd.DataFrame(
        {
            "UF": ["São Paulo", "Bahia"],
            "Extrato": [1, 2],
            "Peso (V0091)": [10, 20],
            "Ordem do Domicílio": [1, 1],
            "Número de Ordem": [1, 2],
        }
    )
    w = compute_legacy_survey_weight(df)
    assert (w > 0).all()
