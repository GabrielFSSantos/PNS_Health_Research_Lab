"""Registry de variáveis derivadas.

Nova variável: implementar uma função aqui, registrar a chave e ligar
`derived.<chave>.enabled` em `configs/preprocessing.yaml`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from pns_health_research.config.models import PreprocessingConfig

DerivedFn = Callable[[pd.DataFrame], pd.DataFrame]


def apply_faixas_idade(df: pd.DataFrame) -> pd.DataFrame:
    """Faixas etárias binárias (substitui a coluna Idade)."""
    out = df.copy()
    if "Idade" not in out.columns:
        return out
    bands = [
        (18, 29, "Idade 18 a 29"),
        (30, 39, "Idade 30 a 39"),
        (40, 49, "Idade 40 a 49"),
        (50, 59, "Idade 50 a 59"),
        (60, 69, "Idade 60 a 69"),
        (70, 79, "Idade 70 a 79"),
        (80, 89, "Idade 80 a 89"),
        (90, 99, "Idade 90 a 99"),
        (100, 104, "Idade 100 a 104"),
    ]
    for lo, hi, name in bands:
        out[name] = ((out["Idade"] >= lo) & (out["Idade"] <= hi)).astype(int)
    return out.drop(columns=["Idade"])


def apply_imc_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Categorias de IMC e faixas de peso/altura (base binária)."""
    out = df.copy()
    if "Peso" not in out.columns or "Altura" not in out.columns:
        return out
    imc = out["Peso"] / (out["Altura"] / 100) ** 2
    out["Baixo Peso"] = (imc < 18.5).astype(int)
    out["Peso Normal"] = ((imc >= 18.5) & (imc <= 24.9)).astype(int)
    out["Sobrepeso"] = ((imc >= 25) & (imc <= 29.9)).astype(int)
    out["Obesidade"] = (imc >= 30).astype(int)

    out["Peso 30 a 56"] = ((out["Peso"] >= 30) & (out["Peso"] <= 56)).astype(int)
    out["Peso 57 a 82"] = ((out["Peso"] >= 57) & (out["Peso"] <= 82)).astype(int)
    out["Peso 83 a 108"] = ((out["Peso"] >= 83) & (out["Peso"] <= 108)).astype(int)
    out["Peso 109 a 134"] = ((out["Peso"] >= 109) & (out["Peso"] <= 134)).astype(int)
    out["Peso 135 a 160"] = ((out["Peso"] >= 135) & (out["Peso"] <= 160)).astype(int)

    out["Altura 129 a 142"] = ((out["Altura"] >= 129) & (out["Altura"] <= 142)).astype(int)
    out["Altura 143 a 156"] = ((out["Altura"] >= 143) & (out["Altura"] <= 156)).astype(int)
    out["Altura 157 a 170"] = ((out["Altura"] >= 157) & (out["Altura"] <= 170)).astype(int)
    out["Altura 171 a 184"] = ((out["Altura"] >= 171) & (out["Altura"] <= 184)).astype(int)
    out["Altura 185 a 199"] = ((out["Altura"] >= 185) & (out["Altura"] <= 199)).astype(int)
    return out.drop(columns=["Peso", "Altura"])


def apply_imc_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """IMC contínuo na base categórica."""
    out = df.copy()
    if "Peso" in out.columns and "Altura" in out.columns:
        out["IMC"] = out["Peso"] / (out["Altura"] / 100) ** 2
    return out


def apply_grupos_comorbidade_binary(
    df: pd.DataFrame,
    config: PreprocessingConfig | None = None,
) -> pd.DataFrame:
    out = df.copy()
    explicit = config is not None and config.binary_encoding_policy == "explicit_missing"

    def _col(name: str) -> pd.Series:
        if name not in out.columns:
            return pd.Series(-1 if explicit else 0, index=out.index)
        s = pd.to_numeric(out[name], errors="coerce")
        if explicit:
            return s
        return s.fillna(0)

    out["Grupo Metabólico"] = _col("Diabetes")
    out["Grupo Respiratório"] = _col("Asma")
    out["Grupo Artrite"] = _col("Artrite ou Reumatismo")
    out["Grupo DORT"] = _col("DORT")
    out["Grupo Mental"] = _col("Doença Mental")
    out["Grupo Depressão"] = _col("Depressão")
    out["Grupo Pulmonar"] = _col("Doença Crônica no Pulmão")
    out["Grupo Renal"] = _col("Insuficiência Renal Crônica")
    out["Grupo Oncológico"] = _col("Câncer")
    h = _col("Hipertensão")
    c = _col("Cardiopatia")
    a = _col("AVC ou Derrame")
    if explicit:
        cardio = np.where(
            (h == 1) | (c == 1) | (a == 1),
            1,
            np.where((h == -1) & (c == -1) & (a == -1), -1, 0),
        )
    else:
        cardio = ((h == 1) | (c == 1) | (a == 1)).astype(int)
    out["Grupo Cardiovascular"] = cardio
    return out


def apply_grupos_comorbidade_categorical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _g(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce").fillna(-1).astype(int)

    if "Diabetes" in out.columns:
        out["Grupo Metabólico"] = _g(out["Diabetes"])
    if "Asma" in out.columns:
        out["Grupo Respiratório"] = _g(out["Asma"])
    if "Artrite ou Reumatismo" in out.columns:
        out["Grupo Artrite"] = _g(out["Artrite ou Reumatismo"])
    if "DORT" in out.columns:
        out["Grupo DORT"] = _g(out["DORT"])
    if "Doença Mental" in out.columns:
        out["Grupo Mental"] = _g(out["Doença Mental"])
    if "Depressão" in out.columns:
        out["Grupo Depressão"] = _g(out["Depressão"])
    if "Doença Crônica no Pulmão" in out.columns:
        out["Grupo Pulmonar"] = _g(out["Doença Crônica no Pulmão"])
    if "Insuficiência Renal Crônica" in out.columns:
        out["Grupo Renal"] = _g(out["Insuficiência Renal Crônica"])
    if "Câncer" in out.columns:
        out["Grupo Oncológico"] = _g(out["Câncer"])
    if all(c in out.columns for c in ("Hipertensão", "Cardiopatia", "AVC ou Derrame")):
        out["Grupo Cardiovascular"] = np.where(
            (out["Hipertensão"].isin([-1]))
            & (out["Cardiopatia"].isin([-1]))
            & (out["AVC ou Derrame"].isin([-1])),
            -1,
            (
                (out["Hipertensão"] == 1)
                | (out["Cardiopatia"] == 1)
                | (out["AVC ou Derrame"] == 1)
            ).astype(int),
        )
    return out


BINARY_DERIVED: dict[str, DerivedFn] = {
    "faixas_idade": apply_faixas_idade,
    "imc": apply_imc_binary,
    "grupos_comorbidade": apply_grupos_comorbidade_binary,
}

CATEGORICAL_DERIVED: dict[str, DerivedFn] = {
    "imc": apply_imc_categorical,
    "grupos_comorbidade": apply_grupos_comorbidade_categorical,
}


def apply_derived_binary(
    df: pd.DataFrame,
    config: PreprocessingConfig | None = None,
) -> pd.DataFrame:
    out = df
    for key, fn in BINARY_DERIVED.items():
        if config is None or config.derived_enabled(key):
            if key == "grupos_comorbidade":
                out = fn(out, config=config)
            else:
                out = fn(out)
    return out


def apply_derived_categorical(
    df: pd.DataFrame,
    config: PreprocessingConfig | None = None,
) -> pd.DataFrame:
    out = df
    for key, fn in CATEGORICAL_DERIVED.items():
        if config is None or config.derived_enabled(key):
            out = fn(out)
    return out
