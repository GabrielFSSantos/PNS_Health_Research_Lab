"""Engenharia de features legada TCC (notebook células 38, 40, 43)."""

from __future__ import annotations

import pandas as pd

from pns_health_research.data.schema import (
    CENTER_WEST_STATES,
    NORTH_STATES,
    NORTHEAST_STATES,
    SOUTH_STATES,
    SOUTHEAST_STATES,
    UF_TO_CODE,
)

DESIGN_COLUMNS = ["Extrato", "Peso (V0091)", "Ordem do Domicílio", "Número de Ordem"]
INTERNAL_WEIGHT_COLS = ["survey_weight", "ml_sample_weight"]

LIMITA_LEVELS: list[tuple[str, str]] = [
    ("Não Limita", "Não limita"),
    ("Limita Pouco", "Um pouco"),
    ("Limita Moderadamente", "Moderadamente"),
    ("Limita Intensamente", "Intensamente"),
    ("Limita Muito Intensamente", "Muito intensamente"),
]

# (prefixo colunas binárias, nome coluna limita)
LIMITA_DISEASES: list[tuple[str, str]] = [
    ("AVC ou Derrame", "AVC ou Derrame Limita Atividades"),
    ("Asma", "Asma Limita Atividades"),
    ("Artrite ou Reumatismo", "Artrite ou Reumatismo Limita Atividades"),
    ("Dor na Coluna", "Dor na Coluna Limita Atividades"),
    ("DORT", "DORT Limita Atividades"),
    ("Depressão", "Depressão Limita Atividades"),
    ("Hipertensão", "Hipertensão Limita Atividades"),
    ("Diabetes", "Diabetes Limita Atividades"),
    ("Cardiopatia", "Cardiopatia Limita Atividades"),
    ("Câncer", "Câncer Limita Atividades"),
    ("Insuficiência Renal", "Insuficiência Renal Limita Atividades"),
    ("Doença Mental", "Doença Mental Limita Atividades"),
    ("Doença Crônica no Pulmão", "Doença Crônica no Pulmão Limita Atividades"),
]


def drop_design_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DESIGN_COLUMNS if c in df.columns]
    return df.drop(columns=cols) if cols else df


def _flag(series: pd.Series, value: str) -> pd.Series:
    return series.apply(lambda x: 0 if pd.isna(x) else int(x == value))


def _encode_limita_binary(df: pd.DataFrame, prefix: str, limita_col: str) -> pd.DataFrame:
    out = df.copy()
    if limita_col not in out.columns:
        return out
    for suffix, raw in LIMITA_LEVELS:
        out[f"{prefix} {suffix}"] = _flag(out[limita_col], raw)
    return out.drop(columns=[limita_col])


def create_binary_features_tcc(
    df: pd.DataFrame,
    *,
    one_hot: bool = True,
) -> pd.DataFrame:
    """One-hot demográfico e limita-atividades (derivadas ficam em derived.py)."""
    out = df.copy()
    if not one_hot:
        return out

    if "UF" in out.columns:
        out["Região Norte"] = out["UF"].isin(NORTH_STATES).astype(int)
        out["Região Nordeste"] = out["UF"].isin(NORTHEAST_STATES).astype(int)
        out["Região Centro Oeste"] = out["UF"].isin(CENTER_WEST_STATES).astype(int)
        out["Região Sudeste"] = out["UF"].isin(SOUTHEAST_STATES).astype(int)
        out["Região Sul"] = out["UF"].isin(SOUTH_STATES).astype(int)
        out = out.drop(columns=["UF"])

    if "Zona" in out.columns:
        out["Urbano"] = _flag(out["Zona"], "Urbano")
        out["Rural"] = _flag(out["Zona"], "Rural")
        out = out.drop(columns=["Zona"])

    if "Grau de Ensino" in out.columns:
        edu_map = {
            "EF Incompleto": "Fundamental incompleto ou equivalente",
            "SP Incompleto": "Superior incompleto ou equivalente",
            "EM Incompleto": "Médio incompleto ou equivalente",
            "EF Completo": "Fundamental completo ou equivalente",
            "SP Completo": "Superior completo",
            "EM Completo": "Médio completo ou equivalente",
            "Sem Instrução": "Sem instrução",
        }
        for col, val in edu_map.items():
            out[col] = _flag(out["Grau de Ensino"], val)
        out = out.drop(columns=["Grau de Ensino"])

    if "Renda" in out.columns:
        renda_map = {
            "Per Capita 1": "Até ¼ salário mínimo",
            "Per Capita 2": "Mais de ¼ até ½ salário mínimo",
            "Per Capita 3": "Mais de ½ até 1 salário mínimo",
            "Per Capita 4": "Mais de 1 até 2 salários mínimos",
            "Per Capita 5": "Mais de 2 até 3 salários mínimos",
            "Per Capita 6": "Mais de 3 até 5 salários mínimos",
            "Per Capita 7": "Mais de 5 salários mínimos",
        }
        for col, val in renda_map.items():
            out[col] = out["Renda"].apply(
                lambda x, v=val, c=col: 0
                if pd.isna(x)
                else int(str(x).startswith(v) if c == "Per Capita 4" else x == v)
            )
        out = out.drop(columns=["Renda"])

    if "Sexo" in out.columns:
        out["Homem"] = _flag(out["Sexo"], "Homem")
        out["Mulher"] = _flag(out["Sexo"], "Mulher")
        out = out.drop(columns=["Sexo"])

    if "Raça" in out.columns:
        for i, race in enumerate(
            ["Branca", "Amarela", "Preta", "Parda", "Indígena"], start=1
        ):
            out[f"Raça {i}"] = _flag(out["Raça"], race)
        out = out.drop(columns=["Raça"])

    if "Estado Civil" in out.columns:
        out["Casado"] = _flag(out["Estado Civil"], "Casado(a)")
        out["Solteiro"] = _flag(out["Estado Civil"], "Solteiro(a)")
        out["Viúvo"] = _flag(out["Estado Civil"], "Viúvo(a)")
        out["Divorciado"] = _flag(
            out["Estado Civil"],
            "Divorciado(a) ou desquitado(a) ou separado(a) judicialmente",
        )
        out = out.drop(columns=["Estado Civil"])

    if "Estado de Saúde" in out.columns:
        out["Saúde Regular"] = _flag(out["Estado de Saúde"], "Regular")
        out["Saúde Muito Ruim"] = _flag(out["Estado de Saúde"], "Muito ruim")
        out["Saúde Bom"] = _flag(out["Estado de Saúde"], "Bom")
        out["Saúde Muito Bom"] = _flag(out["Estado de Saúde"], "Muito bom")
        out["Saúde Ruim"] = _flag(out["Estado de Saúde"], "Ruim")
        out = out.drop(columns=["Estado de Saúde"])

    for prefix, limita_col in LIMITA_DISEASES:
        out = _encode_limita_binary(out, prefix, limita_col)

    return out


def _ordinal_limita(series: pd.Series) -> pd.Series:
    mapping = {
        "Não limita": 1,
        "Um pouco": 2,
        "Moderadamente": 3,
        "Intensamente": 4,
        "Muito intensamente": 5,
    }
    return series.map(mapping).fillna(-1).astype(int)


def transform_categorical_numeric_tcc(
    df: pd.DataFrame,
    *,
    one_hot: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    if not one_hot:
        return out
    binary_like = [
        "Cadastro UBS",
        "Ler e Escrever",
        "Escola",
        "Trabalha",
        "Plano de Saúde",
        "Serviço de Saúde",
        "Internação",
        "Exercício Físico",
        "Colesterol Alto",
        "AVC ou Derrame",
        "Asma",
        "Artrite ou Reumatismo",
        "Dor na Coluna",
        "DORT",
        "Depressão",
        "Hipertensão",
        "Diabetes",
        "Cardiopatia",
        "Câncer",
        "Insuficiência Renal Crônica",
        "Doença Mental",
        "Doença Crônica no Pulmão",
    ]
    for col in binary_like:
        if col not in out.columns:
            continue
        out[col] = out[col].apply(
            lambda x: -1 if pd.isna(x) else (1 if x == "Sim" else (0 if x == "Não" else -1))
        )

    if "UF" in out.columns:
        out["UF"] = out["UF"].map(UF_TO_CODE).fillna(-1).astype(int)
    if "Zona" in out.columns:
        out["Zona"] = out["Zona"].map({"Urbano": 1, "Rural": 2}).fillna(-1).astype(int)

    edu_ord = {
        "Sem instrução": 0,
        "Fundamental incompleto ou equivalente": 1,
        "Fundamental completo ou equivalente": 2,
        "Médio incompleto ou equivalente": 3,
        "Médio completo ou equivalente": 4,
        "Superior incompleto ou equivalente": 5,
        "Superior completo": 6,
    }
    if "Grau de Ensino" in out.columns:
        out["Grau de Ensino"] = out["Grau de Ensino"].map(edu_ord).fillna(-1).astype(int)

    def _renda_ord(x: object) -> int:
        if pd.isna(x):
            return -1
        s = str(x)
        if s == "Até ¼ salário mínimo":
            return 1
        if s == "Mais de ¼ até ½ salário mínimo":
            return 2
        if s == "Mais de ½ até 1 salário mínimo":
            return 3
        if s.startswith("Mais de 1 até 2 salários mínimos"):
            return 4
        if s == "Mais de 2 até 3 salários mínimos":
            return 5
        if s == "Mais de 3 até 5 salários mínimos":
            return 6
        if s == "Mais de 5 salários mínimos":
            return 7
        return -1

    if "Renda" in out.columns:
        out["Renda"] = out["Renda"].map(_renda_ord).astype(int)

    if "Agente Comunitário" in out.columns:
        out["Agente Comunitário"] = out["Agente Comunitário"].apply(
            lambda x: -1
            if pd.isna(x)
            else (
                1
                if str(x)
                in ("Mensalmente", "A cada 2 meses", "De 2 a 4 vezes", "Uma vez")
                else (0 if x == "Nunca recebeu" else -1)
            )
        )

    if "Sexo" in out.columns:
        out["Sexo"] = out["Sexo"].map({"Homem": 1, "Mulher": 2}).fillna(-1).astype(int)

    race_ord = {"Branca": 1, "Amarela": 2, "Preta": 3, "Parda": 4, "Indígena": 5}
    if "Raça" in out.columns:
        out["Raça"] = out["Raça"].map(race_ord).fillna(-1).astype(int)

    civil_ord = {
        "Solteiro(a)": 1,
        "Casado(a)": 2,
        "Viúvo(a)": 3,
        "Divorciado(a) ou desquitado(a) ou separado(a) judicialmente": 4,
    }
    if "Estado Civil" in out.columns:
        out["Estado Civil"] = out["Estado Civil"].map(civil_ord).fillna(-1).astype(int)

    health_ord = {
        "Muito ruim": 1,
        "Ruim": 2,
        "Regular": 3,
        "Bom": 4,
        "Muito bom": 5,
    }
    if "Estado de Saúde" in out.columns:
        out["Estado de Saúde"] = out["Estado de Saúde"].map(health_ord).fillna(-1).astype(int)

    if "Álcool" in out.columns:
        out["Álcool"] = out["Álcool"].apply(
            lambda x: -1
            if pd.isna(x)
            else (
                1
                if str(x) in ("Uma vez ou mais por mês", "Menos de uma vez por mês")
                else (0 if x == "Nunca" else -1)
            )
        )

    if "Tabaco" in out.columns:
        out["Tabaco"] = out["Tabaco"].apply(
            lambda x: -1
            if pd.isna(x)
            else (1 if x != "Não fumo atualmente" else 0)
        )

    for _, limita_col in LIMITA_DISEASES:
        if limita_col in out.columns:
            out[limita_col] = _ordinal_limita(out[limita_col])

    return out
