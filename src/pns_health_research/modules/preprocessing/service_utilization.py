"""Encoding de variáveis de uso de serviços/medicamentos/urgência (bloco J/K/H)."""

from __future__ import annotations

import pandas as pd

from pns_health_research.data.variable_registry import load_variable_registry

MISSING = -1

SERVICE_BINARY_YES_NO: list[str] = [
    "Atendimento Coberto Plano",
    "Pagou Atendimento 2sem",
    "Receitou Medicamento",
    "Pagou Medicamentos",
    "Internação Coberta Plano",
    "Pagou Internação",
    "Urgência no Domicílio",
    "Urgência Coberta Plano",
    "Pagou Urgência",
    "Ambulância",
    "Vacina Gripe 12m",
]

SERVICE_BINARY_UNSURE: list[str] = [
    "Atendimento SUS",
    "Internação SUS",
    "Urgência SUS",
    "Medicamento Contínuo",
]

SERVICE_ORDINAL_PARTIAL: list[str] = [
    "Obteve Medicamentos",
    "Medicamento Coberto Plano",
    "Meds Farmácia Popular",
    "Meds Serviço Público",
]

SERVICE_COUNT: list[str] = [
    "Consultas 12m",
    "Nº Internações 12m",
    "Nº Meds Contínuos 2sem",
]

ULTIMA_CONSULTA_MEDICA_MAP: dict[str, int] = {
    "Até 1 ano": 1,
    "Mais de 1 ano a 2 anos": 2,
    "Mais de 2 anos a 3 anos": 3,
    "Mais de 3 anos": 4,
    "Nunca foi ao médico": 5,
}

ULTIMA_CONSULTA_H_MAP: dict[str, int] = {
    "Até 15 dias": 1,
    "Mais de 15 dias até 1 mês": 2,
    "Mais de 1 mês até 6 meses": 3,
    "Mais de 6 meses até 1 ano": 4,
    "Há mais de 1 ano": 5,
}

MOTIVO_CONSULTA_MAP: dict[str, int] = {
    "Acidente ou lesão": 1,
    "Continuação de tratamento ou terapia": 2,
    "Consulta pré-natal": 3,
    "Exame médico periódico": 4,
    "Outro exame médico (admissional, para carteira de motorista, etc)": 5,
    "Problema de saúde mental": 6,
    "Doença ou outro problema de saúde": 7,
    "Outro": 8,
}

LOCAL_ATENDIMENTO_MAP: dict[str, int] = {
    "Unidade básica de saúde (posto ou centro de saúde ou unidade de saúde da família)": 1,
    "Centro de Especialidades, Policlínica pública ou PAM – Posto de Assistência Médica": 2,
    "CAPS – Centro de Atenção Psicossocial": 3,
    "UPA (Unidade de Pronto Atendimento)": 4,
    "Outro tipo de Pronto Atendimento Público (24 horas)": 5,
    "Pronto-socorro ou emergência de hospital público": 6,
    "Hospital público/ambulatório": 7,
    "Consultório particular ou clínica privada": 8,
    "Pronto-socorro ou emergência de hospital privado": 9,
    "Hospital privado/ambulatório": 10,
    "Consultório de outro profissional de saúde": 11,
    "Domicílio": 12,
    "Outro": 13,
}

MOTIVO_NAO_OBTER_MEDS_MAP: dict[str, int] = {
    "Não conseguiu obter no serviço público de saúde": 1,
    "Não conseguiu o(s) medicamento(s) no programa Aqui Tem Farmácia Popular": 2,
    "Não tinha farmácia próxima ou teve dificuldade de transporte": 3,
    "Não conseguiu encontrar todos os medicamentos na farmácia": 4,
    "Não tinha dinheiro para comprar": 5,
    "Não achou necessário": 6,
    "Outro": 7,
}

PARTIAL_RESPONSE_MAP: dict[str, int] = {
    "Sim, todos": 1,
    "Sim, alguns": 2,
    "Não, nenhum": 3,
}


def _encode_yes_no(value: object, *, yes_value: str = "Sim") -> int:
    if pd.isna(value):
        return MISSING
    if value == yes_value:
        return 1
    if value == "Não":
        return 0
    return MISSING


def _encode_yes_no_unsure(value: object, *, yes_value: str = "Sim") -> int:
    if pd.isna(value):
        return MISSING
    if value == yes_value:
        return 1
    if value == "Não":
        return 0
    if str(value).startswith("Não sabe"):
        return MISSING
    return MISSING


def _encode_mapped(value: object, mapping: dict[str, int]) -> int:
    if pd.isna(value):
        return MISSING
    if value in mapping:
        return mapping[value]
    s = str(value).strip()
    if s in mapping:
        return mapping[s]
    return MISSING


def _encode_count(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(MISSING).astype(int)


def _is_already_encoded(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = series.dropna().unique()
    if len(values) == 0:
        return False
    return set(int(v) for v in values).issubset({0, 1, MISSING})


def _merged_columns(hardcoded: list[str], semantic_type: str) -> list[str]:
    """Une listas legadas com entradas do variable_registry (se houver)."""
    try:
        registry_labels = load_variable_registry().labels_for_type(semantic_type)
        return list(dict.fromkeys(hardcoded + registry_labels))
    except Exception:
        return hardcoded


def encode_service_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica encoding numérico às colunas de uso de serviços (política explicit_missing)."""
    out = df.copy()

    yes_no_cols = _merged_columns(SERVICE_BINARY_YES_NO, "binary")
    for col in yes_no_cols:
        if col in out.columns and not _is_already_encoded(out[col]):
            out[col] = out[col].apply(_encode_yes_no)

    unsure_cols = _merged_columns(SERVICE_BINARY_UNSURE, "binary")
    for col in unsure_cols:
        if col in out.columns and not _is_already_encoded(out[col]):
            out[col] = out[col].apply(_encode_yes_no_unsure)

    for col in SERVICE_ORDINAL_PARTIAL:
        if col in out.columns:
            out[col] = out[col].apply(lambda x, m=PARTIAL_RESPONSE_MAP: _encode_mapped(x, m))

    if "Última Consulta Médica" in out.columns:
        out["Última Consulta Médica"] = out["Última Consulta Médica"].apply(
            lambda x: _encode_mapped(x, ULTIMA_CONSULTA_MEDICA_MAP)
        )

    if "Última Consulta H" in out.columns:
        out["Última Consulta H"] = out["Última Consulta H"].apply(
            lambda x: _encode_mapped(x, ULTIMA_CONSULTA_H_MAP)
        )

    if "Motivo da Consulta" in out.columns:
        out["Motivo da Consulta"] = out["Motivo da Consulta"].apply(
            lambda x: _encode_mapped(x, MOTIVO_CONSULTA_MAP)
        )

    if "Local do Atendimento" in out.columns:
        out["Local do Atendimento"] = out["Local do Atendimento"].apply(
            lambda x: _encode_mapped(x, LOCAL_ATENDIMENTO_MAP)
        )

    if "Motivo Não Obter Meds" in out.columns:
        out["Motivo Não Obter Meds"] = out["Motivo Não Obter Meds"].apply(
            lambda x: _encode_mapped(x, MOTIVO_NAO_OBTER_MEDS_MAP)
        )

    count_cols = _merged_columns(SERVICE_COUNT, "count")
    for col in count_cols:
        if col in out.columns:
            out[col] = _encode_count(out[col])

    return out
