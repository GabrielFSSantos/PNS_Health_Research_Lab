from __future__ import annotations

from typing import Any

import pandas as pd

from pns_health_research.config.models import TargetConfig
from pns_health_research.data.schema import CLUSTERING_COMORBIDITIES


def _as_binary_series(series: pd.Series) -> pd.Series:
    """Converte coluna já binarizada (0/1/-1) ou texto Sim/Não."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return series.apply(lambda x: pd.NA if pd.isna(x) else int(x == "Sim"))


def build_target_columns(df: pd.DataFrame, target_config: TargetConfig) -> pd.DataFrame:
    """Materializa colunas de desfecho configuradas (ex.: servico_j014, servico_composto)."""
    if not target_config.columns:
        return df

    out = df.copy()
    internacao = None
    if "Internação" in out.columns:
        internacao = _as_binary_series(out["Internação"])

    for name, spec in target_config.columns.items():
        if spec.source not in out.columns:
            continue
        base = _as_binary_series(out[spec.source])
        if spec.merge_internacao and internacao is not None:
            merged = ((base == 1) | (internacao == 1)).astype(int)
            out[name] = merged.where(base.notna() | internacao.notna(), other=pd.NA)
        else:
            out[name] = base

    return out


def build_target_definitions(target_config: TargetConfig) -> dict[str, Any]:
    definitions: dict[str, Any] = {
        "primary_column": target_config.primary_column,
        "columns": {},
    }
    for name, spec in target_config.columns.items():
        if spec.merge_internacao:
            definition = (
                f"{spec.source}==1 OR Internação==1 "
                "(consulta nos últimos 12 meses ou internação hospitalar)"
            )
        else:
            definition = f"{spec.source}==1 (consulta nos últimos 12 meses, J014)"
        definitions["columns"][name] = {
            "source": spec.source,
            "merge_internacao": spec.merge_internacao,
            "definition": definition,
        }
    return definitions


def target_prevalence(df: pd.DataFrame, target_config: TargetConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in target_config.columns:
        if name not in df.columns:
            continue
        series = pd.to_numeric(df[name], errors="coerce")
        valid = series.dropna()
        if len(valid) == 0:
            continue
        pos = int((valid == 1).sum())
        out[name] = {
            "n": int(len(valid)),
            "positive": pos,
            "negative": int((valid == 0).sum()),
            "missing": int(series.isna().sum()),
            "prevalence": float(pos / len(valid)),
        }
    return out


def comorbidity_columns_for_audit(columns: list[str] | None = None) -> list[str]:
    base = columns or CLUSTERING_COMORBIDITIES
    return [c for c in base if c != "Obesidade"]
