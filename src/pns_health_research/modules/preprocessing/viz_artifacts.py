"""Agregações de dataset para visualização (sem expor base completa)."""

from __future__ import annotations

from typing import Any


def build_preprocess_funnel(
    filter_stats: dict[str, Any],
    metrics: dict[str, Any],
    *,
    processing_stats: dict[str, Any] | None = None,
    profile_kind: str | None = None,
    methodological_profile: str | None = None,
) -> dict[str, Any]:
    proc = processing_stats or {}
    imputation = proc.get("imputation") or {}
    top_imputed = sorted(
        (imputation.get("columns_with_imputation") or {}).items(),
        key=lambda x: x[1],
        reverse=True,
    )[:8]
    kind = (profile_kind or "legacy").lower()
    is_pns = kind == "pns"

    weights_desc = (
        "Aplica peso amostral oficial da PNS (V00291), alinhado ao desenho IBGE."
        if is_pns
        else "Calcula peso amostral pela fórmula R do TCC (reprodução legada)."
    )
    missing_desc = (
        "Preenche ausentes com -1 nas categóricas (explicit_missing) e 0 nas binárias — "
        "não confunde ausente com 'não'."
        if is_pns
        else "Preenche valores ausentes restantes (NaN) com 0 nas binárias ou -1 nas "
        "categóricas — não remove linhas por comorbidade ausente."
    )

    return {
        "profile_kind": profile_kind,
        "methodological_profile": methodological_profile,
        "stages": [
            {
                "name": "raw",
                "rows": filter_stats.get("rows_in", metrics.get("rows_in")),
                "description": (
                    "Microdados PNS 2019 completos antes de qualquer filtro ou transformação."
                ),
            },
            {
                "name": "columns_selected",
                "rows": filter_stats.get("rows_in", metrics.get("rows_in")),
                "columns": filter_stats.get("columns_selected"),
                "description": (
                    "Seleção das variáveis do TCC (códigos IBGE renomeados para nomes legíveis)."
                ),
            },
            {
                "name": "adult_filter",
                "rows": filter_stats.get("rows_out", metrics.get("rows_out")),
                "removed": filter_stats.get("removed"),
                "details": {
                    "removed_under_18": filter_stats.get("removed_under_18", 0),
                    "removed_missing_weight": filter_stats.get("removed_missing_weight", 0),
                    "removed_missing_height": filter_stats.get("removed_missing_height", 0),
                    "removed_missing_health_service": filter_stats.get(
                        "removed_missing_health_service", 0
                    ),
                },
                "description": (
                    "Remove menores de 18 anos, registros sem peso ou altura (necessários para IMC) "
                    "e sem resposta sobre uso de serviços de saúde."
                ),
            },
            {
                "name": "gestational_imputation",
                "rows": filter_stats.get("rows_out", metrics.get("rows_out")),
                "description": (
                    "Imputa idade ausente pela média por alvo (fluxo legacy) e consolida "
                    "hipertensão/diabetes gestacional nas comorbidades."
                ),
                "enabled": proc.get("impute_age_by_target", False),
            },
            {
                "name": "weights_attached",
                "rows": filter_stats.get("rows_out", metrics.get("rows_out")),
                "description": weights_desc,
            },
            {
                "name": "binary_encoding",
                "rows": filter_stats.get("rows_out", metrics.get("rows_out")),
                "description": (
                    "Transforma respostas em colunas binárias/one-hot: sexo, faixas etárias, IMC, "
                    "comorbidades, região, escolaridade, etc."
                ),
            },
            {
                "name": "missing_imputation",
                "rows": metrics.get("rows_out"),
                "cells_imputed": imputation.get("total_cells_imputed", 0),
                "top_columns": dict(top_imputed),
                "description": missing_desc,
            },
            {
                "name": "processed",
                "rows": metrics.get("rows_out"),
                "binary_columns": metrics.get("binary_columns"),
                "categorical_columns": metrics.get("categorical_columns"),
                "description": (
                    "Bases finais: matriz binária (agrupamento) e categórica numérica (classificação)."
                ),
            },
        ],
    }


def build_dataset_distributions(
    binary_df,
    *,
    max_bins: int = 20,
    sample_columns: list[str] | None = None,
) -> dict[str, Any]:
    import pandas as pd

    cols = sample_columns or [
        "Homem",
        "Mulher",
        "Idade 18 a 29",
        "Idade 30 a 39",
        "Idade 40 a 49",
        "Idade 50 a 59",
        "Idade 60 a 69",
        "Saúde Regular",
        "Saúde Bom",
        "Saúde Ruim",
        "Saúde Muito Bom",
        "Saúde Muito Ruim",
        "Serviço de Saúde",
        "Estado de Saúde",
        "IMC",
    ]
    distributions: dict[str, Any] = {}
    for col in cols:
        if col not in binary_df.columns:
            continue
        series = binary_df[col].dropna()
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > max_bins:
            hist, edges = pd.cut(series, bins=max_bins, retbins=True)
            counts = hist.value_counts().sort_index()
            distributions[col] = {
                "type": "histogram",
                "bins": [float(x) for x in edges],
                "counts": [int(v) for v in counts.values],
            }
        else:
            vc = series.value_counts().head(30)
            distributions[col] = {
                "type": "categorical",
                "values": {str(k): int(v) for k, v in vc.items()},
            }
    return distributions
