"""Serviços de dataset e dicionário."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from pns_health_research.persistence.base_query import normalize_fingerprint
from pns_health_research.web import query


@lru_cache(maxsize=32)
def load_dataset_info(run_id: str) -> dict[str, Any]:
    return query.get_dataset_info(run_id)


def load_dictionary_df() -> pd.DataFrame:
    data = query.get_dictionary()
    cols = data.get("columns") or []
    return pd.DataFrame(cols) if cols else pd.DataFrame()


def load_dictionary_column(internal_name: str) -> dict[str, Any]:
    return query.get_dictionary_column(internal_name)


def load_preprocessing_summary(run_id: str) -> dict[str, Any]:
    info = load_dataset_info(run_id)
    return info.get("preprocessing") or {}


def load_funnel(run_id: str) -> dict[str, Any]:
    info = load_dataset_info(run_id)
    return info.get("preprocess_funnel") or {}


def load_distributions(run_id: str) -> dict[str, Any]:
    info = load_dataset_info(run_id)
    return info.get("dataset_distributions") or {}


def load_survey_snapshot(run_id: str) -> dict[str, Any]:
    info = load_dataset_info(run_id)
    return info.get("survey_snapshot") or {}


def load_fingerprint(run_id: str) -> dict[str, Any]:
    info = load_dataset_info(run_id)
    fp = info.get("fingerprint") or {}
    if not fp:
        return {}
    normalized = normalize_fingerprint(fp) if isinstance(fp, dict) else {}
    prep = info.get("preprocessing") or {}
    rows = normalized.get("rows") or prep.get("rows_out")
    columns = normalized.get("columns") or prep.get("binary_columns")
    categorical_columns = (
        normalized.get("categorical_columns") or prep.get("categorical_columns")
    )
    return {
        **normalized,
        "rows": rows,
        "columns": columns,
        "n_rows": rows,
        "n_columns": columns,
        "categorical_columns": categorical_columns,
    }


def format_fingerprint_metric(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    return str(value)
