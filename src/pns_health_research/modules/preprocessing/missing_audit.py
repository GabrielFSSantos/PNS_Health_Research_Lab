from __future__ import annotations

from typing import Any

import pandas as pd


def build_missing_audit(
    raw_df: pd.DataFrame,
    encoded_df: pd.DataFrame,
    comorbidity_cols: list[str],
) -> dict[str, Any]:
    """Percentual de ausentes por comorbidade antes e depois da binarização."""
    n_rows = len(raw_df)
    columns_audit: dict[str, Any] = {}
    for col in comorbidity_cols:
        if col not in raw_df.columns:
            continue
        raw_series = raw_df[col]
        na_before = int(raw_series.isna().sum())
        non_yes_no = int(
            raw_series.dropna().apply(lambda x: x not in ("Sim", "Não")).sum()
        )
        after = encoded_df[col] if col in encoded_df.columns else pd.Series(dtype=float)
        missing_after = int((after == -1).sum()) if len(after) else 0
        columns_audit[col] = {
            "na_before": na_before,
            "na_before_pct": float(na_before / n_rows) if n_rows else 0.0,
            "non_yes_no_before": non_yes_no,
            "explicit_missing_after": missing_after,
            "explicit_missing_after_pct": float(missing_after / n_rows) if n_rows else 0.0,
        }
    return {
        "n_rows": n_rows,
        "comorbidity_columns": comorbidity_cols,
        "columns": columns_audit,
    }
