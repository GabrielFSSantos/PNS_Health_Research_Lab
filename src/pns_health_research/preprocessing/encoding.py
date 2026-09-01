from __future__ import annotations

import pandas as pd


def target_encode_legacy(
    df: pd.DataFrame,
    categorical_cols: list[str],
    target_col: str = "Serviço de Saúde",
) -> pd.DataFrame:
    out = df.copy()
    for col in categorical_cols:
        if col not in out.columns:
            continue
        means = out.groupby(col)[target_col].mean()
        out[col] = out[col].map(means)
    return out


def merge_internacao_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Internação" in out.columns:
        out["Serviço de Saúde"] = (out["Internação"] == 1) | (out["Serviço de Saúde"] == 1)
        out["Serviço de Saúde"] = out["Serviço de Saúde"].astype(int)
        out = out.drop(columns=["Internação"])
    return out
