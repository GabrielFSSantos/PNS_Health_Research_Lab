from __future__ import annotations

import pandas as pd

from pns_health_research.data.schema import LEGACY_COLUMN_MAP


def select_and_rename_legacy(
    df: pd.DataFrame,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    cmap = column_map or LEGACY_COLUMN_MAP
    raw_cols = [c for c in cmap if c in df.columns]
    out = df[raw_cols].copy()
    return out.rename(columns={k: v for k, v in cmap.items() if k in out.columns})


def filter_adults_legacy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    n_in = len(df)
    mask_age = df["Idade"].isna() | (df["Idade"] >= 18)
    mask = mask_age & df["Peso"].notna() & df["Altura"].notna() & df["Serviço de Saúde"].notna()
    out = df.loc[mask].copy()
    stats = {
        "rows_in": n_in,
        "rows_out": len(out),
        "removed": n_in - len(out),
        "removed_by_age_or_missing": n_in - len(out),
    }
    return out, stats
