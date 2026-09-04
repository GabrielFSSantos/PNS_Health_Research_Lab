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


def filter_adults_legacy(
    df: pd.DataFrame,
    *,
    min_age: int = 18,
    require_imc_inputs: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    n_in = len(df)
    age = df["Idade"] if "Idade" in df.columns else pd.Series([pd.NA] * n_in, index=df.index)
    peso = df["Peso"] if "Peso" in df.columns else pd.Series([pd.NA] * n_in, index=df.index)
    altura = df["Altura"] if "Altura" in df.columns else pd.Series([pd.NA] * n_in, index=df.index)
    servico = (
        df["Serviço de Saúde"]
        if "Serviço de Saúde" in df.columns
        else pd.Series([pd.NA] * n_in, index=df.index)
    )

    under_18 = age.notna() & (age < min_age)
    missing_weight = peso.isna()
    missing_height = altura.isna()
    missing_health_service = servico.isna()

    mask_age = age.isna() | (age >= min_age)
    if require_imc_inputs:
        mask = mask_age & peso.notna() & altura.notna() & servico.notna()
    else:
        mask = mask_age & servico.notna()
    out = df.loc[mask].copy()

    stats = {
        "rows_in": n_in,
        "rows_out": len(out),
        "removed": n_in - len(out),
        "removed_by_age_or_missing": n_in - len(out),
        "removed_under_18": int(under_18.sum()),
        "removed_missing_weight": int(missing_weight.sum()),
        "removed_missing_height": int(missing_height.sum()),
        "removed_missing_health_service": int(missing_health_service.sum()),
        "columns_selected": len(df.columns),
    }
    return out, stats
