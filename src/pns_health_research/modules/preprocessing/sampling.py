from __future__ import annotations

import pandas as pd
from imblearn.under_sampling import NearMiss


def apply_nearmiss(
    df: pd.DataFrame,
    target_column: str = "Serviço de Saúde",
    version: int = 1,
) -> pd.DataFrame:
    x = df.drop(columns=[target_column])
    y = df[target_column]
    nm = NearMiss(version=version)
    x_res, y_res = nm.fit_resample(x, y)
    return pd.concat(
        [pd.DataFrame(x_res, columns=x.columns), pd.DataFrame({target_column: y_res})],
        axis=1,
    )
