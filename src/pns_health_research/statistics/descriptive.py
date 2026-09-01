from __future__ import annotations

import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table)[0]
    n = table.sum().sum()
    r, k = table.shape
    return float((chi2 / (n * min(r - 1, k - 1))) ** 0.5) if n and min(r - 1, k - 1) > 0 else 0.0


def descriptive_proportions(df: pd.DataFrame, col: str) -> pd.DataFrame:
    vc = df[col].value_counts(normalize=True)
    return pd.DataFrame({"value": vc.index.astype(str), "proportion": vc.values})
