from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pns_health_research.statistics.descriptive import cramers_v


def chi_square_association(
    x: pd.Series,
    y: pd.Series,
) -> dict[str, Any]:
    """Teste qui-quadrado de independência entre duas variáveis categóricas."""
    table = pd.crosstab(x, y)
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    n = table.sum().sum()
    cv = cramers_v(x, y)
    return {
        "test": "chi_square",
        "hypothesis": "H0: variáveis independentes",
        "n": int(n),
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_value),
        "cramers_v": float(cv),
        "effect_size": float(cv),
        "interpretation": "Associação fraca/moderada/forte conforme Cramér's V",
    }


def bootstrap_proportion_ci(
    series: pd.Series,
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, Any]:
    """IC bootstrap para proporção de valores == 1 (classe positiva)."""
    rng = np.random.default_rng(random_state)
    values = series.dropna().astype(float).values
    if len(values) == 0:
        return {"n": 0, "proportion": None, "ci_low": None, "ci_high": None}
    prop = float(np.mean(values == 1))
    boots = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(float(np.mean(sample == 1)))
    low = float(np.percentile(boots, 100 * alpha / 2))
    high = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {
        "method": "bootstrap_percentile",
        "n": int(len(values)),
        "proportion": prop,
        "ci_low": low,
        "ci_high": high,
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
    }


def run_statistical_validation(
    df: pd.DataFrame,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Validação estatística descritiva/inferencial para o run."""
    results: dict[str, Any] = {"tests": []}
    if "Cluster" in df.columns and "Serviço de Saúde" in df.columns:
        results["tests"].append(
            chi_square_association(df["Cluster"], df["Serviço de Saúde"])
        )
        results["cramers_v_cluster_service"] = results["tests"][-1]["cramers_v"]
    if "Serviço de Saúde" in df.columns:
        results["target_bootstrap_ci"] = bootstrap_proportion_ci(
            df["Serviço de Saúde"], random_state=seed
        )
    return results
