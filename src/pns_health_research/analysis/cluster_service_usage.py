from __future__ import annotations

import pandas as pd

from pns_health_research.data.schema import LEGACY_WEIGHT_OUTPUT


def service_usage_by_cluster(
    df: pd.DataFrame,
    *,
    service_col: str = "Serviço de Saúde",
    cluster_col: str = "Cluster",
    weight_col: str | None = None,
) -> pd.DataFrame:
    weight_col = weight_col or (
        "survey_weight" if "survey_weight" in df.columns else LEGACY_WEIGHT_OUTPUT
    )
    rows = []
    for cluster, grp in df.groupby(cluster_col):
        if weight_col in grp.columns:
            w = grp[weight_col]
            rate = (grp[service_col] * w).sum() / w.sum()
        else:
            rate = grp[service_col].mean()
        rows.append(
            {
                "cluster": int(cluster),
                "n": len(grp),
                "service_usage_rate": float(rate),
                "weighted": weight_col in grp.columns,
            }
        )
    return pd.DataFrame(rows)


def cluster_profile_table(profiles: pd.DataFrame) -> pd.DataFrame:
    return profiles.reset_index()
