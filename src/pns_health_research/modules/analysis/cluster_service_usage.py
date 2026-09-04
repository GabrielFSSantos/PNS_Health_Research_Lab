from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from pns_health_research.data.schema import LEGACY_WEIGHT_OUTPUT

_META_COLUMNS = frozenset({"Cluster", "survey_weight", LEGACY_WEIGHT_OUTPUT})

_ONE_HOT_PREFIX_GROUPS: dict[str, str] = {
    "Saúde ": "Estado de Saúde (autoavaliação)",
    "Idade ": "Faixa etária",
    "Grau ": "Grau de ensino",
}

_SEX_GROUP = ("Homem", "Mulher")


@dataclass(frozen=True)
class VizVariableOption:
    """Opção de variável para seletores do dashboard."""

    key: str
    label: str
    columns: tuple[str, ...]
    kind: str  # "binary" | "one_hot"


def _is_binary_column(series: pd.Series) -> bool:
    values = series.dropna().unique()
    if len(values) == 0:
        return False
    return set(values).issubset({0, 1, 0.0, 1.0})


def discover_viz_variables(
    df: pd.DataFrame,
    *,
    cluster_col: str = "Cluster",
) -> dict[str, list[VizVariableOption]]:
    """Lista colunas binárias e grupos one-hot para visualização."""
    binary_cols: list[str] = []
    grouped: dict[str, list[str]] = {}
    used: set[str] = set()

    for col in df.columns:
        if col in _META_COLUMNS or col == cluster_col:
            continue
        if not _is_binary_column(df[col]):
            continue
        binary_cols.append(col)

    for prefix, label in _ONE_HOT_PREFIX_GROUPS.items():
        cols = sorted(c for c in binary_cols if c.startswith(prefix))
        if len(cols) >= 2:
            key = f"group:{prefix.strip()}"
            grouped[key] = cols
            used.update(cols)

    sex_cols = [c for c in _SEX_GROUP if c in binary_cols]
    if len(sex_cols) == 2:
        grouped["group:sexo"] = sex_cols
        used.update(sex_cols)

    binary_options = [
        VizVariableOption(key=col, label=col, columns=(col,), kind="binary")
        for col in sorted(binary_cols)
    ]
    group_options = [
        VizVariableOption(
            key=key,
            label=_ONE_HOT_PREFIX_GROUPS.get(
                key.removeprefix("group:") + " ", key.removeprefix("group:")
            )
            if key != "group:sexo"
            else "Sexo",
            columns=tuple(cols),
            kind="one_hot",
        )
        for key, cols in sorted(grouped.items(), key=lambda x: x[0])
    ]
    # Corrigir labels dos grupos por prefixo
    fixed_groups: list[VizVariableOption] = []
    for opt in group_options:
        if opt.key == "group:sexo":
            fixed_groups.append(opt)
            continue
        prefix = opt.key.removeprefix("group:")
        label = _ONE_HOT_PREFIX_GROUPS.get(f"{prefix} ", prefix)
        fixed_groups.append(
            VizVariableOption(key=opt.key, label=label, columns=opt.columns, kind="one_hot")
        )

    standalone_binary = [
        VizVariableOption(key=col, label=col, columns=(col,), kind="binary")
        for col in sorted(set(binary_cols) - used)
    ]

    return {
        "split": binary_options,
        "category": fixed_groups + standalone_binary,
    }


def _category_label(col: str) -> str:
    for prefix in _ONE_HOT_PREFIX_GROUPS:
        if col.startswith(prefix):
            return col[len(prefix) :]
    return col


def bubble_cross_table(
    df: pd.DataFrame,
    split_col: str,
    category_cols: list[str] | tuple[str, ...],
    *,
    cluster_col: str = "Cluster",
) -> pd.DataFrame:
    """Agrega contagens para gráfico de bolhas: split × categoria por cluster."""
    if cluster_col not in df.columns or split_col not in df.columns:
        return pd.DataFrame()
    cols = [c for c in category_cols if c in df.columns]
    if not cols:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for cluster, grp in df.groupby(cluster_col):
        for split_val, split_label in ((1, "Sim"), (0, "Não")):
            sub = grp[grp[split_col] == split_val]
            if len(cols) == 1 and cols[0] == split_col:
                n = len(sub)
                if n > 0:
                    rows.append(
                        {
                            "cluster": int(cluster),
                            "split_label": split_label,
                            "category": "Sim" if split_val == 1 else "Não",
                            "count": n,
                        }
                    )
                continue
            for ccol in cols:
                n = int((sub[ccol] == 1).sum())
                if n > 0:
                    rows.append(
                        {
                            "cluster": int(cluster),
                            "split_label": split_label,
                            "category": _category_label(ccol),
                            "count": n,
                        }
                    )
    return pd.DataFrame(rows)


def bubble_health_service_table(
    df: pd.DataFrame,
    *,
    service_col: str = "Serviço de Saúde",
    cluster_col: str = "Cluster",
) -> pd.DataFrame:
    """Wrapper legado: serviço × autoavaliação de saúde por cluster."""
    health_cols = [c for c in df.columns if c.startswith("Saúde ")]
    out = bubble_cross_table(df, service_col, health_cols, cluster_col=cluster_col)
    if out.empty:
        return out
    return out.rename(
        columns={
            "split_label": "servico",
            "category": "categoria_saude",
        }
    )


def rate_by_cluster(
    df: pd.DataFrame,
    variable_col: str,
    *,
    cluster_col: str = "Cluster",
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Taxa ponderada ou simples de uma variável binária por cluster."""
    if cluster_col not in df.columns or variable_col not in df.columns:
        return pd.DataFrame()
    weight_col = weight_col or (
        "survey_weight" if "survey_weight" in df.columns else LEGACY_WEIGHT_OUTPUT
    )
    rows = []
    for cluster, grp in df.groupby(cluster_col):
        if weight_col in grp.columns:
            w = grp[weight_col]
            rate = (grp[variable_col] * w).sum() / w.sum()
        else:
            rate = grp[variable_col].mean()
        rows.append(
            {
                "cluster": int(cluster),
                "n": len(grp),
                "rate": float(rate),
                "service_usage_rate": float(rate),
                "variable": variable_col,
                "weighted": weight_col in grp.columns,
            }
        )
    return pd.DataFrame(rows)


def service_usage_by_cluster(
    df: pd.DataFrame,
    *,
    service_col: str = "Serviço de Saúde",
    cluster_col: str = "Cluster",
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Compatibilidade: taxa de uso de serviço de saúde por cluster."""
    return rate_by_cluster(
        df, service_col, cluster_col=cluster_col, weight_col=weight_col
    )


def cross_rate_table(
    df: pd.DataFrame,
    primary_col: str,
    secondary_cols: list[str] | tuple[str, ...],
    *,
    cluster_col: str = "Cluster",
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Taxa da variável primária por cluster × categoria da variável secundária."""
    if cluster_col not in df.columns or primary_col not in df.columns:
        return pd.DataFrame()
    cols = [c for c in secondary_cols if c in df.columns]
    if not cols:
        return pd.DataFrame()

    weight_col = weight_col or (
        "survey_weight" if "survey_weight" in df.columns else LEGACY_WEIGHT_OUTPUT
    )
    use_weight = weight_col in df.columns
    rows: list[dict[str, Any]] = []

    for cluster, grp in df.groupby(cluster_col):
        if len(cols) == 1:
            sec_col = cols[0]
            for sec_val, sec_label in ((1, "Sim"), (0, "Não")):
                sub = grp[grp[sec_col] == sec_val]
                if sub.empty:
                    continue
                if use_weight:
                    w = sub[weight_col]
                    rate = (sub[primary_col] * w).sum() / w.sum()
                else:
                    rate = sub[primary_col].mean()
                rows.append(
                    {
                        "cluster": int(cluster),
                        "secondary_category": sec_label,
                        "rate": float(rate),
                        "n": len(sub),
                        "primary_variable": primary_col,
                        "secondary_variable": sec_col,
                    }
                )
            continue

        for sec_col in cols:
            sub = grp[grp[sec_col] == 1]
            if sub.empty:
                continue
            if use_weight:
                w = sub[weight_col]
                rate = (sub[primary_col] * w).sum() / w.sum()
            else:
                rate = sub[primary_col].mean()
            rows.append(
                {
                    "cluster": int(cluster),
                    "secondary_category": _category_label(sec_col),
                    "rate": float(rate),
                    "n": len(sub),
                    "primary_variable": primary_col,
                    "secondary_variable": sec_col,
                }
            )
    return pd.DataFrame(rows)


def cluster_profile_table(profiles: pd.DataFrame) -> pd.DataFrame:
    return profiles.reset_index()


def normalize_cluster_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Garante coluna cluster/Cluster em perfis largos (dashboard e API)."""
    if profiles.empty:
        return profiles
    if "cluster" in profiles.columns or "Cluster" in profiles.columns:
        return profiles
    if profiles.index.name in ("cluster", "Cluster"):
        return profiles.reset_index()
    if profiles.index.dtype.kind in ("i", "u") and not isinstance(profiles.index, pd.RangeIndex):
        out = profiles.reset_index()
        if out.columns[0] not in ("cluster", "Cluster"):
            out = out.rename(columns={out.columns[0]: "cluster"})
        return out
    out = profiles.copy()
    out.insert(0, "cluster", range(len(out)))
    return out
