from __future__ import annotations

import numpy as np
import pandas as pd

from pns_health_research.config.models import PreprocessingConfig
from pns_health_research.data.schema import (
    LEGACY_WEIGHT_OUTPUT,
    ML_SAMPLE_WEIGHT_ALIAS,
    SURVEY_WEIGHT_ALIAS,
    SURVEY_WEIGHT_COLUMN,
    UF_TO_CODE,
)


def compute_legacy_survey_weight(df: pd.DataFrame) -> pd.Series:
    """Legacy R formula: 1/rowSums(UF_code, Extrato, UPA_PNS, Ordem, Nº Ordem)."""
    uf_codes = df["UF"].map(UF_TO_CODE).fillna(-1)
    components = pd.DataFrame(
        {
            "uf": uf_codes,
            "extrato": pd.to_numeric(df["Extrato"], errors="coerce"),
            "upa": pd.to_numeric(df.get("Peso (V0091)", df.get("UPA_PNS")), errors="coerce"),
            "ordem_dom": pd.to_numeric(df["Ordem do Domicílio"], errors="coerce"),
            "num_ordem": pd.to_numeric(df["Número de Ordem"], errors="coerce"),
        }
    )
    row_sums = components.sum(axis=1, min_count=1)
    return 1.0 / row_sums.replace(0, np.nan)


def compute_survey_weight_v2(df: pd.DataFrame) -> pd.Series:
    if SURVEY_WEIGHT_COLUMN in df.columns:
        return pd.to_numeric(df[SURVEY_WEIGHT_COLUMN], errors="coerce")
    # fallback to renamed column if present
    if LEGACY_WEIGHT_OUTPUT in df.columns and df[LEGACY_WEIGHT_OUTPUT].notna().any():
        w = pd.to_numeric(df[LEGACY_WEIGHT_OUTPUT], errors="coerce")
        if w.notna().sum() > len(df) * 0.5:
            return w
    return compute_legacy_survey_weight(df)


def normalize_ml_weights(weights: pd.Series) -> pd.Series:
    w = weights.astype(float)
    return w / w.mean()


def attach_weights(
    df: pd.DataFrame, *, legacy_formula: bool = True
) -> pd.DataFrame:
    out = df.copy()
    if legacy_formula:
        out[SURVEY_WEIGHT_ALIAS] = compute_legacy_survey_weight(out)
    else:
        out[SURVEY_WEIGHT_ALIAS] = compute_survey_weight_v2(out)
    out[LEGACY_WEIGHT_OUTPUT] = out[SURVEY_WEIGHT_ALIAS]
    out[ML_SAMPLE_WEIGHT_ALIAS] = normalize_ml_weights(out[SURVEY_WEIGHT_ALIAS].fillna(1.0))
    return out


def _yes_no_binary(series: pd.Series, yes_value: str = "Sim") -> pd.Series:
    return series.apply(lambda x: 0 if pd.isna(x) else int(x == yes_value))


def transform_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mappings = {
        "Cadastro UBS": ("Sim", None),
        "Agente Comunitário": (None, "Nunca recebeu"),
        "Ler e Escrever": ("Sim", None),
        "Escola": ("Sim", None),
        "Trabalha": ("Sim", None),
        "Plano de Saúde": ("Sim", None),
        "Serviço de Saúde": ("Sim", None),
        "Internação": ("Sim", None),
        "Álcool": ("Uma vez ou mais por mês", None),
        "Exercício Físico": ("Sim", None),
        "Tabaco": (None, "Não fumo atualmente"),
    }
    for col, (yes, _no_val) in mappings.items():
        if col not in out.columns:
            continue
        if col == "Agente Comunitário":
            out[col] = out[col].apply(
                lambda x: 0 if pd.isna(x) else int(x != "Nunca recebeu")
            )
        elif col == "Tabaco":
            out[col] = out[col].apply(
                lambda x: 0 if pd.isna(x) else int(x != "Não fumo atualmente")
            )
        elif col == "Álcool":
            out[col] = out[col].apply(
                lambda x: 0 if pd.isna(x) else int(x == "Uma vez ou mais por mês")
            )
        else:
            out[col] = _yes_no_binary(out[col], yes or "Sim")

    disease_cols = [
        "Colesterol Alto",
        "AVC ou Derrame",
        "Asma",
        "Artrite ou Reumatismo",
        "Dor na Coluna",
        "DORT",
        "Depressão",
        "Hipertensão",
        "Diabetes",
        "Cardiopatia",
        "Câncer",
        "Insuficiência Renal Crônica",
        "Doença Mental",
        "Doença Crônica no Pulmão",
    ]
    for col in disease_cols:
        if col in out.columns:
            out[col] = _yes_no_binary(out[col])
    return out


def reinforce_gestational_and_age(
    df: pd.DataFrame, *, impute_age_by_target: bool = True
) -> pd.DataFrame:
    out = df.copy()
    if impute_age_by_target and "Idade" in out.columns and "Serviço de Saúde" in out.columns:
        svc = out["Serviço de Saúde"]
        for val in (1, 0):
            mask = (svc == val) & out["Idade"].isna()
            if mask.any():
                mean_age = round(out.loc[svc == val, "Idade"].mean())
                out.loc[mask, "Idade"] = mean_age
    if "Diabetes na Gestação" in out.columns:
        gest_diabetes = out["Diabetes na Gestação"].eq("Sim").fillna(False)
        base_diabetes = out["Diabetes"] if "Diabetes" in out.columns else 0
        out["Diabetes"] = np.where(gest_diabetes, 1, base_diabetes)
    if "Hipertensão na Gestação" in out.columns:
        gest_hiper = out["Hipertensão na Gestação"].eq("Sim").fillna(False)
        base_hiper = out["Hipertensão"] if "Hipertensão" in out.columns else 0
        out["Hipertensão"] = np.where(gest_hiper, 1, base_hiper)
    drop_cols = [c for c in ("Diabetes na Gestação", "Hipertensão na Gestação") if c in out.columns]
    return out.drop(columns=drop_cols)


def create_binary_features(
    df: pd.DataFrame,
    *,
    config: PreprocessingConfig | None = None,
) -> pd.DataFrame:
    from pns_health_research.preprocessing.derived import apply_derived_binary
    from pns_health_research.preprocessing.features_tcc import create_binary_features_tcc

    one_hot = True if config is None else config.stages.one_hot_and_numeric.enabled
    out = create_binary_features_tcc(df, one_hot=one_hot)
    return apply_derived_binary(out, config)


def fill_missing_binary(
    df: pd.DataFrame, fill_value: int = 0
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    fill_stats: dict[str, int] = {}
    num_cols = out.select_dtypes(include="number").columns
    for col in num_cols:
        n_missing = int(out[col].isna().sum())
        if n_missing:
            fill_stats[str(col)] = n_missing
    out[num_cols] = out[num_cols].fillna(fill_value)
    obj_cols = out.select_dtypes(exclude="number").columns
    for c in obj_cols:
        n_missing = int((out[c].isna() | (out[c] == "")).sum())
        if n_missing:
            fill_stats[str(c)] = n_missing
        out[c] = out[c].fillna(fill_value).replace("", fill_value)
    return out, {
        "columns_with_imputation": fill_stats,
        "total_cells_imputed": sum(fill_stats.values()),
        "fill_value_binary": fill_value,
    }


def transform_categorical_numeric(
    df: pd.DataFrame,
    *,
    config: PreprocessingConfig | None = None,
) -> pd.DataFrame:
    from pns_health_research.preprocessing.derived import apply_derived_categorical
    from pns_health_research.preprocessing.features_tcc import transform_categorical_numeric_tcc

    one_hot = True if config is None else config.stages.one_hot_and_numeric.enabled
    out = transform_categorical_numeric_tcc(df, one_hot=one_hot)
    return apply_derived_categorical(out, config)
