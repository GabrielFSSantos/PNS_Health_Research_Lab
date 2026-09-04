"""Snapshot agregado de variáveis da pesquisa (pré-one-hot)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pns_health_research.core.paths import CONFIGS_DIR
from pns_health_research.modules.preprocessing.features_tcc import DESIGN_COLUMNS, INTERNAL_WEIGHT_COLS

SKIP_COLUMNS = set(DESIGN_COLUMNS) | set(INTERNAL_WEIGHT_COLS) | {"Peso Amostral"}


def load_survey_metadata(path: Path | None = None) -> dict[str, Any]:
    p = path or CONFIGS_DIR / "data" / "survey_metadata.yaml"
    if not p.exists():
        return {"variables": {}}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {"variables": {}}


def build_survey_snapshot(
    df: pd.DataFrame,
    *,
    max_values: int = 25,
) -> dict[str, Any]:
    """Contagens por variável após filtro de elegibilidade, antes do one-hot."""
    metadata = load_survey_metadata()
    meta_vars = metadata.get("variables") or {}
    variables: dict[str, Any] = {}

    for col in df.columns:
        if col in SKIP_COLUMNS:
            continue
        meta = meta_vars.get(col) or {}
        if meta.get("used_in_study") is False:
            continue

        series = df[col]
        missing = int(series.isna().sum())
        vc = series.astype(str).replace("nan", "(ausente)").value_counts().head(max_values)
        variables[col] = {
            "code": meta.get("code"),
            "question": meta.get("question"),
            "value_labels": meta.get("value_labels") or {},
            "type": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
            "counts": {str(k): int(v) for k, v in vc.items()},
            "missing": missing,
            "total": len(df),
        }

    return {
        "rows": len(df),
        "variable_count": len(variables),
        "variables": variables,
    }
