"""Optional R/PNSIBGE acquisition — requires extra [acquisition]."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pns_health_research.core.exceptions import MissingDatasetError
from pns_health_research.core.paths import RAW_DATA_DIR


def _is_na_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return type(value).__name__.startswith("NA")


def _parquet_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]) and out[col].dtype.name != "category":
            continue
        out[col] = out[col].map(
            lambda v: None
            if _is_na_scalar(v)
            else (v if isinstance(v, (bool, int, float, np.integer, np.floating)) else str(v))
        )
        out[col] = out[col].astype("string")
    return out


def acquire_pns2019(*, year: int = 2019, output: Path | None = None) -> Path:
    try:
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise MissingDatasetError(
            "Instale o extra acquisition: uv sync --extra acquisition"
        ) from exc

    output = output or RAW_DATA_DIR / "pns2019.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)

    pnsibge = importr("PNSIBGE")
    df_r = pnsibge.get_pns(year=year, design=False)
    import rpy2.robjects.pandas2ri as pandas2ri
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter

    with localconverter(default_converter + pandas2ri.converter):
        df = pandas2ri.rpy2py(df_r)

    df = _parquet_safe_dataframe(df)
    df.to_parquet(output, index=False)
    return output
