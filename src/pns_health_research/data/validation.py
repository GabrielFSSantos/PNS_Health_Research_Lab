from __future__ import annotations

import pandas as pd

from pns_health_research.core.exceptions import MissingRequiredColumnError
from pns_health_research.data.schema import LEGACY_RAW_COLUMNS, RECORD_ID_COLUMN


def add_record_id(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    present = [c for c in id_cols if c in out.columns]
    if not present:
        out[RECORD_ID_COLUMN] = out.index.astype(str)
        return out
    out[RECORD_ID_COLUMN] = out[present].astype(str).agg("|".join, axis=1)
    return out


def validate_legacy_columns(df: pd.DataFrame) -> None:
    missing = [c for c in LEGACY_RAW_COLUMNS if c not in df.columns]
    if len(missing) == len(LEGACY_RAW_COLUMNS):
        raise MissingRequiredColumnError(
            f"No legacy PNS columns found. Missing all: {missing[:5]}..."
        )
