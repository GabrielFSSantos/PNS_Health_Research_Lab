from __future__ import annotations

from pathlib import Path

import pandas as pd

from pns_health_research.core.exceptions import MissingDatasetError
from pns_health_research.core.paths import RAW_DATA_DIR


def load_raw_pns(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        for candidate in (
            RAW_DATA_DIR / "pns2019.parquet",
            RAW_DATA_DIR / "pns2019.csv",
        ):
            if candidate.exists():
                path = candidate
                break
    if path is None or not Path(path).exists():
        raise MissingDatasetError(
            "PNS dataset not found. Place data at data/raw/pns2019.parquet "
            "— see docs/data_required.md"
        )
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)
