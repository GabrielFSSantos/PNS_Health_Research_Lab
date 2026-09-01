from __future__ import annotations

from pathlib import Path

import pandas as pd

from pns_health_research.core.exceptions import MissingDatasetError
from pns_health_research.data.catalog import default_raw_candidates, resolve_raw_path


def load_raw_pns(
    path: str | Path | None = None,
    *,
    dataset: str = "pns",
    year: int = 2019,
) -> pd.DataFrame:
    if path is not None:
        p = Path(path)
        resolved = p if p.is_absolute() else Path(__file__).resolve().parents[3] / p
        if resolved.exists():
            return _read_file(resolved)

    candidates = [
        resolve_raw_path(dataset=dataset, year=year, source=None),
        *default_raw_candidates(year),
    ]
    for candidate in candidates:
        if candidate.exists():
            return _read_file(candidate)

    raise MissingDatasetError(
        f"PNS {year} não encontrada. Execute: ./scripts/run_preprocess.sh"
    )


def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)
