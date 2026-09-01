from pathlib import Path

import pandas as pd

from pns_health_research.data.catalog import resolve_edition
from pns_health_research.data.download import validate_raw_parquet


def test_validate_raw_parquet_synthetic(tmp_path: Path):
    edition = resolve_edition(2019)
    df = pd.DataFrame(
        {c: list(range(120)) for c in list(edition.columns.keys())[:10]}
    )
    path = tmp_path / "pns_2019.parquet"
    df.to_parquet(path)
    validate_raw_parquet(path, edition)
