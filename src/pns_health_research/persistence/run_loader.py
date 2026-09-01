from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pns_health_research.core.paths import RUNS_DIR


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def load_clustered_categorical(run_id: str) -> pd.DataFrame:
    path = run_dir(run_id) / "tables" / "clustered_categorical.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"clustered_categorical.parquet ausente em {path}. "
            "Execute clustering ou backfill antes de resumir classificação."
        )
    return pd.read_parquet(path)


def load_run_metrics(run_id: str, metric_name: str) -> dict:
    path = run_dir(run_id) / "metrics" / metric_name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
