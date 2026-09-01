#!/usr/bin/env python3
"""Backfill clustered_categorical.parquet em run anterior (one-off)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pns_health_research.clustering.pipelines import run_clustering_stage
from pns_health_research.config.loader import load_config
from pns_health_research.core.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, RUNS_DIR


def backfill(run_id: str, config_path: Path) -> Path:
    cfg = load_config(config_path)
    binary_path = PROCESSED_DATA_DIR / f"{run_id}_binary.parquet"
    categorical_path = PROCESSED_DATA_DIR / f"{run_id}_categorical.parquet"
    if not binary_path.exists() or not categorical_path.exists():
        raise FileNotFoundError(
            f"Parquets processados ausentes para {run_id} em {PROCESSED_DATA_DIR}"
        )
    binary_df = pd.read_parquet(binary_path)
    categorical_df = pd.read_parquet(categorical_path)
    clust = run_clustering_stage(binary_df, categorical_df, cfg)
    out = RUNS_DIR / run_id / "tables" / "clustered_categorical.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    clust.categorical_with_cluster.to_parquet(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="ID do run com clustering válido")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/legacy_reproduction.yaml",
    )
    args = parser.parse_args()
    path = backfill(args.run_id, args.config)
    print(f"Escrito: {path}")


if __name__ == "__main__":
    main()
