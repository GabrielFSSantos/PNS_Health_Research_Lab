#!/usr/bin/env python3
"""Regenera artefatos de dataset no showcase legacy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pns_health_research.analysis.cluster_service_usage import bubble_health_service_table
from pns_health_research.clustering.pipelines import run_clustering_stage
from pns_health_research.config.loader import load_config
from pns_health_research.core.paths import ARTIFACTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, RUNS_DIR
from pns_health_research.persistence.showcase import SHOWCASE_DIR
from pns_health_research.preprocessing.pipelines import run_preprocessing
from pns_health_research.preprocessing.viz_artifacts import (
    build_dataset_distributions,
    build_preprocess_funnel,
)

DEFAULT_SHOWCASE = "pesquisa-original-legacy"
LEGACY_CONFIG = PROJECT_ROOT / "configs/experiments/legacy.yaml"


def _resolve_binary_parquet(run_id: str | None) -> Path:
    if run_id:
        path = PROCESSED_DATA_DIR / f"{run_id}_binary.parquet"
        if path.exists():
            return path
    for path in (PROCESSED_DATA_DIR / "binary_latest.parquet",):
        if path.exists():
            return path
    raise FileNotFoundError("Parquet binário não encontrado — execute o pré-processamento.")


def _resolve_run_id(showcase_dir: Path, run_id: str | None) -> str | None:
    if run_id:
        return run_id
    manifest = showcase_dir / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return data.get("run_id")
    return None


def _build_clustered_binary(run_id: str, binary_df: pd.DataFrame) -> pd.DataFrame | None:
    run_dir = RUNS_DIR / run_id
    clustered_binary_path = run_dir / "tables" / "clustered_binary.parquet"
    if clustered_binary_path.exists():
        return pd.read_parquet(clustered_binary_path)

    clustered_cat_path = run_dir / "tables" / "clustered_categorical.parquet"
    if clustered_cat_path.exists():
        clustered = pd.read_parquet(clustered_cat_path)
        if "Cluster" in clustered.columns and len(clustered) == len(binary_df):
            work = binary_df.copy()
            work["Cluster"] = clustered["Cluster"].values
            return work

    binary_path = run_dir / "tables" / "clustered_binary.parquet"
    showcase_binary = ARTIFACTS_DIR / "showcase" / "pesquisa-original-legacy" / "tables" / "clustered_binary.parquet"
    if binary_path.exists():
        return pd.read_parquet(binary_path)

    processed = PROCESSED_DATA_DIR / f"{run_id}_binary.parquet"
    if processed.exists() and clustered_cat_path.exists():
        binary_legacy = pd.read_parquet(processed)
        clustered = pd.read_parquet(clustered_cat_path)
        if "Cluster" in clustered.columns:
            work = binary_legacy.copy()
            n = min(len(work), len(clustered))
            work = work.iloc[:n].copy()
            work["Cluster"] = clustered["Cluster"].values[:n]
            return work
    return None


def _cluster_from_preprocessing(
    binary_df: pd.DataFrame, categorical_df: pd.DataFrame, config_path: Path
):
    cfg = load_config(config_path)
    if "clustering" not in cfg.stages:
        return None
    return run_clustering_stage(binary_df, categorical_df, cfg)


def backfill(
    *,
    run_id: str | None = None,
    showcase_label: str = DEFAULT_SHOWCASE,
    config_path: Path = LEGACY_CONFIG,
) -> None:
    showcase_dir = SHOWCASE_DIR / showcase_label
    if not showcase_dir.exists():
        raise FileNotFoundError(f"Showcase não encontrado: {showcase_dir}")

    cfg = load_config(config_path)
    prep = run_preprocessing(cfg)
    binary_df = prep.binary_df

    metrics_dir = showcase_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    distributions = build_dataset_distributions(binary_df)
    dist_path = metrics_dir / "dataset_distributions.json"
    dist_path.write_text(json.dumps(distributions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {dist_path} ({len(distributions)} variáveis)")

    funnel = build_preprocess_funnel(
        prep.filter_stats,
        prep.metrics,
        processing_stats=prep.processing_stats,
    )
    funnel_path = metrics_dir / "preprocess_funnel.json"
    funnel_path.write_text(json.dumps(funnel, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {funnel_path} ({len(funnel.get('stages', []))} estágios)")

    survey_path = metrics_dir / "survey_snapshot.json"
    survey_path.write_text(
        json.dumps(prep.survey_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"✓ {survey_path} ({prep.survey_snapshot.get('variable_count', 0)} variáveis)")

    prep_path = metrics_dir / "preprocessing.json"
    prep_path.write_text(json.dumps(prep.metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {prep_path}")

    resolved_run_id = _resolve_run_id(showcase_dir, run_id)
    tables_dir = showcase_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    clustered_binary_path = tables_dir / "clustered_binary.parquet"
    clustered_binary = None
    if resolved_run_id:
        clustered_binary = _build_clustered_binary(resolved_run_id, binary_df)
    if clustered_binary is None:
        print("↻ Gerando clustered_binary via clustering (run fonte ausente)...")
        clust = _cluster_from_preprocessing(binary_df, prep.categorical_df, config_path)
        if clust is not None:
            clustered_binary = clust.binary_with_cluster
    if clustered_binary is not None and not clustered_binary.empty:
        clustered_binary.to_parquet(clustered_binary_path, index=False)
        print(f"✓ {clustered_binary_path} ({len(clustered_binary)} linhas)")
    elif clustered_binary_path.exists():
        print(f"✓ clustered_binary já presente: {clustered_binary_path}")
    else:
        print("⚠ clustered_binary não gerado — run fonte sem labels de cluster")

    bubble_path = tables_dir / "bubble_health_service.parquet"
    if bubble_path.exists():
        print(f"✓ bubble já presente: {bubble_path}")
    elif resolved_run_id or clustered_binary is not None:
        work = clustered_binary
        if work is None and resolved_run_id:
            work = _build_clustered_binary(resolved_run_id, binary_df)
        if work is not None and "Cluster" in work.columns:
            bubble_df = bubble_health_service_table(work)
            if not bubble_df.empty:
                bubble_df.to_parquet(bubble_path, index=False)
                print(f"✓ {bubble_path} ({len(bubble_df)} linhas)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="Run fonte para bubble (opcional)")
    parser.add_argument("--showcase", default=DEFAULT_SHOWCASE)
    parser.add_argument("--config", type=Path, default=LEGACY_CONFIG)
    args = parser.parse_args()
    backfill(run_id=args.run_id, showcase_label=args.showcase, config_path=args.config)


if __name__ == "__main__":
    main()
