from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pns_health_research.analysis.cluster_service_usage import (
    cluster_profile_table,
    service_usage_by_cluster,
)
from pns_health_research.classification.pipelines import run_classification_stage
from pns_health_research.clustering.pipelines import run_clustering_stage
from pns_health_research.config.models import ExperimentConfig
from pns_health_research.core.paths import PROCESSED_DATA_DIR, RUNS_DIR, ensure_dirs
from pns_health_research.core.reproducibility import configure_logging, set_global_seed
from pns_health_research.data.fingerprint import fingerprint_dataframe
from pns_health_research.experiments.reproduction import build_reproduction_report
from pns_health_research.persistence.artifact_store import (
    ArtifactStore,
    build_manifest,
    make_run_id,
)
from pns_health_research.persistence.run_loader import load_clustered_categorical, load_run_metrics
from pns_health_research.preprocessing.pipelines import run_preprocessing


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        ensure_dirs()
        self.run_id = make_run_id(config.experiment.name)
        self.run_dir = RUNS_DIR / self.run_id
        self.store = ArtifactStore(self.run_dir)
        self.logger = configure_logging(run_id=self.run_id)
        self.started_at = datetime.now(UTC).isoformat()
        self.performance: dict[str, Any] = {}
        self.source_run_id: str | None = config.classification.input_run

    def run(self) -> Path:
        set_global_seed(self.config.experiment.seed)
        cfg_dict = self.config.model_dump()
        if self.source_run_id:
            cfg_dict["source_run_id"] = self.source_run_id
        self.store.write_yaml("config.resolved.yaml", cfg_dict)

        preprocessing_metrics: dict = {}
        clustering_metrics: dict = {}
        classification_metrics: dict = {}
        data_fp = "resume"
        clustered_cat: pd.DataFrame | None = None

        resume_classification = (
            self.source_run_id is not None
            and "classification" in self.config.stages
            and "clustering" not in self.config.stages
            and "preprocessing" not in self.config.stages
        )

        if resume_classification:
            self.logger.info(
                "resuming classification from run_id=%s", self.source_run_id
            )
            clustered_cat = load_clustered_categorical(self.source_run_id)
            preprocessing_metrics = load_run_metrics(self.source_run_id, "preprocessing.json")
            clustering_metrics = load_run_metrics(self.source_run_id, "clustering.json")
            source_fp_path = RUNS_DIR / self.source_run_id / "data_fingerprint.json"
            if source_fp_path.exists():
                import json

                data_fp = json.loads(source_fp_path.read_text()).get("hash", data_fp)
        else:
            t0 = time.perf_counter()
            prep = run_preprocessing(self.config)
            self.performance["preprocessing_seconds"] = time.perf_counter() - t0
            preprocessing_metrics = prep.metrics
            self.store.write_json("metrics/preprocessing.json", prep.metrics)

            fp = fingerprint_dataframe(prep.binary_df, "binary_processed")
            data_fp = fp["hash"]
            self.store.write_json("data_fingerprint.json", fp)

            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            prep.binary_df.to_parquet(PROCESSED_DATA_DIR / f"{self.run_id}_binary.parquet")
            prep.categorical_df.to_parquet(
                PROCESSED_DATA_DIR / f"{self.run_id}_categorical.parquet"
            )

            if "clustering" in self.config.stages:
                t1 = time.perf_counter()
                clust = run_clustering_stage(prep.binary_df, prep.categorical_df, self.config)
                self.performance["clustering_seconds"] = time.perf_counter() - t1
                clustering_metrics = clust.metrics
                self.store.write_json("metrics/clustering.json", clust.metrics)
                self.store.write_parquet("tables/elbow_results.parquet", clust.elbow_results)
                self.store.write_parquet(
                    "tables/pca_explained_variance.parquet",
                    clust.pca_variance,
                )
                self.store.write_parquet(
                    "tables/cluster_profiles.parquet",
                    cluster_profile_table(clust.profiles),
                )
                self.store.write_parquet(
                    "tables/cross_analysis.parquet",
                    service_usage_by_cluster(clust.binary_with_cluster),
                )
                clustered_cat = clust.categorical_with_cluster
                self.store.write_parquet(
                    "tables/clustered_categorical.parquet", clustered_cat
                )
            elif "classification" in self.config.stages:
                clustered_cat = prep.categorical_df

        if "classification" in self.config.stages and clustered_cat is not None:
            t2 = time.perf_counter()
            clf = run_classification_stage(clustered_cat, self.config)
            self.performance["classification_seconds"] = time.perf_counter() - t2
            classification_metrics = clf.metrics
            self.store.write_json("metrics/classification.json", clf.metrics)
            for name, res in clf.model_results.items():
                from pns_health_research.classification.validation import ModelResult

                if isinstance(res, ModelResult):
                    self.store.write_parquet(f"predictions/{name}.parquet", res.predictions)

        if (
            "cross_analysis" in self.config.stages
            and clustered_cat is not None
            and "Cluster" in clustered_cat.columns
            and not resume_classification
        ):
            self.store.write_json("metrics/cross_analysis.json", {"status": "tables persisted"})

        if (
            "statistics" in self.config.stages
            and clustered_cat is not None
            and "Cluster" in clustered_cat.columns
        ):
            from pns_health_research.statistics.descriptive import cramers_v

            stats_metrics: dict[str, float] = {}
            if "Serviço de Saúde" in clustered_cat.columns:
                stats_metrics["cramers_v_cluster_service"] = cramers_v(
                    clustered_cat["Cluster"], clustered_cat["Serviço de Saúde"]
                )
            self.store.write_json("metrics/statistics.json", stats_metrics)

        self.performance["total_seconds"] = sum(
            v for k, v in self.performance.items() if k.endswith("_seconds")
        )
        if self.performance:
            self.store.write_json("metrics/performance.json", self.performance)

        report = build_reproduction_report(
            self.config, preprocessing_metrics, clustering_metrics, classification_metrics
        )
        if self.source_run_id:
            report["source_run_id"] = self.source_run_id
        self.store.write_json("reproduction_report.json", report)

        finished = datetime.now(UTC).isoformat()
        manifest = build_manifest(
            self.run_id,
            self.config,
            "completed",
            self.started_at,
            finished,
            data_fp,
        )
        if self.source_run_id:
            manifest["source_run_id"] = self.source_run_id
        self.store.write_json("manifest.json", manifest)
        self.logger.info("run completed run_id=%s status=completed", self.run_id)
        return self.run_dir


def compare_runs(baseline_dir: Path, candidate_dir: Path) -> pd.DataFrame:
    rows = []
    for metric_file in ("preprocessing.json", "clustering.json", "classification.json"):
        b_path = baseline_dir / "metrics" / metric_file
        c_path = candidate_dir / "metrics" / metric_file
        if not b_path.exists() or not c_path.exists():
            continue
        import json

        b_data = json.loads(b_path.read_text())
        c_data = json.loads(c_path.read_text())
        if isinstance(b_data, dict) and isinstance(c_data, dict):
            for key in set(b_data) & set(c_data):
                if isinstance(b_data[key], (int, float)) and isinstance(c_data[key], (int, float)):
                    rows.append(
                        {
                            "metric": f"{metric_file}:{key}",
                            "baseline": b_data[key],
                            "candidate": c_data[key],
                            "absolute_difference": c_data[key] - b_data[key],
                        }
                    )
    return pd.DataFrame(rows)
