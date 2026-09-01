from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pns_health_research.analysis.cluster_service_usage import (
    bubble_health_service_table,
    cluster_profile_table,
    service_usage_by_cluster,
)
from pns_health_research.classification.curve_artifacts import (
    aggregate_predictions_summary,
    persist_curve_artifacts,
)
from pns_health_research.classification.persistence import (
    persist_classification_aggregates,
    persist_model_artifacts,
)
from pns_health_research.classification.pipelines import run_classification_stage
from pns_health_research.classification.validation import ModelResult
from pns_health_research.clustering.pipelines import run_clustering_stage
from pns_health_research.config.models import ExperimentConfig
from pns_health_research.core.logging import configure_logging
from pns_health_research.core.paths import PROCESSED_DATA_DIR, RUNS_DIR, ensure_dirs
from pns_health_research.core.reproducibility import set_global_seed
from pns_health_research.data.fingerprint import fingerprint_dataframe
from pns_health_research.experiments.reproduction import build_reproduction_report
from pns_health_research.persistence.artifact_store import make_run_id
from pns_health_research.persistence.run_contract import (
    build_lineage,
    build_manifest_v2,
    build_summary,
    compute_artifacts_status,
    infer_run_type,
    validate_required_artifacts,
)
from pns_health_research.persistence.run_loader import load_clustered_categorical, load_run_metrics
from pns_health_research.preprocessing.pipelines import run_preprocessing
from pns_health_research.preprocessing.viz_artifacts import (
    build_dataset_distributions,
    build_preprocess_funnel,
)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        ensure_dirs()
        self.run_id = make_run_id(config.experiment.name)
        self.run_dir = RUNS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        from pns_health_research.persistence.artifact_store import ArtifactStore

        self.store = ArtifactStore(self.run_dir)
        log_file = self.run_dir / "logs" / "run.log"
        self.logger = configure_logging(run_id=self.run_id, log_file=log_file)
        self.started_at = datetime.now(UTC).isoformat()
        self.performance: dict[str, Any] = {}
        self.source_run_id: str | None = config.classification.input_run
        self.preprocessing_run_id: str | None = None
        self.clustering_run_id: str | None = None
        self._write_initial_manifest()

    def _write_initial_manifest(self) -> None:
        manifest = build_manifest_v2(
            self.run_id,
            self.config,
            "created",
            self.started_at,
            None,
            "pending",
        )
        self.store.write_json("manifest.json", manifest)

    def _set_status(self, status: str, *, error: dict[str, Any] | None = None) -> None:
        manifest = build_manifest_v2(
            self.run_id,
            self.config,
            status,  # type: ignore[arg-type]
            self.started_at,
            datetime.now(UTC).isoformat() if status in ("completed", "failed") else None,
            getattr(self, "_data_fp", "pending"),
            lineage=getattr(self, "_lineage", {}),
            duration_seconds=sum(
                v for k, v in self.performance.items() if k.endswith("_seconds")
            )
            or None,
            error=error,
        )
        self.store.write_json("manifest.json", manifest)

    def _persist_clustering_viz(self, clust: Any) -> None:
        if not clust.pca_loadings.empty:
            self.store.write_parquet("tables/pca_loadings.parquet", clust.pca_loadings)
        if clust.pca_transformed is not None:
            n = clust.pca_transformed.shape[0]
            sample_size = min(5000, n)
            rng = np.random.default_rng(self.config.experiment.seed)
            idx = rng.choice(n, size=sample_size, replace=False)
            scatter = pd.DataFrame(
                {
                    "pc1": clust.pca_transformed[idx, 0],
                    "pc2": (
                        clust.pca_transformed[idx, 1]
                        if clust.pca_transformed.shape[1] > 1
                        else 0
                    ),
                    "cluster": clust.labels[idx],
                }
            )
            self.store.write_parquet("tables/pca_scatter_sample.parquet", scatter)
        if clust.centroids is not None:
            centroids_df = pd.DataFrame(
                clust.centroids,
                columns=[f"pc{i+1}" for i in range(clust.centroids.shape[1])],
            )
            centroids_df.insert(0, "cluster", range(len(centroids_df)))
            self.store.write_parquet("tables/centroids.parquet", centroids_df)
        if clust.silhouette_summary:
            self.store.write_json("metrics/silhouette_summary.json", clust.silhouette_summary)
        if (
            getattr(clust, "silhouette_samples", None) is not None
            and clust.pca_transformed is not None
        ):
            n = len(clust.silhouette_samples)
            sample_size = min(2000, n)
            rng = np.random.default_rng(self.config.experiment.seed)
            idx = rng.choice(n, size=sample_size, replace=False)
            sil_df = pd.DataFrame(
                {
                    "silhouette": clust.silhouette_samples[idx],
                    "cluster": clust.labels[idx],
                }
            )
            self.store.write_parquet("tables/silhouette_samples.parquet", sil_df)

    def run(self) -> Path:
        set_global_seed(self.config.experiment.seed)
        self._set_status("running")

        cfg_dict = self.config.model_dump()
        if self.source_run_id:
            cfg_dict["source_run_id"] = self.source_run_id
        self.store.write_yaml("config.resolved.yaml", cfg_dict)

        preprocessing_metrics: dict = {}
        clustering_metrics: dict = {}
        classification_metrics: dict = {}
        self._data_fp = "resume"
        clustered_cat: pd.DataFrame | None = None

        resume_classification = (
            self.source_run_id is not None
            and "classification" in self.config.stages
            and "clustering" not in self.config.stages
            and "preprocessing" not in self.config.stages
        )

        try:
            if resume_classification:
                self.logger.info(
                    "resuming classification from run_id=%s", self.source_run_id
                )
                clustered_cat = load_clustered_categorical(self.source_run_id)
                preprocessing_metrics = load_run_metrics(
                    self.source_run_id, "preprocessing.json"
                )
                clustering_metrics = load_run_metrics(self.source_run_id, "clustering.json")
                self.preprocessing_run_id = self.source_run_id
                self.clustering_run_id = self.source_run_id
                source_fp_path = RUNS_DIR / self.source_run_id / "data_fingerprint.json"
                if source_fp_path.exists():
                    import json

                    self._data_fp = json.loads(source_fp_path.read_text()).get(
                        "hash", self._data_fp
                    )
            else:
                t0 = time.perf_counter()
                prep = run_preprocessing(self.config)
                self.performance["preprocessing_seconds"] = time.perf_counter() - t0
                preprocessing_metrics = prep.metrics
                self.store.write_json("metrics/preprocessing.json", prep.metrics)
                self.store.write_json(
                    "metrics/preprocess_funnel.json",
                    build_preprocess_funnel(
                        prep.filter_stats,
                        prep.metrics,
                        processing_stats=prep.processing_stats,
                    ),
                )
                self.store.write_json(
                    "metrics/survey_snapshot.json",
                    prep.survey_snapshot,
                )
                self.store.write_json(
                    "metrics/dataset_distributions.json",
                    build_dataset_distributions(prep.binary_df),
                )

                fp = fingerprint_dataframe(prep.binary_df, "binary_processed")
                self._data_fp = fp["hash"]
                self.store.write_json("data_fingerprint.json", fp)
                self.preprocessing_run_id = self.run_id

                PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
                prep.binary_df.to_parquet(
                    PROCESSED_DATA_DIR / f"{self.run_id}_binary.parquet"
                )
                prep.categorical_df.to_parquet(
                    PROCESSED_DATA_DIR / f"{self.run_id}_categorical.parquet"
                )

                if "clustering" in self.config.stages:
                    t1 = time.perf_counter()
                    clust = run_clustering_stage(
                        prep.binary_df, prep.categorical_df, self.config
                    )
                    self.performance["clustering_seconds"] = time.perf_counter() - t1
                    clustering_metrics = clust.metrics
                    self.clustering_run_id = self.run_id
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
                    self.store.write_parquet(
                        "tables/clustered_binary.parquet", clust.binary_with_cluster
                    )
                    bubble_df = bubble_health_service_table(clust.binary_with_cluster)
                    if not bubble_df.empty:
                        self.store.write_parquet(
                            "tables/bubble_health_service.parquet", bubble_df
                        )
                    clustered_cat = clust.categorical_with_cluster
                    self.store.write_parquet(
                        "tables/clustered_categorical.parquet", clustered_cat
                    )
                    self._persist_clustering_viz(clust)
                elif "classification" in self.config.stages:
                    clustered_cat = prep.categorical_df

            clustering_used = (
                "clustering" in self.config.stages
                or resume_classification
                or self.clustering_run_id is not None
            )
            self._lineage = build_lineage(
                self.config,
                preprocessing_run_id=self.preprocessing_run_id,
                clustering_run_id=self.clustering_run_id if clustering_used else None,
                clustering_used=clustering_used,
            )

            if "classification" in self.config.stages and clustered_cat is not None:
                t2 = time.perf_counter()
                clf = run_classification_stage(clustered_cat, self.config)
                self.performance["classification_seconds"] = time.perf_counter() - t2
                classification_metrics = clf.metrics
                self.store.write_json("metrics/classification.json", clf.metrics)
                for name, res in clf.model_results.items():
                    if isinstance(res, ModelResult):
                        self.store.write_parquet(f"predictions/{name}.parquet", res.predictions)
                        persist_model_artifacts(self.store, res)
                persist_classification_aggregates(self.store, clf.model_results)
                persist_curve_artifacts(self.store, clf.model_results)
                pred_summary = aggregate_predictions_summary(clf.model_results)
                if pred_summary:
                    self.store.write_json("predictions_summary.json", pred_summary)

            if (
                "cross_analysis" in self.config.stages
                and clustered_cat is not None
                and "Cluster" in clustered_cat.columns
                and not resume_classification
            ):
                self.store.write_json(
                    "metrics/cross_analysis.json", {"status": "tables persisted"}
                )

            if (
                "statistics" in self.config.stages
                and clustered_cat is not None
                and "Cluster" in clustered_cat.columns
            ):
                from pns_health_research.statistics.inference import run_statistical_validation

                stats_metrics = run_statistical_validation(
                    clustered_cat, seed=self.config.experiment.seed
                )
                self.store.write_json("metrics/statistics.json", stats_metrics)

            self.performance["total_seconds"] = sum(
                v for k, v in self.performance.items() if k.endswith("_seconds")
            )
            if self.performance:
                self.store.write_json("metrics/performance.json", self.performance)

            report = build_reproduction_report(
                self.config,
                preprocessing_metrics,
                clustering_metrics,
                classification_metrics,
            )
            if self.source_run_id:
                report["source_run_id"] = self.source_run_id
            self.store.write_json("reproduction_report.json", report)

            duration = self.performance.get("total_seconds")
            summary = build_summary(
                self.run_id,
                self.config,
                "completed",
                preprocessing_metrics=preprocessing_metrics or None,
                clustering_metrics=clustering_metrics or None,
                classification_metrics=classification_metrics or None,
                duration_seconds=duration,
                models=list(classification_metrics.keys()) if classification_metrics else None,
            )
            self.store.write_json("summary.json", summary)

            run_type = infer_run_type(self.config)
            missing = validate_required_artifacts(self.run_dir, run_type)
            if missing:
                self.logger.warning("artifacts obrigatórios ausentes: %s", missing)

            finished = datetime.now(UTC).isoformat()
            manifest = build_manifest_v2(
                self.run_id,
                self.config,
                "completed",
                self.started_at,
                finished,
                self._data_fp,
                lineage=self._lineage,
                duration_seconds=duration,
            )
            if summary.get("primary_metric"):
                manifest["primary_metric"] = summary["primary_metric"]
            manifest["artifacts_checklist"] = {"missing": missing, "complete": not missing}
            manifest["artifacts_status"] = compute_artifacts_status(self.run_dir)
            self.store.write_json("manifest.json", manifest)
            self.logger.info("run completed run_id=%s status=completed", self.run_id)
            return self.run_dir

        except Exception as exc:
            self.logger.exception("run failed run_id=%s", self.run_id)
            error = {
                "stage": getattr(exc, "stage", "unknown"),
                "message": str(exc),
                "type": type(exc).__name__,
            }
            self._set_status("failed", error=error)
            raise


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
