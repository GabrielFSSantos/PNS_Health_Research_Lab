"""Contrato v2 de runs: manifest, summary, lineage e validação de artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pns_health_research.config.models import ExperimentConfig
from pns_health_research.persistence.artifact_store import (
    config_hash,
    git_commit,
)

SCHEMA_VERSION = 2

RunType = Literal["experiment", "clustering", "classification", "preprocessing"]
RunStatus = Literal["created", "running", "completed", "failed", "cancelled"]


def infer_run_type(config: ExperimentConfig) -> RunType:
    if config.run.type:
        return config.run.type  # type: ignore[return-value]
    stages = set(config.stages)
    if stages <= {"classification", "statistics"}:
        return "classification"
    if stages <= {"preprocessing", "clustering"}:
        return "clustering"
    if stages == {"preprocessing"}:
        return "preprocessing"
    return "experiment"


def build_lineage(
    config: ExperimentConfig,
    *,
    preprocessing_run_id: str | None,
    clustering_run_id: str | None,
    clustering_used: bool,
) -> dict[str, Any]:
    return {
        "dataset": {
            "source": config.data.dataset,
            "year": config.data.year,
        },
        "preprocessing_run_id": preprocessing_run_id,
        "clustering": {
            "used": clustering_used,
            "run_id": clustering_run_id,
        },
    }


def build_manifest_v2(
    run_id: str,
    config: ExperimentConfig,
    status: RunStatus,
    started_at: str,
    finished_at: str | None,
    data_fingerprint: str,
    *,
    lineage: dict[str, Any] | None = None,
    duration_seconds: float | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_type = infer_run_type(config)
    display_name = config.run.name or config.experiment.name
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "name": display_name,
        "experiment_name": config.experiment.name,
        "profile": config.experiment.profile,
        "type": run_type,
        "status": status,
        "description": config.run.description,
        "notes": config.run.notes,
        "tags": list(config.run.tags),
        "preserve": config.run.preserve,
        "started_at": started_at,
        "finished_at": finished_at,
        "random_seed": config.experiment.seed,
        "git_commit": git_commit(),
        "config_hash": config_hash(config),
        "data_fingerprint": data_fingerprint,
        "primary_metric": {
            "name": config.evaluation.primary_metric,
        },
        "environment": {
            "python_version": __import__("sys").version,
            "platform": __import__("platform").platform(),
        },
        "duration_seconds": duration_seconds,
        "lineage": lineage or {},
    }
    if error:
        manifest["error"] = error
    return manifest


def build_summary(
    run_id: str,
    config: ExperimentConfig,
    status: RunStatus,
    *,
    preprocessing_metrics: dict[str, Any] | None = None,
    clustering_metrics: dict[str, Any] | None = None,
    classification_metrics: dict[str, Any] | None = None,
    duration_seconds: float | None = None,
    models: list[str] | None = None,
) -> dict[str, Any]:
    display_name = config.run.name or config.experiment.name
    run_type = infer_run_type(config)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "name": display_name,
        "type": run_type,
        "status": status,
        "profile": config.experiment.profile,
        "tags": list(config.run.tags),
        "preserve": config.run.preserve,
        "dataset": {
            "source": config.data.dataset,
            "year": config.data.year,
        },
        "duration_seconds": duration_seconds,
        "primary_metric": {
            "name": config.evaluation.primary_metric,
        },
    }
    if preprocessing_metrics:
        summary["dataset"]["rows"] = preprocessing_metrics.get("rows_out")
        summary["dataset"]["features_binary"] = preprocessing_metrics.get("binary_columns")
        summary["dataset"]["features_categorical"] = preprocessing_metrics.get(
            "categorical_columns"
        )
    if clustering_metrics:
        summary["clustering"] = {
            "algorithm": clustering_metrics.get("algorithm"),
            "n_clusters": clustering_metrics.get("n_clusters"),
            "silhouette": clustering_metrics.get("silhouette"),
            "wcss": clustering_metrics.get("wcss"),
        }
    if classification_metrics:
        summary["models"] = models or list(classification_metrics.keys())
        primary = config.evaluation.primary_metric
        best_val: float | None = None
        best_model: str | None = None
        for model_name, metrics in classification_metrics.items():
            if isinstance(metrics, dict) and metrics.get("status") == "failed":
                continue
            if isinstance(metrics, dict) and primary in metrics:
                val = float(metrics[primary])
                if best_val is None or val > best_val:
                    best_val = val
                    best_model = model_name
        if best_val is not None:
            summary["primary_metric"]["value"] = best_val
            summary["primary_metric"]["model"] = best_model
    return summary


REQUIRED_ARTIFACTS: dict[str, list[str]] = {
    "experiment": [
        "manifest.json",
        "summary.json",
        "metrics/preprocessing.json",
        "metrics/clustering.json",
        "metrics/classification.json",
    ],
    "clustering": [
        "manifest.json",
        "summary.json",
        "metrics/preprocessing.json",
        "metrics/clustering.json",
    ],
    "classification": [
        "manifest.json",
        "summary.json",
        "metrics/classification.json",
    ],
    "preprocessing": [
        "manifest.json",
        "summary.json",
        "metrics/preprocessing.json",
    ],
}


def validate_required_artifacts(run_dir: Path, run_type: RunType) -> list[str]:
    """Retorna lista de artifacts obrigatórios ausentes."""
    missing: list[str] = []
    for rel in REQUIRED_ARTIFACTS.get(run_type, ["manifest.json", "summary.json"]):
        if not (run_dir / rel).exists():
            missing.append(rel)
    return missing


def migrate_manifest_v1(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compatibilidade com manifests antigos."""
    if manifest.get("schema_version", 1) >= SCHEMA_VERSION:
        return manifest
    migrated = dict(manifest)
    migrated["schema_version"] = SCHEMA_VERSION
    migrated.setdefault("name", manifest.get("experiment_name", "unknown"))
    migrated.setdefault("type", "experiment")
    migrated.setdefault("lineage", {})
    if "source_run_id" in manifest:
        migrated["lineage"] = {
            "dataset": {"source": "pns", "year": 2019},
            "preprocessing_run_id": manifest["source_run_id"],
            "clustering": {
                "used": True,
                "run_id": manifest["source_run_id"],
            },
        }
    migrated.setdefault("tags", manifest.get("tags", []))
    migrated.setdefault("preserve", manifest.get("preserve", False))
    return migrated


def build_lineage_from_dir(run_dir: Path) -> dict[str, Any]:
    """Reconstrói lineage a partir de config.resolved.yaml ou manifest."""
    config_path = run_dir / "config.resolved.yaml"
    manifest_path = run_dir / "manifest.json"
    if config_path.exists():
        import yaml

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        data = cfg.get("data", {})
        source_run = cfg.get("source_run_id") or cfg.get("classification", {}).get("input_run")
        stages = set(cfg.get("stages", []))
        clustering_used = "clustering" in stages or bool(source_run)
        run_id = manifest_path.exists() and json.loads(
            manifest_path.read_text(encoding="utf-8")
        ).get("run_id")
        return {
            "dataset": {
                "source": data.get("dataset", "pns"),
                "year": data.get("year", 2019),
            },
            "preprocessing_run_id": source_run or run_id,
            "clustering": {
                "used": clustering_used,
                "run_id": (source_run or run_id) if clustering_used else None,
            },
        }
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        return m.get("lineage", {})
    return {}


def compute_artifacts_status(run_dir: Path) -> dict[str, str]:
    """Status honesto dos artifacts opcionais para UI."""
    checks = {
        "grid_search": bool(list(run_dir.glob("tables/grid_search/*.parquet"))),
        "feature_importance": bool(list(run_dir.glob("tables/feature_importance/*.parquet"))),
        "pca_loadings": (run_dir / "tables/pca_loadings.parquet").exists(),
        "pca_scatter": (run_dir / "tables/pca_scatter_sample.parquet").exists(),
        "silhouette": (run_dir / "metrics/silhouette_summary.json").exists(),
        "preprocess_funnel": (run_dir / "metrics/preprocess_funnel.json").exists(),
        "roc_curves": bool(list((run_dir / "metrics/roc_curves").glob("*.json")))
        if (run_dir / "metrics/roc_curves").exists()
        else False,
        "predictions_summary": (run_dir / "predictions_summary.json").exists(),
    }
    return {k: "present" if v else "missing" for k, v in checks.items()}


def load_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
