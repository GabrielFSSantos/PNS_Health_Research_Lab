"""Persistência unificada de bases nomeadas em data/bases/."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

import pandas as pd

from pns_health_research.config.models import ExperimentConfig
from pns_health_research.core.paths import BASES_DIR
from pns_health_research.data.fingerprint import fingerprint_dataframe, save_fingerprint
from pns_health_research.persistence.preprocess_manifest import (
    audit_resolved_for_profile,
    methodological_warnings,
    preprocessing_config_path,
)
from pns_health_research.modules.preprocessing.contract import ColumnSpec
from pns_health_research.modules.preprocessing.pipelines import PreprocessingResult
from pns_health_research.modules.preprocessing.targets import build_target_definitions
from pns_health_research.modules.preprocessing.viz_artifacts import build_preprocess_funnel

ModuleName = Literal["preprocessing", "clustering", "classification"]


@dataclass(frozen=True)
class ProcessedBundleRef:
    bundle_id: str
    path: Path
    manifest: dict[str, Any]
    binary_path: Path
    categorical_path: Path


@dataclass(frozen=True)
class ClusteringBaseRef:
    base_name: str
    path: Path
    manifest: dict[str, Any]
    binary_path: Path
    categorical_path: Path


def base_path(name: str) -> Path:
    return BASES_DIR / name.strip()


def preprocessing_path(name: str) -> Path:
    return base_path(name) / "preprocessing"


def clustering_path(name: str) -> Path:
    return base_path(name) / "clustering"


def classification_path(name: str) -> Path:
    return base_path(name) / "classification"


def _registry_path(name: str) -> Path:
    return base_path(name) / "registry.json"


def _base_manifest_path(name: str) -> Path:
    return base_path(name) / "base_manifest.json"


def load_registry(name: str) -> dict[str, Any]:
    path = _registry_path(name)
    if not path.exists():
        return {
            "schema_version": 1,
            "base_name": name,
            "created_at": None,
            "modules": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _save_registry(name: str, registry: dict[str, Any]) -> None:
    base_path(name).mkdir(parents=True, exist_ok=True)
    _registry_path(name).write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def update_registry(
    name: str,
    module: ModuleName,
    *,
    entry_config: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    registry = load_registry(name)
    if registry.get("created_at") is None:
        registry["created_at"] = now
    registry["base_name"] = name
    registry.setdefault("modules", {})
    mod = dict(registry["modules"].get(module) or {})
    mod["last_run_at"] = now
    mod["run_count"] = int(mod.get("run_count", 0)) + 1
    if entry_config:
        mod["entry_config"] = entry_config
    if extra:
        mod.update(extra)
    registry["modules"][module] = mod
    _save_registry(name, registry)
    return registry


def _refresh_base_manifest(name: str) -> None:
    root = base_path(name)
    modules_present: list[str] = []
    if (root / "preprocessing" / "manifest.json").exists():
        modules_present.append("preprocessing")
    if (root / "clustering" / "manifest.json").exists():
        modules_present.append("clustering")
    if (root / "classification" / "manifest.json").exists():
        modules_present.append("classification")
    registry = load_registry(name)
    manifest = {
        "schema_version": 1,
        "base_name": name,
        "created_at": registry.get("created_at"),
        "modules_present": modules_present,
        "registry_path": "registry.json",
    }
    _base_manifest_path(name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _resolve_preprocessing_root(name: str) -> Path:
    mod_path = preprocessing_path(name)
    if (mod_path / "manifest.json").exists():
        return mod_path
    raise FileNotFoundError(
        f"Pré-processamento da base '{name}' não encontrado em {mod_path}"
    )


def resolve_preprocessing_source(name: str) -> ProcessedBundleRef:
    root = _resolve_preprocessing_root(name)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}
    binary_name = outputs.get("clustering", "binary.parquet")
    cat_name = outputs.get("classification", "categorical.parquet")
    binary_path = root / binary_name
    categorical_path = root / cat_name
    if not binary_path.exists() or not categorical_path.exists():
        raise FileNotFoundError(
            f"Parquets ausentes em {root} (esperado {binary_name} e {cat_name})"
        )
    return ProcessedBundleRef(
        bundle_id=name,
        path=root,
        manifest=manifest,
        binary_path=binary_path,
        categorical_path=categorical_path,
    )


def resolve_clustering_source(name: str) -> ClusteringBaseRef:
    root = clustering_path(name)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Agrupamento da base '{name}' não encontrado em {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binary_path = root / "clustered_binary.parquet"
    categorical_path = root / "clustered_categorical.parquet"
    if not binary_path.exists() or not categorical_path.exists():
        raise FileNotFoundError(f"Parquets de clustering ausentes em {root}")
    return ClusteringBaseRef(
        base_name=name,
        path=root,
        manifest=manifest,
        binary_path=binary_path,
        categorical_path=categorical_path,
    )


def write_preprocessing_base(
    result: PreprocessingResult,
    config: ExperimentConfig,
    *,
    base_name: str,
    entry_config_path: str | Path | None = None,
    profile_kind: str | None = None,
) -> Path:
    """Grava módulo preprocessing em data/bases/<name>/preprocessing/ (sobrescreve)."""
    out = preprocessing_path(base_name)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_dir = out / "metrics"
    schema_dir = out / "schema"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    schema_dir.mkdir(parents=True, exist_ok=True)

    result.binary_df.to_parquet(out / "binary.parquet")
    result.categorical_df.to_parquet(out / "categorical.parquet")

    preprocessing_metrics = dict(result.metrics)
    target_defs = build_target_definitions(config.preprocessing.target)
    preprocessing_metrics["target_prevalence"] = preprocessing_metrics.get(
        "target_prevalence", {}
    )

    (metrics_dir / "preprocessing.json").write_text(
        json.dumps(preprocessing_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    funnel = build_preprocess_funnel(
        result.filter_stats,
        result.metrics,
        processing_stats=result.processing_stats,
        profile_kind=profile_kind,
        methodological_profile=config.experiment.profile,
    )
    (metrics_dir / "preprocess_funnel.json").write_text(
        json.dumps(funnel, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (metrics_dir / "target_definitions.json").write_text(
        json.dumps(target_defs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    missing_audit = result.processing_stats.get("missing_audit")
    if missing_audit:
        (metrics_dir / "missing_audit.json").write_text(
            json.dumps(missing_audit, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    survey_snapshot = result.survey_snapshot or result.processing_stats.get("survey_snapshot")
    if survey_snapshot:
        (metrics_dir / "survey_snapshot.json").write_text(
            json.dumps(survey_snapshot, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    from pns_health_research.modules.preprocessing.viz_artifacts import build_dataset_distributions

    distributions = build_dataset_distributions(result.binary_df)
    if distributions:
        (metrics_dir / "dataset_distributions.json").write_text(
            json.dumps(distributions, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    fp_binary = fingerprint_dataframe(result.binary_df, f"{base_name}/binary.parquet")
    fp_cat = fingerprint_dataframe(result.categorical_df, f"{base_name}/categorical.parquet")
    fingerprint = {"binary": fp_binary, "categorical": fp_cat}
    save_fingerprint(out / "fingerprint.json", fingerprint)

    resolved = config.model_dump(mode="json")
    (out / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    pp_resolved = config.preprocessing.model_dump(mode="json")
    (out / "preprocessing.resolved.yaml").write_text(
        yaml.safe_dump(pp_resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    column_catalog = result.column_catalog or []
    columns_payload = {
        "schema_version": 1,
        "columns": [c.to_dict() if isinstance(c, ColumnSpec) else c for c in column_catalog],
    }
    (schema_dir / "columns.json").write_text(
        json.dumps(columns_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    created_at = datetime.now(UTC).isoformat()
    pp_config_path = preprocessing_config_path(config)
    entry_path = str(entry_config_path) if entry_config_path else None

    provenance: dict[str, Any] = {
        "raw_dataset": config.data.dataset,
        "raw_year": config.data.year,
        "raw_source": config.data.source,
        "pipeline_executed": result.processing_stats.get("pipeline_executed", []),
        "created_at": created_at,
        "base_name": base_name,
        "entry_config_path": entry_path,
        "preprocessing_config_path": pp_config_path,
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "base_name": base_name,
        "module": "preprocessing",
        "profile_kind": profile_kind or "custom",
        "dataset_id": base_name,
        "bundle_id": base_name,
        "profile": config.experiment.profile,
        "experiment_name": config.experiment.name,
        "created_at": created_at,
        "entry_config_path": entry_path,
        "preprocessing_config_path": pp_config_path,
        "preprocessing_schema_version": config.preprocessing.schema_version,
        "methodological_profile": config.experiment.profile,
        "pipeline_executed": result.processing_stats.get("pipeline_executed", []),
        "audit_resolved": audit_resolved_for_profile(config),
        "methodological_warnings": methodological_warnings(config),
        "audit_documentation": "docs/trajetoria.md#parte-5",
        "rows_out": result.metrics.get("rows_out"),
        "binary_columns": result.metrics.get("binary_columns"),
        "categorical_columns": result.metrics.get("categorical_columns"),
        "target_prevalence": result.metrics.get("target_prevalence", {}),
        "fingerprint": fingerprint,
        "outputs": {
            "clustering": "binary.parquet",
            "classification": "categorical.parquet",
        },
        "downstream": {
            "source_processed_field": "data.source_processed",
            "preprocessing_config": "preprocessing.resolved.yaml",
            "column_schema": "schema/columns.json",
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    update_registry(
        base_name,
        "preprocessing",
        entry_config=entry_path,
        extra={
            "rows_out": result.metrics.get("rows_out"),
            "profile_kind": profile_kind or "custom",
            "methodological_profile": config.experiment.profile,
        },
    )
    _refresh_base_manifest(base_name)
    return out


def list_bases() -> list[dict[str, Any]]:
    """Lista bases em data/bases/ com resumo do registry."""
    if not BASES_DIR.exists():
        return []
    rows: list[dict[str, Any]] = []
    for child in sorted(BASES_DIR.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        registry = load_registry(name)
        modules = registry.get("modules") or {}
        rows.append(
            {
                "base_name": name,
                "created_at": registry.get("created_at"),
                "modules": modules,
            }
        )
    return rows


def write_clustering_base(
    *,
    base_name: str,
    result: Any,
    config: ExperimentConfig,
    source_base: str,
    entry_config_path: str | Path | None = None,
    profile_kind: str | None = None,
) -> Path:
    """Grava módulo clustering em data/bases/<name>/clustering/ (sobrescreve)."""
    from pns_health_research.modules.clustering.results import ClusteringStageResult

    assert isinstance(result, ClusteringStageResult)
    out = clustering_path(base_name)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_dir = out / "metrics"
    tables_dir = out / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    result.binary_with_cluster.to_parquet(out / "clustered_binary.parquet")
    result.categorical_with_cluster.to_parquet(out / "clustered_categorical.parquet")

    (metrics_dir / "clustering.json").write_text(
        json.dumps(result.metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    from pns_health_research.modules.clustering.viz_artifacts import persist_clustering_viz

    persist_clustering_viz(result, out, seed=config.experiment.seed)

    fp_binary = fingerprint_dataframe(
        result.binary_with_cluster, f"{base_name}/clustered_binary.parquet"
    )
    fp_cat = fingerprint_dataframe(
        result.categorical_with_cluster, f"{base_name}/clustered_categorical.parquet"
    )
    fingerprint = {"clustered_binary": fp_binary, "clustered_categorical": fp_cat}
    save_fingerprint(out / "fingerprint.json", fingerprint)

    resolved = config.model_dump(mode="json")
    (out / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    cc = config.clustering
    cl_resolved = cc.model_dump(mode="json")
    (out / "clustering.resolved.yaml").write_text(
        yaml.safe_dump(cl_resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    created_at = datetime.now(UTC).isoformat()
    entry_path = str(entry_config_path) if entry_config_path else None

    preprocessing_manifest: dict[str, Any] | None = None
    pp_manifest_path = preprocessing_path(source_base) / "manifest.json"
    if pp_manifest_path.exists():
        preprocessing_manifest = json.loads(pp_manifest_path.read_text(encoding="utf-8"))

    provenance: dict[str, Any] = {
        "source_base": source_base,
        "preprocessing_manifest_path": str(pp_manifest_path) if pp_manifest_path.exists() else None,
        "preprocessing_profile_kind": (
            preprocessing_manifest.get("profile_kind") if preprocessing_manifest else None
        ),
        "pipeline_executed": result.metrics.get("pipeline_executed", []),
        "created_at": created_at,
        "base_name": base_name,
        "entry_config_path": entry_path,
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "base_name": base_name,
        "module": "clustering",
        "profile_kind": profile_kind or "custom",
        "source_base": source_base,
        "profile": config.experiment.profile,
        "experiment_name": config.experiment.name,
        "methodological_profile": config.experiment.profile,
        "created_at": created_at,
        "entry_config_path": entry_path,
        "clustering_schema_version": cc.schema_version,
        "pipeline_executed": result.metrics.get("pipeline_executed", []),
        "algorithm": result.metrics.get("algorithm"),
        "n_clusters": result.metrics.get("n_clusters"),
        "silhouette": result.metrics.get("silhouette"),
        "wcss": result.metrics.get("wcss"),
        "rows_out": result.metrics.get("rows_out") or len(result.binary_with_cluster),
        "outputs": {
            "clustered_binary": "clustered_binary.parquet",
            "clustered_categorical": "clustered_categorical.parquet",
        },
        "downstream": {
            "metrics": "metrics/clustering.json",
            "clustering_config": "clustering.resolved.yaml",
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    update_registry(
        base_name,
        "clustering",
        entry_config=entry_path,
        extra={
            "source_base": source_base,
            "estimator": result.metrics.get("algorithm"),
            "n_clusters": result.metrics.get("n_clusters"),
            "profile_kind": profile_kind or "custom",
            "methodological_profile": config.experiment.profile,
            "silhouette": result.metrics.get("silhouette"),
            "wcss": result.metrics.get("wcss"),
        },
    )
    _refresh_base_manifest(base_name)
    return out


def _best_classification_model(
    metrics: dict[str, Any],
    primary_metric: str,
) -> tuple[str | None, float | None]:
    best_name: str | None = None
    best_score: float | None = None
    for name, row in metrics.items():
        if not isinstance(row, dict) or row.get("status") == "failed":
            continue
        score = row.get(primary_metric)
        if score is None:
            continue
        try:
            val = float(score)
        except (TypeError, ValueError):
            continue
        if best_score is None or val > best_score:
            best_score = val
            best_name = name
    return best_name, best_score


def write_classification_base(
    result: Any,
    config: ExperimentConfig,
    *,
    base_name: str,
    source_base: str,
    entry_config_path: str | Path | None = None,
    profile_kind: str | None = None,
) -> Path:
    """Grava módulo classification em data/bases/<name>/classification/ (sobrescreve)."""
    from pns_health_research.modules.classification.curve_artifacts import (
        aggregate_predictions_summary,
        persist_curve_artifacts,
    )
    from pns_health_research.modules.classification.persistence import (
        persist_classification_aggregates,
        persist_model_artifacts,
    )
    from pns_health_research.modules.classification.pipelines import ClassificationStageResult
    from pns_health_research.modules.classification.validation import ModelResult
    from pns_health_research.persistence.artifact_store import ArtifactStore

    assert isinstance(result, ClassificationStageResult)
    out = classification_path(base_name)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(out)

    for name, res in result.model_results.items():
        if isinstance(res, ModelResult):
            store.write_parquet(f"predictions/{name}.parquet", res.predictions)
            persist_model_artifacts(store, res)

    persist_classification_aggregates(store, result.model_results)
    persist_curve_artifacts(store, result.model_results)
    pred_summary = aggregate_predictions_summary(result.model_results)
    if pred_summary:
        store.write_json("predictions_summary.json", pred_summary)

    store.write_json("metrics/classification.json", result.metrics)

    clf_resolved = config.classification.model_dump(mode="json")
    (out / "classification.resolved.yaml").write_text(
        yaml.safe_dump(clf_resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    resolved = config.model_dump(mode="json")
    (out / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    pred_frames = [
        res.predictions
        for res in result.model_results.values()
        if isinstance(res, ModelResult) and not res.predictions.empty
    ]
    fingerprint: dict[str, Any] = {}
    if pred_frames:
        combined = pd.concat(pred_frames, ignore_index=True)
        fingerprint["predictions"] = fingerprint_dataframe(combined, f"{base_name}/predictions")
    save_fingerprint(out / "fingerprint.json", fingerprint)

    created_at = datetime.now(UTC).isoformat()
    entry_path = str(entry_config_path) if entry_config_path else None
    clustering_manifest: dict[str, Any] | None = None
    cl_manifest_path = clustering_path(source_base) / "manifest.json"
    if cl_manifest_path.exists():
        clustering_manifest = json.loads(cl_manifest_path.read_text(encoding="utf-8"))

    provenance: dict[str, Any] = {
        "source_base": source_base,
        "clustering_manifest_path": str(cl_manifest_path) if cl_manifest_path.exists() else None,
        "clustering_profile_kind": (
            clustering_manifest.get("profile_kind") if clustering_manifest else None
        ),
        "models": list(result.metrics.keys()),
        "created_at": created_at,
        "base_name": base_name,
        "entry_config_path": entry_path,
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    primary_metric = config.evaluation.primary_metric
    best_model, best_score = _best_classification_model(result.metrics, primary_metric)
    models_run = [
        name
        for name, row in result.model_results.items()
        if not (isinstance(row, dict) and row.get("status") == "failed")
    ]

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "base_name": base_name,
        "module": "classification",
        "profile_kind": profile_kind or "custom",
        "source_base": source_base,
        "profile": config.experiment.profile,
        "experiment_name": config.experiment.name,
        "methodological_profile": config.experiment.profile,
        "created_at": created_at,
        "entry_config_path": entry_path,
        "models": models_run,
        "primary_metric": primary_metric,
        "best_model": best_model,
        "best_score": best_score,
        "metrics": result.metrics,
        "outputs": {
            "classification_metrics": "metrics/classification.json",
            "classification_config": "classification.resolved.yaml",
        },
        "downstream": {
            "predictions": "predictions/",
            "models": "models/",
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    update_registry(
        base_name,
        "classification",
        entry_config=entry_path,
        extra={
            "source_base": source_base,
            "profile_kind": profile_kind or "custom",
            "methodological_profile": config.experiment.profile,
            "models": models_run,
            "primary_metric": primary_metric,
            "best_model": best_model,
            "best_score": best_score,
        },
    )
    _refresh_base_manifest(base_name)
    return out
