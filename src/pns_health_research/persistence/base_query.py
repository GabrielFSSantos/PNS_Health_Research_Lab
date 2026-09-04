"""Resolução de artefatos modulares em data/bases/ para dashboard e API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pns_health_research.core.paths import BASES_DIR
from pns_health_research.persistence.base_store import (
    classification_path,
    clustering_path,
    load_registry,
    preprocessing_path,
)


def is_base_name(name: str) -> bool:
    return (BASES_DIR / name / "registry.json").exists()


def list_complete_bases() -> list[str]:
    """Bases com os três módulos materializados."""
    names: list[str] = []
    if not BASES_DIR.exists():
        return names
    for child in sorted(BASES_DIR.iterdir()):
        if not child.is_dir():
            continue
        registry = load_registry(child.name)
        modules = set((registry.get("modules") or {}).keys())
        if {"preprocessing", "clustering", "classification"}.issubset(modules):
            names.append(child.name)
    return names


def build_base_summary(base_name: str) -> dict[str, Any]:
    registry = load_registry(base_name)
    clf_manifest_path = classification_path(base_name) / "manifest.json"
    primary_metric: dict[str, Any] | None = None
    if clf_manifest_path.exists():
        clf_manifest = json.loads(clf_manifest_path.read_text(encoding="utf-8"))
        primary_metric = {
            "name": clf_manifest.get("primary_metric", "f1_positive"),
            "value": clf_manifest.get("best_score"),
            "model": clf_manifest.get("best_model"),
        }
    modules = registry.get("modules") or {}
    pp = modules.get("preprocessing") or {}
    cl = modules.get("clustering") or {}
    return {
        "run_id": base_name,
        "name": base_name,
        "type": "experiment",
        "status": "completed",
        "tags": [pp.get("profile_kind", "custom")],
        "preserve": True,
        "source": "base",
        "primary_metric": primary_metric,
        "profile_kind": pp.get("profile_kind"),
        "rows_out": pp.get("rows_out"),
        "clustering_k": cl.get("k"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_table_path(base_name: str, table_name: str) -> Path | None:
    """Mapeia nomes de tabela do contrato run → paths modulares."""
    cl_root = clustering_path(base_name)
    aliases = {
        "elbow_results": cl_root / "tables" / "elbow_results.parquet",
        "pca_explained_variance": cl_root / "tables" / "pca_explained_variance.parquet",
        "pca_variance": cl_root / "tables" / "pca_variance.parquet",
        "pca_loadings": cl_root / "tables" / "pca_loadings.parquet",
        "pca_scatter_sample": cl_root / "tables" / "pca_scatter_sample.parquet",
        "cluster_profiles": cl_root / "tables" / "cluster_profiles.parquet",
        "cluster_profiles_table": cl_root / "tables" / "cluster_profiles_table.parquet",
        "cross_analysis": cl_root / "tables" / "cross_analysis.parquet",
        "centroids": cl_root / "tables" / "centroids.parquet",
        "bubble_health_service": cl_root / "tables" / "bubble_health_service.parquet",
        "clustered_binary": cl_root / "tables" / "clustered_binary.parquet",
        "clustered_categorical": cl_root / "tables" / "clustered_categorical.parquet",
        "silhouette_samples": cl_root / "tables" / "silhouette_samples.parquet",
    }
    path = aliases.get(table_name)
    if path and path.exists():
        return path
    fallback = cl_root / f"{table_name}.parquet"
    if fallback.exists():
        return fallback
    return None


def resolve_metric_path(base_name: str, metric_name: str) -> Path | None:
    pp = preprocessing_path(base_name) / "metrics" / f"{metric_name}.json"
    if pp.exists():
        return pp
    cl = clustering_path(base_name) / "metrics" / f"{metric_name}.json"
    if cl.exists():
        return cl
    cf = classification_path(base_name) / "metrics" / f"{metric_name}.json"
    if cf.exists():
        return cf
    return None


def resolve_classification_root(base_name: str) -> Path:
    return classification_path(base_name)


def resolve_fingerprint(base_name: str) -> dict[str, Any] | None:
    fp = preprocessing_path(base_name) / "fingerprint.json"
    if not fp.exists():
        return None
    return normalize_fingerprint(_read_json(fp))


def normalize_fingerprint(data: dict[str, Any]) -> dict[str, Any]:
    """Unifica fingerprint modular (binary.*) e showcase (plano)."""
    binary = data.get("binary") or {}
    rows = (
        data.get("n_rows")
        or data.get("rows")
        or binary.get("n_rows")
        or binary.get("rows")
    )
    columns = (
        data.get("n_columns")
        or data.get("columns")
        or binary.get("n_columns")
        or binary.get("columns")
    )
    categorical = data.get("categorical") or {}
    return {
        "hash": data.get("hash") or binary.get("hash"),
        "rows": rows,
        "columns": columns,
        "n_rows": rows,
        "n_columns": columns,
        "binary_columns": binary.get("n_columns") or binary.get("columns") or columns,
        "categorical_columns": categorical.get("n_columns") or categorical.get("columns"),
    }
