"""Serviço de consulta read-only sobre artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pns_health_research.modules.clustering.alignment import align_cluster_labels
from pns_health_research.core.paths import CONFIGS_DIR, RUNS_DIR
from pns_health_research.persistence import base_query
from pns_health_research.persistence.run_contract import migrate_manifest_v1
from pns_health_research.persistence.showcase import SHOWCASE_DIR, list_showcase_runs


class RunNotFoundError(Exception):
    pass


class ArtifactError(Exception):
    pass


def _resolve_run_dir(run_id: str) -> Path:
    if base_query.is_base_name(run_id):
        raise RunNotFoundError(run_id)
    for base in (SHOWCASE_DIR, RUNS_DIR):
        candidate = base / run_id
        if candidate.exists() and candidate.is_dir():
            return candidate
    # showcase labels may differ from run_id
    showcase = SHOWCASE_DIR / run_id
    if showcase.exists():
        return showcase
    run_path = RUNS_DIR / run_id
    if run_path.exists():
        return run_path
    raise RunNotFoundError(run_id)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactError(f"Artifact ausente: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_parquet(path).to_dict(orient="records")


def list_all_runs(
    *,
    run_type: str | None = None,
    status: str | None = None,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for base_name in base_query.list_complete_bases():
        summary = base_query.build_base_summary(base_name)
        rows.append(
            {
                "run_id": base_name,
                "showcase_label": None,
                "name": base_name,
                "type": "experiment",
                "status": "completed",
                "tags": summary.get("tags", []),
                "preserve": True,
                "source": "base",
                "profile_kind": summary.get("profile_kind"),
                "primary_metric": summary.get("primary_metric"),
                "duration_seconds": None,
            }
        )

    showcase_source_ids: set[str] = set()
    for item in list_showcase_runs():
        summary = item.get("summary") or {}
        manifest = item.get("manifest") or {}
        source_id = item.get("source_run_id") or item.get("label")
        if source_id:
            showcase_source_ids.add(source_id)
        row = {
            "run_id": source_id or item.get("label"),
            "showcase_label": item.get("label"),
            "name": summary.get("name") or item.get("name") or item.get("label"),
            "type": summary.get("type") or manifest.get("type", "experiment"),
            "status": summary.get("status") or manifest.get("status", "completed"),
            "tags": summary.get("tags") or manifest.get("tags", []),
            "preserve": summary.get("preserve", True),
            "source": "showcase",
            "primary_metric": summary.get("primary_metric"),
            "duration_seconds": summary.get("duration_seconds"),
        }
        rows.append(row)

    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir(), reverse=True):
            if not d.is_dir() or d.name.startswith(".") or d.name == "_archive":
                continue
            if d.name in showcase_source_ids:
                continue
            if any(r.get("run_id") == d.name for r in rows):
                continue
            summary_path = d / "summary.json"
            manifest_path = d / "manifest.json"
            if summary_path.exists():
                local_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            elif manifest_path.exists():
                local_summary = migrate_manifest_v1(
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                )
            else:
                continue
            rows.append(
                {
                    "run_id": d.name,
                    "showcase_label": None,
                    "name": local_summary.get("name", d.name),
                    "type": local_summary.get("type", "experiment"),
                    "status": local_summary.get("status", "unknown"),
                    "tags": local_summary.get("tags", []),
                    "preserve": local_summary.get("preserve", False),
                    "source": "runs",
                    "primary_metric": local_summary.get("primary_metric"),
                    "duration_seconds": local_summary.get("duration_seconds"),
                }
            )

    if run_type:
        rows = [r for r in rows if r.get("type") == run_type]
    if status:
        rows = [r for r in rows if r.get("status") == status]
    if tags:
        rows = [r for r in rows if tags and all(t in r.get("tags", []) for t in tags)]
    return rows


def get_run_summary(run_id: str) -> dict[str, Any]:
    if base_query.is_base_name(run_id):
        return base_query.build_base_summary(run_id)
    run_dir = _resolve_run_dir(run_id)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        return migrate_manifest_v1(json.loads(manifest_path.read_text(encoding="utf-8")))
    raise ArtifactError("summary.json ausente")


def get_run_detail(run_id: str) -> dict[str, Any]:
    if base_query.is_base_name(run_id):
        summary = base_query.build_base_summary(run_id)
        from pns_health_research.persistence.base_store import load_registry

        registry = load_registry(run_id)
        return {
            "manifest": {
                "run_id": run_id,
                "name": run_id,
                "status": "completed",
                "type": "experiment",
                "modules": registry.get("modules", {}),
            },
            "summary": summary,
            "lineage": {"source_base": run_id},
        }
    run_dir = _resolve_run_dir(run_id)
    manifest = migrate_manifest_v1(_read_json(run_dir / "manifest.json"))
    summary = get_run_summary(run_id)
    return {
        "manifest": manifest,
        "summary": summary,
        "lineage": manifest.get("lineage", {}),
    }


def get_run_metrics(run_id: str, metric: str | None = None) -> dict[str, Any]:
    if base_query.is_base_name(run_id):
        if metric:
            path = base_query.resolve_metric_path(run_id, metric)
            if path:
                return _read_json(path)
            raise ArtifactError(f"Métrica ausente: {metric}")
        out: dict[str, Any] = {}
        for name in ("preprocessing", "clustering", "classification"):
            path = base_query.resolve_metric_path(run_id, name)
            if path:
                out[name] = json.loads(path.read_text(encoding="utf-8"))
        return out
    run_dir = _resolve_run_dir(run_id)
    if metric:
        path = run_dir / "metrics" / f"{metric}.json"
        return _read_json(path)
    out: dict[str, Any] = {}
    metrics_dir = run_dir / "metrics"
    if metrics_dir.exists():
        for p in sorted(metrics_dir.glob("*.json")):
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


def get_grid_search(run_id: str, model: str) -> dict[str, Any]:
    root = (
        base_query.resolve_classification_root(run_id)
        if base_query.is_base_name(run_id)
        else _resolve_run_dir(run_id)
    )
    search_path = root / "models" / model / "search.json"
    if not search_path.exists():
        return {"model": model, "status": "missing"}
    data: dict[str, Any] = {
        "model": model,
        "search": _read_json(search_path),
        "best_params": _read_json(root / "models" / model / "best_params.json")
        if (root / "models" / model / "best_params.json").exists()
        else {},
        "final_params": _read_json(root / "models" / model / "final_params.json")
        if (root / "models" / model / "final_params.json").exists()
        else {},
    }
    score_path = root / "models" / model / "best_score.json"
    if score_path.exists():
        data["best_score"] = _read_json(score_path)
    cv_path = root / "tables" / "grid_search" / f"{model}.parquet"
    data["candidates"] = _read_parquet(cv_path)
    folds_path = root / "tables" / "grid_search_folds" / f"{model}.parquet"
    data["folds"] = _read_parquet(folds_path)
    return data


def get_feature_importance(run_id: str, model: str) -> dict[str, Any]:
    root = (
        base_query.resolve_classification_root(run_id)
        if base_query.is_base_name(run_id)
        else _resolve_run_dir(run_id)
    )
    native = root / "tables" / "feature_importance" / f"{model}.parquet"
    perm = root / "tables" / "permutation_importance" / f"{model}.parquet"
    return {
        "model": model,
        "native": _read_parquet(native),
        "permutation": _read_parquet(perm),
    }


def get_clusters(run_id: str) -> dict[str, Any]:
    metrics = get_run_metrics(run_id, "clustering")
    tables: dict[str, Any] = {}
    for name in (
        "elbow_results",
        "pca_explained_variance",
        "pca_loadings",
        "pca_scatter_sample",
        "cluster_profiles",
        "cross_analysis",
        "centroids",
        "bubble_health_service",
        "clustered_binary",
    ):
        if base_query.is_base_name(run_id):
            path = base_query.resolve_table_path(run_id, name)
        else:
            path = _resolve_run_dir(run_id) / "tables" / f"{name}.parquet"
        if path and path.exists():
            tables[name] = _read_parquet(path)
    if base_query.is_base_name(run_id):
        from pns_health_research.persistence.base_store import clustering_path

        sil_path = clustering_path(run_id) / "metrics" / "silhouette_summary.json"
        sil_data = json.loads(sil_path.read_text()) if sil_path.exists() else {}
        if not sil_data and isinstance(metrics, dict):
            sil_data = metrics.get("silhouette_summary") or {}
        sil_samples_path = base_query.resolve_table_path(run_id, "silhouette_samples")
    else:
        run_dir = _resolve_run_dir(run_id)
        silhouette = run_dir / "metrics" / "silhouette_summary.json"
        sil_data = json.loads(silhouette.read_text()) if silhouette.exists() else {}
        sil_samples_path = run_dir / "tables" / "silhouette_samples.parquet"
    sil_samples = _read_parquet(sil_samples_path) if sil_samples_path and sil_samples_path.exists() else []
    return {
        "metrics": metrics,
        "tables": tables,
        "silhouette_summary": sil_data,
        "silhouette_samples": sil_samples,
    }


def get_dataset_info(run_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if base_query.is_base_name(run_id):
        from pns_health_research.persistence.base_store import preprocessing_path

        pp_root = preprocessing_path(run_id)
        for name in ("preprocessing", "preprocess_funnel", "dataset_distributions", "survey_snapshot"):
            path = pp_root / "metrics" / f"{name}.json"
            if path.exists():
                out[name] = json.loads(path.read_text(encoding="utf-8"))
        fp = base_query.resolve_fingerprint(run_id)
        if fp:
            out["fingerprint"] = fp
        return out
    run_dir = _resolve_run_dir(run_id)
    out: dict[str, Any] = {}
    for name in ("preprocessing", "preprocess_funnel", "dataset_distributions", "survey_snapshot"):
        path = run_dir / "metrics" / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    fp = run_dir / "data_fingerprint.json"
    if fp.exists():
        from pns_health_research.persistence.base_query import normalize_fingerprint

        out["fingerprint"] = normalize_fingerprint(json.loads(fp.read_text(encoding="utf-8")))
    return out


def get_dictionary() -> dict[str, Any]:
    path = CONFIGS_DIR / "data" / "dictionary.yaml"
    if not path.exists():
        return {"columns": []}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {"columns": []}


def get_dictionary_column(internal_name: str) -> dict[str, Any]:
    data = get_dictionary()
    for col in data.get("columns", []):
        if col.get("internal_name") == internal_name:
            return col
    raise ArtifactError(f"Coluna não encontrada: {internal_name}")


def get_predictions_summary(run_id: str) -> dict[str, Any]:
    root = (
        base_query.resolve_classification_root(run_id)
        if base_query.is_base_name(run_id)
        else _resolve_run_dir(run_id)
    )
    path = root / "predictions_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def get_curves(run_id: str, model: str) -> dict[str, Any]:
    root = (
        base_query.resolve_classification_root(run_id)
        if base_query.is_base_name(run_id)
        else _resolve_run_dir(run_id)
    )
    path = root / "metrics" / "roc_curves" / f"{model}.json"
    if not path.exists():
        return {"model": model, "status": "missing"}
    return {"model": model, **json.loads(path.read_text(encoding="utf-8"))}


def _load_centroids(run_dir: Path) -> np.ndarray | None:
    path = run_dir / "tables" / "centroids.parquet"
    if not path.exists() and run_dir.parent.name == "bases":
        alt = base_query.resolve_table_path(run_dir.name, "centroids")
        path = alt or path
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    num_cols = [c for c in df.columns if c.startswith("pc")]
    if not num_cols:
        return None
    return df[num_cols].values


def compare_clustering(runs: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    centroids_map: dict[str, np.ndarray] = {}
    for run_id in runs:
        try:
            metrics = get_run_metrics(run_id, "clustering")
            summary = get_run_summary(run_id)
            rows.append(
                {
                    "run_id": run_id,
                    "name": summary.get("name", run_id),
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
                }
            )
            from pns_health_research.persistence.base_store import clustering_path

            run_dir = (
                clustering_path(run_id)
                if base_query.is_base_name(run_id)
                else _resolve_run_dir(run_id)
            )
            cent = _load_centroids(run_dir)
            if cent is not None:
                centroids_map[run_id] = cent
        except (RunNotFoundError, ArtifactError):
            continue

    alignment: dict[str, Any] = {"aligned": False, "mapping": {}}
    alignment_warning = None
    if len(centroids_map) >= 2:
        ref_id = runs[0]
        if ref_id in centroids_map:
            alignment["aligned"] = True
            for rid, cent in centroids_map.items():
                if rid == ref_id:
                    alignment["mapping"][rid] = list(range(cent.shape[0]))
                elif cent.shape == centroids_map[ref_id].shape:
                    dummy_labels = np.arange(cent.shape[0])
                    aligned = align_cluster_labels(
                        centroids_map[ref_id], dummy_labels, cent
                    )
                    alignment["mapping"][rid] = [int(x) for x in aligned]
                else:
                    alignment_warning = (
                        "Centroids incompatíveis entre runs; labels não equivalentes"
                    )
                    alignment["aligned"] = False
    elif len(runs) >= 2:
        alignment_warning = "Centroids ausentes; comparação apenas por índice de cluster"

    return {
        "runs": runs,
        "comparison": rows,
        "alignment": alignment,
        "alignment_warning": alignment_warning,
    }


def compare_classification(runs: list[str]) -> dict[str, Any]:
    rows = []
    for run_id in runs:
        try:
            metrics = get_run_metrics(run_id, "classification")
            summary = get_run_summary(run_id)
            for model, model_metrics in metrics.items():
                if isinstance(model_metrics, dict) and model_metrics.get("status") != "failed":
                    rows.append(
                        {
                            "run_id": run_id,
                            "name": summary.get("name", run_id),
                            "model": model,
                            **{
                                k: v
                                for k, v in model_metrics.items()
                                if isinstance(v, (int, float))
                            },
                        }
                    )
        except (RunNotFoundError, ArtifactError):
            continue
    return {"runs": runs, "comparison": rows}


def dashboard_stats() -> dict[str, Any]:
    runs = list_all_runs()
    clustering = [r for r in runs if r.get("type") in ("experiment", "clustering")]
    classification = [r for r in runs if r.get("type") in ("experiment", "classification")]
    preserved = [r for r in runs if r.get("preserve") or r.get("source") == "showcase"]
    best_clf = None
    best_val = -1.0
    for r in classification:
        pm = r.get("primary_metric") or {}
        val = pm.get("value")
        if val is not None and val > best_val:
            best_val = val
            best_clf = r
    return {
        "pns_year": 2019,
        "total_runs": len(runs),
        "clustering_runs": len(clustering),
        "classification_runs": len(classification),
        "preserved_runs": len(preserved),
        "best_classification": best_clf,
        "recent_runs": runs[:10],
        "showcase_available": SHOWCASE_DIR.exists() and any(SHOWCASE_DIR.iterdir())
        if SHOWCASE_DIR.exists()
        else False,
    }
