"""Exportação compacta de runs para showcase versionado."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pns_health_research.core.paths import ARTIFACTS_DIR, BASES_DIR, RUNS_DIR
from pns_health_research.persistence.base_store import (
    classification_path,
    clustering_path,
    preprocessing_path,
)
from pns_health_research.persistence.run_contract import (
    SCHEMA_VERSION,
    build_lineage_from_dir,
    compute_artifacts_status,
    migrate_manifest_v1,
)

SHOWCASE_DIR = ARTIFACTS_DIR / "showcase"

# Artifacts pequenos suficientes para versionar no Git
SHOWCASE_PATTERNS = [
    "manifest.json",
    "summary.json",
    "config.resolved.yaml",
    "data_fingerprint.json",
    "reproduction_report.json",
    "metrics/*.json",
    "models/*/search.json",
    "models/*/best_params.json",
    "models/*/best_score.json",
    "models/*/final_params.json",
    "tables/elbow_results.parquet",
    "tables/pca_explained_variance.parquet",
    "tables/pca_loadings.parquet",
    "tables/pca_scatter_sample.parquet",
    "tables/cluster_profiles.parquet",
    "tables/cross_analysis.parquet",
    "tables/bubble_health_service.parquet",
    "tables/clustered_binary.parquet",
    "tables/grid_search/*.parquet",
    "tables/grid_search_folds/*.parquet",
    "tables/feature_importance/*.parquet",
    "tables/permutation_importance/*.parquet",
    "tables/centroids.parquet",
    "tables/silhouette_samples.parquet",
    "metrics/silhouette_summary.json",
    "metrics/roc_curves/*.json",
    "predictions_summary.json",
]


def export_showcase(
    run_id: str | None = None,
    *,
    base_name: str | None = None,
    label: str,
    name: str | None = None,
    description: str | None = None,
) -> Path:
    if base_name:
        return export_showcase_from_base(
            base_name, label=label, name=name, description=description
        )
    if not run_id:
        raise ValueError("Informe run_id ou base_name")
    src = RUNS_DIR / run_id
    if not src.exists():
        raise FileNotFoundError(
            f"Run não encontrada: {run_id}. Use base_name= para exportar de data/bases/."
        )

    dest = SHOWCASE_DIR / label
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for pattern in SHOWCASE_PATTERNS:
        for path in src.glob(pattern):
            rel = path.relative_to(src)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    # Migrar summary/manifest v2 se ausente
    _backfill_showcase_contract(dest, name=name or label)

    meta = {
        "label": label,
        "source_run_id": run_id,
        "name": name or label,
        "description": description or "",
    }
    (dest / "showcase_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    readme = dest / "README.md"
    readme.write_text(
        f"# Showcase: {meta['name']}\n\n"
        f"Export compacto da run `{run_id}`.\n\n"
        f"{meta['description']}\n\n"
        "Predictions completas e modelos joblib não são versionados.\n",
        encoding="utf-8",
    )
    return dest


def export_showcase_from_base(
    base_name: str,
    *,
    label: str,
    name: str | None = None,
    description: str | None = None,
) -> Path:
    """Exporta base modular data/bases/<nome>/ para layout showcase."""
    src_root = BASES_DIR / base_name
    if not src_root.is_dir():
        raise FileNotFoundError(f"Base modular não encontrada: {src_root}")

    dest = SHOWCASE_DIR / label
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    pp = preprocessing_path(base_name)
    cl = clustering_path(base_name)
    cf = classification_path(base_name)

    def _copy(src: Path, rel: str) -> None:
        if not src.exists():
            return
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)

    _copy(pp / "metrics" / "preprocessing.json", "metrics/preprocessing.json")
    for metric_name in (
        "preprocess_funnel",
        "dataset_distributions",
        "survey_snapshot",
    ):
        _copy(pp / "metrics" / f"{metric_name}.json", f"metrics/{metric_name}.json")
    _write_showcase_fingerprint(pp / "fingerprint.json", dest / "data_fingerprint.json")
    _copy(pp / "config.resolved.yaml", "config.resolved.yaml")
    _copy(cl / "metrics" / "clustering.json", "metrics/clustering.json")
    _copy(cl / "metrics" / "silhouette_summary.json", "metrics/silhouette_summary.json")
    _copy(cf / "metrics" / "classification.json", "metrics/classification.json")
    _copy(cf / "predictions_summary.json", "predictions_summary.json")
    if (cl / "tables").exists():
        for path in (cl / "tables").glob("*.parquet"):
            _copy(path, f"tables/{path.name}")
    if (cf / "models").exists():
        for path in (cf / "models").rglob("*.json"):
            _copy(path, str(path.relative_to(cf)))
    manifest_src = cf / "manifest.json"
    if manifest_src.exists():
        _copy(manifest_src, "manifest.json")

    _backfill_showcase_contract(dest, name=name or label)
    meta = {
        "label": label,
        "source_run_id": None,
        "source_base": base_name,
        "name": name or label,
        "description": description or f"Export da base modular {base_name}",
    }
    (dest / "showcase_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (dest / "README.md").write_text(
        f"# Showcase: {meta['name']}\n\n"
        f"Export da base modular `{base_name}`.\n\n"
        f"{meta['description']}\n",
        encoding="utf-8",
    )
    return dest


def _write_showcase_fingerprint(src: Path, dest: Path) -> None:
    """Grava fingerprint no formato plano consumido pelo dashboard (showcase)."""
    if not src.exists():
        return
    from pns_health_research.persistence.base_query import normalize_fingerprint

    data = json.loads(src.read_text(encoding="utf-8"))
    flat = normalize_fingerprint(data)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "n_rows": flat.get("rows"),
                "n_columns": flat.get("columns"),
                "hash": flat.get("hash"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _backfill_showcase_contract(dest: Path, *, name: str) -> None:
    """Gera summary.json e manifest v2 mínimo para runs exportadas sem contrato novo."""
    manifest_path = dest / "manifest.json"
    summary_path = dest / "summary.json"
    if not manifest_path.exists():
        return
    manifest = migrate_manifest_v1(json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["name"] = name
    manifest.setdefault("type", "experiment")
    if not manifest.get("lineage"):
        manifest["lineage"] = build_lineage_from_dir(dest)
    manifest["artifacts_status"] = compute_artifacts_status(dest)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if summary_path.exists():
        return

    prep = dest / "metrics" / "preprocessing.json"
    clust = dest / "metrics" / "clustering.json"
    clf = dest / "metrics" / "classification.json"
    preprocessing = json.loads(prep.read_text()) if prep.exists() else {}
    clustering = json.loads(clust.read_text()) if clust.exists() else {}
    classification = json.loads(clf.read_text()) if clf.exists() else {}

    primary_name = "f1_positive"
    best_val = None
    best_model = None
    for model, metrics in classification.items():
        if isinstance(metrics, dict) and primary_name in metrics:
            v = float(metrics[primary_name])
            if best_val is None or v > best_val:
                best_val = v
                best_model = model

    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest.get("run_id"),
        "name": name,
        "type": manifest.get("type", "experiment"),
        "status": manifest.get("status", "completed"),
        "profile": manifest.get("profile"),
        "tags": manifest.get("tags", []),
        "preserve": True,
        "dataset": {
            "source": "pns",
            "year": preprocessing.get("data_year", 2019),
            "rows": preprocessing.get("rows_out"),
            "features_binary": preprocessing.get("binary_columns"),
            "features_categorical": preprocessing.get("categorical_columns"),
        },
        "clustering": {
            "algorithm": clustering.get("algorithm"),
            "n_clusters": clustering.get("n_clusters"),
            "silhouette": clustering.get("silhouette"),
            "wcss": clustering.get("wcss"),
        }
        if clustering
        else None,
        "models": list(classification.keys()) if classification else [],
        "primary_metric": {
            "name": primary_name,
            "value": best_val,
            "model": best_model,
        },
        "duration_seconds": None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def list_showcase_runs() -> list[dict]:
    rows: list[dict] = []
    if not SHOWCASE_DIR.exists():
        return rows
    for d in sorted(SHOWCASE_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "showcase_meta.json"
        manifest_path = d / "manifest.json"
        summary_path = d / "summary.json"
        row: dict = {"label": d.name, "path": str(d.name), "source": "showcase"}
        if meta_path.exists():
            row.update(json.loads(meta_path.read_text(encoding="utf-8")))
        if manifest_path.exists():
            row["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if summary_path.exists():
            row["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(row)
    return rows
