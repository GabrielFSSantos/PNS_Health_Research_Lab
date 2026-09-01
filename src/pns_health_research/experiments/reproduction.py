from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pns_health_research.config.models import ExperimentConfig


def load_tolerances(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compare_metric(
    name: str,
    expected: float,
    actual: float,
    abs_tol: float,
) -> dict[str, Any]:
    diff = actual - expected
    status = "PASS" if abs(diff) <= abs_tol else "FAIL"
    return {
        "metric": name,
        "expected": expected,
        "actual": actual,
        "absolute_difference": diff,
        "tolerance": abs_tol,
        "status": status,
    }


def build_reproduction_report(
    config: ExperimentConfig,
    preprocessing_metrics: dict,
    clustering_metrics: dict,
    classification_metrics: dict,
) -> dict[str, Any]:
    if not config.baseline.enabled:
        return {"status": "skipped", "reason": "baseline disabled"}

    tol_path = Path(config.baseline.tolerances_path)
    if not tol_path.is_absolute():
        from pns_health_research.core.paths import PROJECT_ROOT

        tol_path = PROJECT_ROOT / tol_path
    tolerances = load_tolerances(tol_path) if tol_path.exists() else {}

    checks: list[dict[str, Any]] = []

    pre_tol = tolerances.get("preprocessing", {})
    if "rows_out" in preprocessing_metrics:
        exp_rows = pre_tol.get("rows_out", 87678)
        checks.append(
            {
                "metric": "preprocessed_rows",
                "expected": exp_rows,
                "actual": preprocessing_metrics["rows_out"],
                "status": "PASS" if preprocessing_metrics["rows_out"] == exp_rows else "FAIL",
            }
        )

    clust_tol = tolerances.get("clustering", {})
    if "silhouette" in clustering_metrics:
        checks.append(
            compare_metric(
                "silhouette",
                clust_tol.get("silhouette", 0.513881101838751),
                clustering_metrics["silhouette"],
                clust_tol.get("silhouette_absolute_tolerance", 0.005),
            )
        )

    class_tol = tolerances.get("classification", {})
    for model, metrics in classification_metrics.items():
        if isinstance(metrics, dict) and metrics.get("status") == "failed":
            continue
        baseline = class_tol.get(model, {})
        for m in ("accuracy", "precision", "recall", "f1"):
            if m in metrics and m in baseline:
                checks.append(
                    compare_metric(
                        f"{model}_{m}",
                        baseline[m],
                        metrics[m],
                        class_tol.get("absolute_tolerance", 0.01),
                    )
                )

    overall = "PASS" if all(c.get("status") == "PASS" for c in checks) else "FAIL"
    return {"overall_status": overall, "checks": checks}
