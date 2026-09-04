"""Smoke: artefatos materializados em data/bases/ (sem reexecutar pipeline)."""

from __future__ import annotations

import json

import pytest

from pns_health_research.core.paths import BASES_DIR, RAW_DATA_DIR


@pytest.mark.regression_smoke
def test_legacy_base_clustering_metrics():
    raw = RAW_DATA_DIR / "pns2019.parquet"
    if not raw.exists():
        pytest.skip("PNS real não disponível")

    metrics_path = BASES_DIR / "legacy" / "clustering" / "metrics" / "clustering.json"
    if not metrics_path.exists():
        pytest.skip("Base legacy/clustering não materializada — rode run_experiment.sh --profile legacy")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert abs(float(metrics.get("silhouette", 0)) - 0.514) < 0.02
    assert int(metrics.get("n_clusters", 0)) == 6


@pytest.mark.regression_smoke
def test_legacy_base_classification_models_present():
    raw = RAW_DATA_DIR / "pns2019.parquet"
    if not raw.exists():
        pytest.skip("PNS real não disponível")

    metrics_path = BASES_DIR / "legacy" / "classification" / "metrics" / "classification.json"
    if not metrics_path.exists():
        pytest.skip("Base legacy/classification não materializada")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for model in ("random_forest", "mlp", "xgboost"):
        row = metrics.get(model, {})
        assert isinstance(row, dict) and row.get("status") != "failed"
        assert float(row.get("f1", 0)) > 0.5


@pytest.mark.regression_smoke
def test_pns_base_clustering_metrics():
    raw = RAW_DATA_DIR / "pns2019.parquet"
    if not raw.exists():
        pytest.skip("PNS real não disponível")

    metrics_path = BASES_DIR / "pns" / "clustering" / "metrics" / "clustering.json"
    if not metrics_path.exists():
        pytest.skip("Base pns/clustering não materializada — rode run_experiment.sh --profile pns")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert int(metrics.get("n_clusters", 0)) == 6
    assert float(metrics.get("silhouette", 0)) > 0


@pytest.mark.regression_smoke
def test_pns_base_classification_models_present():
    raw = RAW_DATA_DIR / "pns2019.parquet"
    if not raw.exists():
        pytest.skip("PNS real não disponível")

    metrics_path = BASES_DIR / "pns" / "classification" / "metrics" / "classification.json"
    if not metrics_path.exists():
        pytest.skip("Base pns/classification não materializada")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "random_forest" in metrics
    row = metrics["random_forest"]
    assert isinstance(row, dict) and row.get("status") != "failed"
