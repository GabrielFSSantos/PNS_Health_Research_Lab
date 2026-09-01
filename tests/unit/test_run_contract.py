"""Testes do contrato v2 de runs."""

from pns_health_research.config.models import ExperimentConfig, ExperimentMeta, RunMeta
from pns_health_research.persistence.run_contract import (
    build_manifest_v2,
    build_summary,
    infer_run_type,
    migrate_manifest_v1,
)


def _cfg(stages: list[str] | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentMeta(name="test"),
        run=RunMeta(name="Test Run", tags=["unit"]),
        stages=stages or ["preprocessing", "clustering", "classification"],
    )


def test_infer_run_type():
    assert infer_run_type(_cfg(["classification"])) == "classification"
    assert infer_run_type(_cfg(["preprocessing", "clustering"])) == "clustering"
    assert infer_run_type(_cfg()) == "experiment"


def test_manifest_v2_fields():
    cfg = _cfg()
    manifest = build_manifest_v2(
        "run123", cfg, "completed", "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "fp"
    )
    assert manifest["schema_version"] == 2
    assert manifest["name"] == "Test Run"
    assert manifest["type"] == "experiment"
    assert manifest["tags"] == ["unit"]


def test_build_summary_primary_metric():
    cfg = _cfg()
    summary = build_summary(
        "run123",
        cfg,
        "completed",
        preprocessing_metrics={"rows_out": 100, "binary_columns": 10},
        classification_metrics={
            "xgboost": {"f1_positive": 0.8},
            "mlp": {"f1_positive": 0.75},
        },
    )
    assert summary["primary_metric"]["value"] == 0.8
    assert summary["primary_metric"]["model"] == "xgboost"


def test_migrate_manifest_v1():
    old = {"experiment_name": "legacy", "status": "completed", "source_run_id": "abc"}
    migrated = migrate_manifest_v1(old)
    assert migrated["name"] == "legacy"
    assert migrated["lineage"]["clustering"]["run_id"] == "abc"
