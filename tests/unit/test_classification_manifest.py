"""Testes do manifest e registry de classificação."""

from __future__ import annotations

import json
from pathlib import Path

from tests.conftest import apply_fast_classification

from pns_health_research.modules.classification.pipeline_runner import run_classification_pipeline
from pns_health_research.modules.clustering.pipeline_runner import run_clustering_pipeline
from pns_health_research.config.profile_resolver import resolve_module_config
from pns_health_research.core.paths import PROJECT_ROOT
from pns_health_research.persistence.base_store import (
    load_registry,
    write_classification_base,
    write_clustering_base,
    write_preprocessing_base,
)
from pns_health_research.modules.preprocessing.pipelines import run_preprocessing


def test_classification_base_artifacts(synthetic_raw_path: Path, monkeypatch_bases_dir: Path):
    pp_cfg = resolve_module_config("pns", "preprocessing")
    pp_result = run_preprocessing(pp_cfg, raw_path=str(synthetic_raw_path))
    write_preprocessing_base(pp_result, pp_cfg, base_name="testbase", profile_kind="pns")

    cl_cfg = resolve_module_config("pns", "clustering")
    cl_result = run_clustering_pipeline(cl_cfg, base_name="testbase", source_base="testbase")
    write_clustering_base(
        base_name="testbase",
        result=cl_result,
        config=cl_cfg,
        source_base="testbase",
        profile_kind="pns",
    )

    cf_cfg = apply_fast_classification(
        resolve_module_config("pns", "classification")
    )
    cf_result = run_classification_pipeline(cf_cfg, base_name="testbase", source_base="testbase")
    out = write_classification_base(
        cf_result,
        cf_cfg,
        base_name="testbase",
        source_base="testbase",
        entry_config_path="configs/profiles/pns.yaml#classification",
        profile_kind="pns",
    )

    assert out == monkeypatch_bases_dir / "testbase" / "classification"
    assert (out / "fingerprint.json").exists()
    assert (out / "provenance.json").exists()
    assert (out / "metrics/classification.json").exists()

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["module"] == "classification"
    assert manifest["profile_kind"] == "pns"
    assert "random_forest" in manifest["models"]

    registry = load_registry("testbase")
    assert registry["modules"]["classification"]["run_count"] == 1

    write_classification_base(
        cf_result,
        cf_cfg,
        base_name="testbase",
        source_base="testbase",
        profile_kind="pns",
    )
    assert load_registry("testbase")["modules"]["classification"]["run_count"] == 2
