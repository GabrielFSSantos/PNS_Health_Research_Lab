"""Integração: preprocess + clustering + classification na mesma base."""

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


def test_preprocess_cluster_classify_same_base(
    synthetic_raw_path: Path, monkeypatch_bases_dir: Path
):
    pp_cfg = resolve_module_config("pns", "preprocessing")
    pp_result = run_preprocessing(pp_cfg, raw_path=str(synthetic_raw_path))
    write_preprocessing_base(pp_result, pp_cfg, base_name="pns", profile_kind="pns")

    cl_cfg = resolve_module_config("pns", "clustering")
    cl_result = run_clustering_pipeline(cl_cfg, base_name="pns", source_base="pns")
    write_clustering_base(
        base_name="pns",
        result=cl_result,
        config=cl_cfg,
        source_base="pns",
        profile_kind="pns",
    )

    cf_cfg = apply_fast_classification(
        resolve_module_config("pns", "classification")
    )
    cf_result = run_classification_pipeline(cf_cfg, base_name="pns", source_base="pns")
    out = write_classification_base(
        cf_result,
        cf_cfg,
        base_name="pns",
        source_base="pns",
        profile_kind="pns",
    )

    assert (out / "manifest.json").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile_kind"] == "pns"
    assert manifest["schema_version"] == 2

    registry = load_registry("pns")
    assert "preprocessing" in registry["modules"]
    assert "clustering" in registry["modules"]
    assert "classification" in registry["modules"]
    assert registry["modules"]["classification"]["profile_kind"] == "pns"
