"""Smoke E2E do pipeline modular (preprocess → clustering → classification)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tests.conftest import apply_fast_classification

from pns_health_research.modules.classification.pipeline_runner import run_classification_pipeline
from pns_health_research.modules.clustering.pipeline_runner import run_clustering_pipeline
from pns_health_research.config.profile_resolver import resolve_module_config
from pns_health_research.core.paths import PROJECT_ROOT
from pns_health_research.persistence.base_store import (
    load_registry,
    resolve_clustering_source,
    write_classification_base,
    write_clustering_base,
    write_preprocessing_base,
)
from pns_health_research.modules.preprocessing.pipelines import run_preprocessing


def test_modular_pipeline_flow(synthetic_raw_path: Path, monkeypatch_bases_dir: Path):
    base_name = "pns"

    pp_cfg = resolve_module_config("pns", "preprocessing")
    pp_result = run_preprocessing(pp_cfg, raw_path=str(synthetic_raw_path))
    write_preprocessing_base(pp_result, pp_cfg, base_name=base_name, profile_kind="pns")

    cl_cfg = resolve_module_config("pns", "clustering")
    cl_result = run_clustering_pipeline(cl_cfg, base_name=base_name, source_base=base_name)
    write_clustering_base(
        base_name=base_name,
        result=cl_result,
        config=cl_cfg,
        source_base=base_name,
        profile_kind="pns",
    )

    cl_ref = resolve_clustering_source(base_name)
    clustered = pd.read_parquet(cl_ref.categorical_path)
    assert "Cluster" in clustered.columns

    cf_cfg = apply_fast_classification(
        resolve_module_config("pns", "classification")
    )
    cf_result = run_classification_pipeline(cf_cfg, base_name=base_name, source_base=base_name)
    clf_out = write_classification_base(
        cf_result,
        cf_cfg,
        base_name=base_name,
        source_base=base_name,
        profile_kind="pns",
    )

    assert (clf_out / "metrics/classification.json").exists()
    registry = load_registry(base_name)
    assert set(registry["modules"]) == {"preprocessing", "clustering", "classification"}

    manifest = json.loads((clf_out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["profile_kind"] == "pns"
