"""Integração: preprocess + clustering na mesma base."""

from __future__ import annotations

import json
from pathlib import Path

from pns_health_research.modules.clustering.pipeline_runner import run_clustering_pipeline
from pns_health_research.config.profile_resolver import resolve_module_config
from pns_health_research.core.paths import PROJECT_ROOT
from pns_health_research.persistence.base_store import (
    load_registry,
    write_clustering_base,
    write_preprocessing_base,
)
from pns_health_research.modules.preprocessing.pipelines import run_preprocessing


def test_preprocess_then_clustering_same_base(synthetic_raw_path: Path, monkeypatch_bases_dir: Path):
    pp_cfg = resolve_module_config("pns", "preprocessing")
    pp_result = run_preprocessing(pp_cfg, raw_path=str(synthetic_raw_path))
    write_preprocessing_base(pp_result, pp_cfg, base_name="pns")

    cl_cfg = resolve_module_config("pns", "clustering")
    cl_result = run_clustering_pipeline(cl_cfg, base_name="pns", source_base="pns")
    out = write_clustering_base(
        base_name="pns",
        result=cl_result,
        config=cl_cfg,
        source_base="pns",
        entry_config_path="configs/profiles/pns.yaml#clustering",
        profile_kind="pns",
    )

    assert (out / "clustered_binary.parquet").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_base"] == "pns"
    assert manifest["profile_kind"] == "pns"

    registry = load_registry("pns")
    assert "preprocessing" in registry["modules"]
    assert "clustering" in registry["modules"]
